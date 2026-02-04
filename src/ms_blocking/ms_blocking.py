from ms_blocking.utils import *  # noqa: F403

import networkx as nx


class BlockerNode:
    """Abstract class from which derive all classes in the module"""

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        self.blocking_columns = None
        self.equivalence_columns = None
        self.overlap_columns = None
        self.overlap = None
        self.normalize = None
        self.must_not_be_different = None
        self.word_level = None

    def __and__(self, other):
        if self == other:
            return self
        else:
            return merge_blockers(self, other)

    def __or__(self, other):
        if self == other:
            return self
        else:
            return OrNode(self, other)

    def __repr__(self):
        return f"Node{{{self.left}, {self.right}}}"

    def __eq__(self, other):
        if not isinstance(other, BlockerNode):
            return False
        else:
            return self.left == other.left and self.right == other.right


class AndNode(BlockerNode):
    """Used to compute the intersection of the outputs of two Blockers."""

    def __init__(self, left, right):
        super().__init__(left, right)

    def __repr__(self):
        return f"AndNode{{{self.left}, {self.right}}}"

    def block(self, df: pd.DataFrame, motives: bool = False) -> Coords:
        # In order not to perform redundant computations, we first filter out the rows that were not considered by the first blocker before running the second blocker
        coords_left = self.left.block(df, motives=motives)

        id_lists = (
            set(flatten(coords_left.keys()))
            if type(coords_left) is dict
            else set(flatten(coords_left))
        )
        df_shortened = (
            df[df.index.isin(id_lists)].copy()
            if id_lists
            else pd.DataFrame(columns=df.columns)
        )
        # Rows that are in no pairs following the first blocking step cannot be in any pair of the interection
        coords_right = self.right.block(df_shortened, motives=motives)

        result = merge_blocks_and(coords_left, coords_right)
        return result


class OrNode(BlockerNode):
    """Used to compute the union of the outputs of two Blockers."""

    def __init__(self, left, right):
        super().__init__(left, right)

    def __repr__(self):
        return f"OrNode{{{self.left}, {self.right}}}"

    def block(self, df: pd.DataFrame, motives: bool = False) -> Coords:
        # Note: for performance, it would be wise to remove rows that are already paired with all other rows, though this case should be pretty rare in real situations
        coords_left = self.left.block(df, motives=motives)

        coords_right = self.right.block(df, motives=motives)

        result = merge_blocks_or(coords_left, coords_right)
        return result


class AttributeEquivalenceBlocker(BlockerNode):  # Leaf
    """To regroup rows based on equality across columns."""

    def __init__(
        self,
        blocking_columns: str | Collection[str],
        must_not_be_different: str | Collection[str] = None,
        normalize_strings: bool = True,
    ):
        super().__init__()

        # Column(s) to block on
        if type(blocking_columns) is str:
            self.blocking_columns = [blocking_columns]
        else:
            self.blocking_columns = list(set(blocking_columns))  # Ensure no duplicates

        # Define it to block on a second column where NaNs are accepted
        if must_not_be_different is not None:
            if type(must_not_be_different) is str:
                must_not_be_different = [must_not_be_different]
            if len(must_not_be_different) > 1:
                raise ValueError("There must be only one extra column")
            elif (
                must_not_be_different
                and must_not_be_different[0] in self.blocking_columns
            ):
                raise ValueError("Cannot block twice on the same column")
            else:
                self.must_not_be_different = must_not_be_different
        else:
            self.must_not_be_different = []

        self.normalize = normalize_strings  # if True, will casefold+remove punctation+strip spaces for all strings before comparing them

    def __repr__(self):
        return f"AttributeEquivalenceBlocker({self.blocking_columns}{', ' + str(self.must_not_be_different) if self.must_not_be_different else ''}{', NON-NORMALIZED' if not self.normalize else ''})"

    def __eq__(self, other):
        if type(other) is AttributeEquivalenceBlocker:
            return (
                set(self.blocking_columns) == set(other.blocking_columns)
                and self.must_not_be_different == other.must_not_be_different
                and self.normalize == other.normalize
            )
        elif type(other) is MixedBlocker:
            return (
                set(self.blocking_columns) == set(other.equivalence_columns)
                and not other.overlap_columns
                and self.must_not_be_different == other.must_not_be_different
                and self.normalize == other.normalize
            )
        else:
            return False

    def block(self, data: pd.DataFrame, motives: bool = False) -> Coords:
        """Regroup rows based on equality of one or more columns"""

        print("Processing", self)

        temp_data = data.copy()

        for col in self.blocking_columns:
            if self.normalize:
                temp_data[col] = temp_data[col].apply(normalize)
        temp_data = temp_data.dropna(subset=self.blocking_columns)
        temp_data = remove_rows_if_value_appears_only_once(
            temp_data, self.blocking_columns
        )

        if len(temp_data) == 0:  # No pairs
            if motives:
                return dict()
            else:
                return set()

        if self.must_not_be_different:  # Perform a second row of blocking on a new attribute, but this time, NaNs validate the blocking condition
            temp_data = must_not_be_different_apply(
                temp_data,
                blocking_columns=self.blocking_columns,
                must_not_be_different_columns=self.must_not_be_different,
            )

            if len(temp_data) == 0:  # No pairs
                if motives:
                    return dict()
                else:
                    return set()

        # Use the DataFrame index for grouping and forming pairs
        # Using frozenset since they are ahshable and thus can be used as dictionary keys
        groups = temp_data.groupby(
            self.blocking_columns + self.must_not_be_different
        ).apply(lambda x: frozenset(x.index), include_groups=False)
        coords = {
            frozenset(pair)
            for group_list in groups
            for pair in combinations(group_list, 2)
        }

        if motives:
            explanations = [EquivalenceMotive(col) for col in self.blocking_columns]
            return add_motives_to_coords(coords, explanations)
        else:
            return set(coords)  # set is unnnecessary


class OverlapBlocker(BlockerNode):  # Leaf
    """To regroup rows based on overlap of one or more columns."""

    def __init__(
        self,
        blocking_columns: str | Collection[str],
        overlap: int = 1,
        word_level: bool = False,
        normalize_strings: bool = True,
    ):
        super().__init__()

        # Column(s) to block on
        if type(blocking_columns) is str:
            self.blocking_columns = [blocking_columns]
        else:
            self.blocking_columns = list(set(blocking_columns))  # Ensure no duplicates

        # Minimum overlap between objects for blocking
        if overlap > 0:
            self.overlap = overlap
        else:
            raise ValueError("Overlap must be greater than 0")

        self.word_level = word_level  # if True, will compare each space-separated word instead of each element of a list
        self.normalize = normalize_strings  # if True, will casefold+remove punctation+strip spaces for all strings before comparing them

    def __repr__(self):
        return f"OverlapBlocker({self.blocking_columns}, {self.overlap}{', WORD-LEVEL' if self.word_level else ''}{', NON-NORMALIZED' if not self.normalize else ''})"

    def __eq__(self, other):
        if type(other) is OverlapBlocker:
            return (
                set(self.blocking_columns) == set(other.blocking_columns)
                and self.normalize == other.normalize
                and self.overlap == other.overlap
                and self.word_level == other.word_level
            )
        elif type(other) is MixedBlocker:
            return (
                set(self.blocking_columns) == set(other.overlap_columns)
                and not other.equivalence_columns
                and self.normalize == other.normalize
                and self.word_level == other.word_level
                and self.overlap == other.overlap
            )
        else:
            return False

    def block(self, data: pd.DataFrame, motives: bool = False) -> Coords:
        """Regroup rows based on overlap of one or more columns"""

        print("Processing", self)

        temp_data = data.copy()

        temp_data = temp_data[self.blocking_columns].copy()

        for col in self.blocking_columns:
            temp_data[col] = temp_data[col].apply(
                parse_list, word_level=self.word_level
            )
            temp_data = temp_data.explode(col)
            if self.normalize:
                temp_data[col] = temp_data[col].apply(normalize)
        temp_data = temp_data.dropna(
            subset=self.blocking_columns
        )  # Remove empty objects
        temp_data = remove_rows_if_value_appears_only_once(
            temp_data, self.blocking_columns
        )

        if len(temp_data) == 0:  # No pairs fulfill any overlap
            if motives:
                return dict()
            else:
                return set()

        # Use the DataFrame index for grouping and forming pairs
        # Using frozenset since they are ahshable and thus can be used as dictionary keys
        groups = temp_data.groupby(self.blocking_columns).apply(
            lambda x: frozenset(x.index), include_groups=False
        )

        coords = block_overlap(groups=groups, overlap=self.overlap)

        if motives:
            explanations = [
                OverlapMotive(col, self.overlap, self.word_level)
                for col in self.blocking_columns
            ]
            return add_motives_to_coords(coords, explanations)
        else:
            return set(coords)


class MixedBlocker(BlockerNode):  # Leaf; For ANDs and RAM
    """Represent the intersection of an AttributeEquivalenceBlocker and an OverlapBlocker.
    Used for performance and RAM efficiency.
    """

    def __init__(
        self,
        equivalence_columns: str | Collection[str],
        overlap_columns: str | Collection[str],
        must_not_be_different: str | Collection[str] = None,
        overlap: int = 1,
        word_level: bool = False,
        normalize_strings: bool = True,
    ):
        super().__init__()

        # Column(s) to block on
        if type(equivalence_columns) is str:
            self.equivalence_columns = [equivalence_columns]
        else:
            self.equivalence_columns = list(
                set(equivalence_columns)
            )  # Ensure no duplicates

        if type(overlap_columns) is str:
            self.overlap_columns = [overlap_columns]
        else:
            self.overlap_columns = list(set(overlap_columns))  # Ensure no duplicates

        # Minimum overlap between objects for blocking
        if overlap > 0:
            self.overlap = overlap
        else:
            raise ValueError("Overlap must be greater than 0")

        # Define it to block on a second column where NaNs are accepted
        if must_not_be_different is not None:
            if type(must_not_be_different) is str:
                must_not_be_different = [must_not_be_different]
            if len(must_not_be_different) > 1:
                raise ValueError("There must be only one extra column")
            elif (
                must_not_be_different
                and must_not_be_different[0]
                in self.equivalence_columns + self.overlap_columns
            ):
                raise ValueError("Cannot block twice on the same column")
            else:
                self.must_not_be_different = must_not_be_different
        else:
            self.must_not_be_different = []

        self.word_level = word_level  # if True, will compare each space-separated word instead of each element of a list
        self.normalize = normalize_strings  # if True, will casefold+remove punctation+strip spaces for all strings before comparing them

    def __repr__(self):
        return str(
            AndNode(
                AttributeEquivalenceBlocker(
                    self.equivalence_columns, self.must_not_be_different, self.normalize
                ),
                OverlapBlocker(
                    self.overlap_columns, self.overlap, self.word_level, self.normalize
                ),
            )
        )

    def __eq__(self, other):
        if type(other) is AttributeEquivalenceBlocker:
            return (
                set(self.equivalence_columns) == set(other.blocking_columns)
                and self.normalize == other.normalize
                and self.must_not_be_different == other.must_not_be_different
            )
        elif type(other) is OverlapBlocker:
            return (
                set(self.overlap_columns) == set(other.blocking_columns)
                and self.normalize == other.normalize
                and self.overlap == other.overlap
                and self.word_level == other.word_level
            )
        elif type(other) is MixedBlocker:
            return (
                set(self.equivalence_columns) == set(other.equivalence_columns)
                and set(self.overlap_columns) == set(other.overlap_columns)
                and self.must_not_be_different == other.must_not_be_different
                and self.normalize == other.normalize
                and self.word_level == other.word_level
                and self.overlap == other.overlap
            )
        else:
            return False

    def block(self, data: pd.DataFrame, motives: bool = False) -> Coords:
        """Regroup rows based on overlap of one or more columns"""

        print("Processing", self)

        total_columns = self.equivalence_columns + self.overlap_columns

        temp_data = data[total_columns].copy()

        for col in total_columns:
            if col in self.equivalence_columns:
                temp_data[col] = temp_data[col].apply(normalize)
            elif col in self.overlap_columns:
                temp_data[col] = temp_data[col].apply(
                    lambda x: [
                        normalize(item) for item in parse_list(x, self.word_level)
                    ]
                    if self.normalize
                    else parse_list(x, self.word_level)
                )
                temp_data = temp_data.explode(col)

        temp_data = temp_data.dropna(subset=total_columns)  # Remove empty objects
        temp_data = remove_rows_if_value_appears_only_once(temp_data, total_columns)

        if len(temp_data) == 0:  # No pairs fulfill any overlap
            if motives:
                return dict()
            else:
                return set()

        if self.must_not_be_different:  # Perform a second row of blocking on a new attribute, but this time, NaNs validate the blocking condition
            temp_data = must_not_be_different_apply(
                temp_data,
                blocking_columns=total_columns,
                must_not_be_different_columns=self.must_not_be_different,
            )

        # Using frozenset since they are ahshable and thus can be used as dictionary keys
        groups_equivalence = temp_data.groupby(self.equivalence_columns).apply(
            lambda x: frozenset(x.index), include_groups=False
        )
        groups_overlap = temp_data.groupby(self.overlap_columns).apply(
            lambda x: frozenset(x.index), include_groups=False
        )

        coords_equivalence = {
            frozenset(pair)
            for group_list in groups_equivalence
            for pair in combinations(group_list, 2)
        }

        coords_overlap = block_overlap(groups=groups_overlap, overlap=self.overlap)

        coords = coords_equivalence.intersection(coords_overlap)

        if motives:
            explanations = [
                EquivalenceMotive(col) for col in self.equivalence_columns
            ] + [
                OverlapMotive(col, self.overlap, self.word_level)
                for col in self.overlap_columns
            ]

            return add_motives_to_coords(coords, explanations)
        else:
            return set(coords)


def add_blocks_to_dataset(
    data: pd.DataFrame,
    coords: Coords,
    sort: bool = True,
    keep_ungrouped_rows: bool = False,
    merge_blocks: bool = True,
    motives: bool = False,
    show_as_pairs: bool = False,
    output_columns: Columns = None,
    score: bool = False,
) -> pd.DataFrame:
    """Returns the intersection of an array of links

    Takes two lists of paired elements, with or without motives, returns their intersection

    Parameters
    ----------
       data : DataFrame
           DataFrame for blocking
       coords : Array
           Blocked coordinates
       sort : bool
           Whether to sort the result by block, thereby regrouping rows of the same block
       keep_ungrouped_rows : bool
           Whether to display rows that do not belong to any block
       merge_blocks : bool
           Whether to merge transitively merge blocks
       motives : bool
           Whether to display the reason behind each block
       show_as_pairs : bool
           Whether to show the output as pairs or rows rather than simply reordering the initial DataFrame
       output_columns : list
           Columns to show. Useful in combination with show_as_pairs as column names are altered
       score : bool
           Whether to show a score (computed from the number of motives)

    Returns
    -------
    DataFrame
      Blocked DataFrame

    Examples
    --------
    >>> add_blocks_to_dataset(data=pd.DataFrame(
       [
           [0, 'first', 4],
           [1, 'second', 6],
           [2, 'first', 2],
           [3, 'third', 5]
       ],
       columns=['id', 'rank', 'score']),
       coords=np.array([{0, 2}]),
       show_as_pairs=True,
       output_columns=['id', 'rank'])
        id_l rank_l  id_r rank_r  block
       0     0  first     2  first      0
    """

    if show_as_pairs and keep_ungrouped_rows:
        raise ValueError("Cannot both return pairs and keep ungrouped rows")

    if motives:
        if type(coords) is not dict:
            raise TypeError("Cannot specify 'motives=True' without passing motives")

    # Ensure the index is a unique identifier
    if not data.index.is_unique:
        raise ValueError("DataFrame index must be unique to be used as an identifier.")

    if score and not motives:
        raise ValueError("Cannot specify 'score=True' without passing motives")

    if "_motive" in data.columns:
        if motives:
            raise ValueError(
                "Please rename existing '_motive' column OR do not pass 'motives=True'"
            )

    if "score" in data.columns:
        if score:
            raise ValueError(
                "Please rename existing '_score' column OR do not pass 'score=True'"
            )

    if "_block" in data.columns:
        raise ValueError("Please rename existing '_block' column")

    if output_columns is None:
        output_columns = data.columns

    data = data[output_columns].copy()

    if len(coords) == 0 and not keep_ungrouped_rows:  # Empty graph
        if show_as_pairs:
            columns = [col + "_l" for col in data.columns] + [
                col + "_r" for col in data.columns
            ]
            output_data = pd.DataFrame(columns=columns)
        else:
            output_data = pd.DataFrame(columns=data.columns)

        if motives:
            output_data["_motive"] = ""
        if score:
            output_data["_score"] = 0
        output_data["_block"] = -1

    else:
        output_data = data
        # Map coords to connected component labels
        if merge_blocks:  # We solve the connected components problem
            cc_labels = solve_connected_components_from_coords(coords)
            # Match original index to new block ID
            matcher = {
                idx: label
                for idx, label in enumerate(cc_labels)
                if label != -1 and idx in data.index
            }
        else:  # We solve the cliques problem
            g = nx.Graph()
            # noinspection PyTypeChecker
            g.add_edges_from(coords)
            complete_subgraphs = list(nx.find_cliques(g))
            complete_subgraphs = sorted(complete_subgraphs)
            # matcher = {row_id:([i for i in range(len(complete_subgraphs)) if row_id in complete_subgraphs[i]]) for row_id in set(flatten(complete_subgraphs))}
            matcher = dict()
            for i, clique in enumerate(complete_subgraphs):
                for node_idx in clique:
                    if node_idx in matcher.keys():
                        matcher[node_idx].append(i)
                    else:
                        matcher[node_idx] = [i]

        if show_as_pairs:
            output_data = pd.DataFrame()
            for pair in coords:
                left_row = data.loc[[tuple(pair)[0]]].copy()
                current_index = left_row.index
                right_row = data.loc[[tuple(pair)[1]]].copy()
                left_row.columns = [col + "_l" for col in left_row.columns]
                right_row.columns = [col + "_r" for col in right_row.columns]
                current_row = pd.concat(
                    [left_row.reset_index(drop=True), right_row.reset_index(drop=True)],
                    axis=1,
                )
                current_row.index = current_index
                if motives:
                    motives_solved = solve_motives(coords[pair])
                    current_row["_motive"] = [list(map(str, motives_solved))]
                    if score:
                        current_row["_score"] = len(
                            motives_solved
                        )  # Score is simply the number of non-redundant motives
                output_data = pd.concat([output_data, current_row])

        # Assign blocks to rows based on their original index
        output_data["_block"] = output_data.index.map(matcher)
        if not merge_blocks:
            output_data = output_data.explode("_block")

        if keep_ungrouped_rows:
            output_data["_block"] = output_data["_block"].fillna(-1)
            matcher_ungrouped_rows = {}
            block_temp = []
            i = 0  # Track # of blocks processed
            for b in output_data["_block"]:
                if b == -1:
                    block_temp.append(i)
                    i += 1
                elif b not in matcher_ungrouped_rows:
                    matcher_ungrouped_rows[b] = i
                    block_temp.append(i)
                    i += 1
                else:
                    block_temp.append(matcher_ungrouped_rows[b])
            output_data["_block"] = block_temp
        else:
            if not show_as_pairs:
                output_data = output_data[
                    output_data["_block"].duplicated(keep=False)
                    & output_data["_block"].notna()
                ]

        output_data.loc[:, ["_block"]] = start_from_zero(output_data["_block"])

        if sort:
            # Sort by block, then by original index
            sort_cols = ["_block"]
            if output_data.index.name:
                output_data = output_data.sort_values(
                    sort_cols + [output_data.index.name]
                )
            else:
                # If no named index, use the first column of the DataFrame
                output_data = output_data.reset_index()
                output_data = output_data.sort_values(
                    sort_cols + [output_data.columns[0]]
                )
                output_data = output_data.set_index(output_data.columns[0])

        if not show_as_pairs and motives:
            id_list = flatten(coords.keys())
            motive_matcher = {
                row_id: list(map(str, solve_motives(coords[pair])))
                for pair in coords.keys()
                for row_id in id_list
                if row_id in pair
            }
            # noinspection PyTypeChecker
            output_data["_motive"] = output_data.index.map(motive_matcher)
            if score:
                output_data["_score"] = 0
                score_matcher = {  # Horribly repetitive
                    row_id: len(solve_motives(coords[pair]))
                    for pair in coords.keys()
                    for row_id in id_list
                    if row_id in pair
                }
                output_data["_score"] = output_data.index.map(score_matcher)

    output_data = output_data.reset_index(drop=True)
    output_data["_block"] = output_data["_block"].astype(int)

    return output_data


def generate_blocking_report(
    data: pd.DataFrame, coords: Coords, output_columns: Collection[str] = None
) -> pd.DataFrame:
    """
    Shorthand for add_blocks_to_dataset with below arguments
    """
    return add_blocks_to_dataset(
        data,
        coords,
        sort=True,
        merge_blocks=False,
        motives=True,
        show_as_pairs=True,
        output_columns=output_columns,
    )


def merge_blockers(
    left: BlockerNode, right: BlockerNode
) -> AttributeEquivalenceBlocker | OverlapBlocker | MixedBlocker | AndNode:
    """Convert two blockers into a single one for performance purposes

    This function outputs a new blocker that combines the functionalities of the two input blockers, to prevent redundant operations.

    Parameters
    ----------
    left : BlockerNode
      Blocker that represents the first condition

    right : BlockerNode
      Blocker that represents the second condition

    Returns
    -------
    AttributeEquivalenceBlocker|OverlapBlocker|MixedBlocker|AndNode
      Blocker that represents both conditions
    """
    if (
        type(left) is AttributeEquivalenceBlocker
        and type(right) is AttributeEquivalenceBlocker
        and left.normalize == right.normalize
        and left.must_not_be_different == right.must_not_be_different
    ):
        return AttributeEquivalenceBlocker(
            blocking_columns=left.blocking_columns + right.blocking_columns,
            normalize_strings=left.normalize,
            must_not_be_different=left.must_not_be_different,
        )

    elif (
        type(left) is OverlapBlocker
        and type(right) is OverlapBlocker
        and left.normalize == right.normalize
        and left.overlap == right.overlap
        and left.word_level == right.word_level
    ):
        return OverlapBlocker(
            blocking_columns=left.blocking_columns + right.blocking_columns,
            normalize_strings=left.normalize,
            overlap=left.overlap,
            word_level=left.word_level,
        )

    elif (
        type(left) is AttributeEquivalenceBlocker
        and type(right) is OverlapBlocker
        and left.normalize == right.normalize
    ):
        return MixedBlocker(
            equivalence_columns=left.blocking_columns,
            overlap_columns=right.blocking_columns,
            normalize_strings=left.normalize,
            overlap=right.overlap,
            word_level=right.word_level,
        )

    elif (
        type(left) is OverlapBlocker
        and type(right) is AttributeEquivalenceBlocker
        and left.normalize == right.normalize
    ):
        return MixedBlocker(
            equivalence_columns=right.blocking_columns,
            overlap_columns=left.blocking_columns,
            normalize_strings=left.normalize,
            overlap=left.overlap,
            word_level=left.word_level,
        )

    elif (
        type(left) is MixedBlocker
        and type(right) is MixedBlocker
        and left.normalize == right.normalize
        and left.overlap == right.overlap
        and left.word_level == right.word_level
    ):
        return MixedBlocker(
            equivalence_columns=left.equivalence_columns + right.equivalence_columns,
            overlap_columns=left.overlap_columns + right.overlap_columns,
            must_not_be_different=list(
                set(left.must_not_be_different + right.must_not_be_different)
            ),
            normalize_strings=left.normalize,
            overlap=left.overlap,
            word_level=left.word_level,
        )

    elif (
        type(left) is MixedBlocker
        and type(right) is AttributeEquivalenceBlocker
        and left.normalize == right.normalize
    ):
        return MixedBlocker(
            equivalence_columns=left.equivalence_columns + right.blocking_columns,
            overlap_columns=left.overlap_columns,
            must_not_be_different=list(
                set(left.must_not_be_different + right.must_not_be_different)
            ),
            normalize_strings=left.normalize,
            overlap=left.overlap,
            word_level=left.word_level,
        )

    elif (
        type(left) is AttributeEquivalenceBlocker
        and type(right) is MixedBlocker
        and left.normalize == right.normalize
    ):
        return MixedBlocker(
            equivalence_columns=left.blocking_columns + right.equivalence_columns,
            overlap_columns=right.overlap_columns,
            must_not_be_different=list(
                set(left.must_not_be_different + right.must_not_be_different)
            ),
            normalize_strings=left.normalize,
            overlap=right.overlap,
            word_level=right.word_level,
        )

    elif (
        type(left) is MixedBlocker
        and type(right) is OverlapBlocker
        and left.normalize == right.normalize
        and left.overlap == right.overlap
        and left.word_level == right.word_level
    ):
        return MixedBlocker(
            equivalence_columns=left.equivalence_columns,
            overlap_columns=left.overlap_columns + right.blocking_columns,
            must_not_be_different=left.must_not_be_different,
            normalize_strings=left.normalize,
            overlap=left.overlap,
            word_level=left.word_level,
        )

    elif (
        type(left) is OverlapBlocker
        and type(right) is MixedBlocker
        and left.normalize == right.normalize
        and left.overlap == right.overlap
        and left.word_level == right.word_level
    ):
        return MixedBlocker(
            equivalence_columns=right.equivalence_columns,
            overlap_columns=left.blocking_columns + right.overlap_columns,
            must_not_be_different=right.must_not_be_different,
            normalize_strings=left.normalize,
            overlap=left.overlap,
            word_level=left.word_level,
        )
    else:
        return AndNode(left, right)


# TODO: deport logic in a way that enables .progress_apply
