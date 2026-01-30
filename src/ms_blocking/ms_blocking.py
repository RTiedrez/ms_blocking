from ms_blocking.utils import *  # noqa: F403


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


class AndNode(BlockerNode):
    """Used to compute the intersection of the outputs of two Blockers."""

    def __init__(self, left, right):
        super().__init__(left, right)

    def __repr__(self):
        return f"AndNode{{{self.left}, {self.right}}}"

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def block(self, df, motives=False):
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

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def block(self, df, motives=False):
        # Note: for performance, it would be wise to remove rows that are already paired with all other rows, though this case should be pretty rare in real situations
        coords_left = self.left.block(df, motives=motives)

        coords_right = self.right.block(df, motives=motives)

        result = merge_blocks_or(coords_left, coords_right)
        return result


class AttributeEquivalenceBlocker(BlockerNode):  # Leaf
    """To regroup rows based on equality across columns."""

    def __init__(
        self, blocking_columns, normalize_strings=True, must_not_be_different=None
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
        return f"AttributeEquivalenceBlocker({self.blocking_columns}, {self.must_not_be_different})"

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

    def block(self, data, motives=False):
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
            explanations = {
                f"Same '{column_name}'" for column_name in self.blocking_columns
            }
            return add_motives_to_coords(coords, explanations)
        else:
            return set(coords)  # set is unnnecessary


class OverlapBlocker(BlockerNode):  # Leaf
    """To regroup rows based on overlap of one or more columns."""

    def __init__(
        self, blocking_columns, overlap=1, word_level=False, normalize_strings=True
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
        return f"OverlapBlocker({self.blocking_columns}, {self.overlap})"

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

    def block(self, data, motives=False):
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
            explanations = {
                f">={self.overlap}{' word_level' if self.word_level else ''} overlap in '{column_name}'"
                for column_name in self.blocking_columns
            }
            return add_motives_to_coords(coords, explanations)
        else:
            return set(coords)


class MixedBlocker(BlockerNode):  # Leaf; For ANDs and RAM
    """Represent the intersection of an AttributeEquivalenceBlocker and an OverlapBlocker.
    Designed for performance and RAM efficiency.
    """

    def __init__(
        self,
        equivalence_columns,
        overlap_columns,
        must_not_be_different=None,
        overlap=1,
        word_level=False,
        normalize_strings=True,
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
        return f"MixedBlocker({self.equivalence_columns}, {self.overlap_columns}, {self.overlap})"

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

    def block(self, data, motives=False):
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
            explanations = {
                f"Same '{column_name}'" for column_name in self.equivalence_columns
            } | {
                f">={self.overlap}{' word_level' if self.word_level else ''} overlap in '{column_name}'"
                for column_name in self.overlap_columns
            }
            return add_motives_to_coords(coords, explanations)
        else:
            return set(coords)


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


# /!\ TODO: make class for motives (+ pair, motive dict)?
