import numpy as np
import ast
import re
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
import networkx as nx
import random
from collections import Counter

from itertools import combinations
from typing import List, Set, Iterable, Dict, Collection, Any

Columns = List[str]
Pair = Collection[int]
CoordsBasic = Set[Pair]
CoordsMotives = Dict[Pair, Set[str]]
Coords = CoordsBasic | CoordsMotives

_PUNCT_RE = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]')
_SPACE_RE = re.compile(r"\s+")


def remove_rows_if_value_appears_only_once(
    data: pd.DataFrame, cols: Columns
) -> pd.DataFrame:
    """Drop rows of a Pandas DataFrame where a certain column's values appears only once.

    Ensures all elements of provided columns appear at least twice in their column

    Parameters
    ----------
    data : DataFrame
      DataFrame to preprocess

    cols : List[str]
      List of columns where rows that contain non-duplicated elements shall be discarded

    Returns
    -------
    DataFrame
      DataFrame with reduced number of rows

    Examples
    --------
    >>> remove_rows_if_value_appears_only_once(data, ['name', 'city'])
    """
    for col in cols:
        counts = data[col].map(data[col].value_counts())
        data = data[counts >= 2]
    return data


def start_from_zero(figures: Collection[int]) -> List[int]:
    """Turns a list of integers into a same-length list that starts at 0, without gaps

    Parameters
    ----------
    figures : Collection[int]
      List of integers

    Returns
    -------
    List[int]
      List of integers that starts at 0

    Examples
    --------
    >>> start_from_zero([1, 3, 7, 9, 10, 9, 8, 1, 5])
    [0, 1, 3, 5, 6, 5, 4, 0, 2]
    """
    temp = dict(zip(np.unique(figures), range(len(np.unique(figures)))))
    return [temp[f] for f in figures]


def solve_connected_components_from_coords(coords: Coords) -> Collection[Set[int]]:
    """Solves the connected components graph problem based on a set of links between nodes

    Turns a list of paired elements into a list of groups

    Parameters
    ----------
    coords : Coords
      List of pairs

    Returns
    -------
    Collection[Set[int]]
      List of transitively-linked elements

    Examples
    --------
    >>> solve_connected_components_from_coords({{1, 4}, {1, 5}, {6, 7}}))
    array({1, 4, 5}, {6, 7})
    """

    filtered_coords = [list(pair) for pair in coords if len(pair) == 2]

    if not filtered_coords:
        return np.array([])

    coords_array = np.array(filtered_coords)
    # Map original indices to integers
    unique_indices = np.unique(coords_array.flatten())
    index_to_int = {idx: i for i, idx in enumerate(unique_indices)}
    int_to_index = {i: idx for i, idx in enumerate(unique_indices)}
    mapped_coords = np.array(
        [[index_to_int[pair[0]], index_to_int[pair[1]]] for pair in filtered_coords]
    )
    n = len(unique_indices)

    graph = coo_matrix(
        (
            np.ones(len(mapped_coords), dtype=bool),
            (mapped_coords[:, 0], mapped_coords[:, 1]),
        ),
        shape=(n, n),
    )
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # Map labels back to original indices
    original_labels = np.full(np.max(unique_indices) + 1, -1, dtype=int)
    for i, original_idx in int_to_index.items():
        original_labels[original_idx] = labels[i]

    return original_labels


def normalize_function(string: Any) -> Any:
    """Normalize a single string or return original if not a string.

    Ensures successful comparison of different comparisons of the same string

    Parameters
    ----------
    string : Any
        Text to preprocess

    Returns
    -------
    Any
      Normalized string or original value if not a string

    Examples
    --------
    >>> normalize_function('I like_Music!!! :)')
    'i like music'
    """
    if not isinstance(string, str):
        return string

    string = _PUNCT_RE.sub(" ", string)
    string = _SPACE_RE.sub("", string)
    string = string.casefold().strip()
    return string


def normalize(text: Any) -> Any:
    """Normalize a string, an iterable of strings, or other types.

    Ensures successful comparison of different comparisons of the same string

    Parameters
    ----------
    text : Any
      Text(s) to preprocess

    Returns
    -------
    Any
      Normalized string, list of normalized strings, or original value if not applicable

    Examples
    --------
    >>> normalize('I like_Music!!! :)')
    'i like music'
    """
    if pd.isna(text):
        return text

    if isinstance(text, str):
        return normalize_function(text)
    elif isinstance(text, (list, tuple, set)):
        return [normalize_function(item) for item in text]
    else:
        return normalize_function(str(text))


def flatten(list_of_iterables_: Collection[Iterable]) -> List[Any] | None:
    """Returns a flattened list from a list of iterables

    The iterables may be lists, sets...

    Parameters
    ----------
    list_of_iterables_ : Collection[Iterable]
      List to flatten

    Returns
    -------
    List[Any]
      1-Dimensional list

    Examples
    --------
    >>> flatten([[1,2,3], [4,5], [6]])
    [1,2,3,4,5,6]
    >>> flatten([{1,2,3}, {4,5}, {6})
    [1,2,3,4,5,6]
    """
    try:
        return [element for iterable_ in list_of_iterables_ for element in iterable_]
    except TypeError:
        print("Argument must a list-like object")


def merge_blocks_or(coords_1: Coords, coords_2: Coords) -> Coords:
    """Returns the union of an array of links

    Takes two lists of paired elements, with or without motives, returns their union

    Parameters
    ----------
    coords_1 : Coords
      Array of coordinates

    coords_2 : Coords
      Array of coordinates

    Returns
    -------
    Coords
      Array of coordinates

    Examples
    --------
    >>> merge_blocks_or(np.array([{1, 4}, {1, 5}, {6, 7}], np.array([{4, 5}, {6, 7}, {2, 9}]))
    array({1, 4}, {1, 5}, {2, 9}, {4, 5}, {6, 7})
    """
    if type(coords_1) is type(coords_2) is dict:  # We have motives
        return {
            pair: (
                (coords_1[pair] | coords_2[pair])
                if (pair in coords_1 and pair in coords_2)
                else coords_1[pair]
                if (pair in coords_1)
                else coords_2[pair]
            )
            for y in (coords_1, coords_2)
            for pair in y.keys()
        }
    else:
        return coords_1.union(coords_2)


def merge_blocks_and(coords_1: Coords, coords_2: Coords) -> Coords:
    """Returns the intersection of an array of links

    Takes two lists of paired elements, with or without motives, returns their intersection

    Parameters
    ----------
    coords_1 : Coords
      Array of coordinates

    coords_2 : Coords
      Array of coordinates

    Returns
    -------
    Coords
      Array of coordinates

    Examples
    --------
    >>> merge_blocks_and(np.array([{1, 4}, {1, 5}, {6, 7}], np.array([{4, 5}, {6, 7}, {2, 9}]))
    array({1, 4})
    """
    if type(coords_1) is type(coords_2) is dict:  # We have motives
        return {
            pair: (coords_1[pair] | coords_2[pair])
            for y in (coords_1, coords_2)
            for pair in y.keys()
            if (pair in coords_1 and pair in coords_2)
        }
    else:
        return coords_1.intersection(coords_2)


def add_blocks_to_dataset(
    data: pd.DataFrame,
    coords: Coords,
    sort: bool = True,
    keep_ungrouped_rows: bool = False,
    merge_blocks: bool = True,
    motives: bool = False,
    show_as_pairs: bool = False,
    output_columns: Columns = None,
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
            raise TypeError("Cannot specify motives=True without passing motives")

    # Ensure the index is a unique identifier
    if not data.index.is_unique:
        raise ValueError("DataFrame index must be unique to be used as an identifier.")

    if "_motive" in data.columns:
        if motives:
            raise ValueError(
                "Please rename existing '_motive' column OR do not pass 'motives=True'"
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

    if motives:
        output_data["_motive"] = ""
        id_list = flatten(coords.keys())
        motive_matcher = {
            row_id: frozenset(
                reason
                for pair in coords.keys()
                if row_id in pair
                for reason in coords[pair]
            )
            for row_id in id_list
        }
        output_data["_motive"] = output_data.index.map(motive_matcher)

    if "_block" not in output_data.columns:  # Empty coords
        output_data["_block"] = -1

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


def parse_list(s: str | List, word_level: bool = False) -> List[str]:
    """Turns a stringified list into an actual python list, taking extra inner quotes into account

    Ensures compatibility across different data formats, including ones that do not natively support list or table data.

    Parameters
    ----------
    s : str
      Stringified representation of a list e.g. "['string 1', 'string 2', ...]"

    word_level : bool
      Whether to return a list of all words within s instead of a list of each comma-separated element

    Returns
    -------
    List[str]
      s turned into a List

    Examples
    --------
    >>> parse_list("['string 1', 'string 2', ...]")
    ['string 1', 'string 2', ...]
    >>> parse_list("['string 1', 'string 2', ...]", word_level=True)
    ['string', '1', 'string' '2', ...]
    """

    if type(s) is list:  # If we already have a list
        if len(s) == 1 and s[0][0] == "[" and s[0][-1] == "]":
            s = s[0]
        else:
            return s

    if pd.isna(s):
        return []

    s = str(s).strip()

    if not s:
        return []

    try:
        parts = ast.literal_eval(s)
    except ValueError:  # doesn't seem to be a stringified list
        parts = s.split("', '")

    cleaned_items = [str(part).strip().strip("''") for part in parts]

    if word_level:
        return [w for s in cleaned_items for w in s.split() if len(w) > 0]
    else:
        return [s for s in cleaned_items if len(s) > 0]


def scoring(data: pd.DataFrame, motives_column: str = "_motive") -> pd.Series:
    """Add a score to a blocked DataFrame based on the number of motives

    Parameters
    ----------
    data : DataFrame
      DataFrame with motives

    motives_column : str
      Name of the column containing the motives

    Returns
    -------
    Series[int]
      A column of scores
    """

    # Check that we do have motives
    if motives_column not in data.columns:
        if motives_column == "_motive":
            raise ValueError("No motives in DataFrame")
        else:
            raise ValueError(
                f'Specified motives column "{motives_column}" does not exist'
            )

    if "score" in data.columns:
        print("Renaming 'score' column to 'score_old'")
        data = data.rename(columns={"score": "score_old"})

    scores = data[motives_column].apply(len)
    return scores


def must_not_be_different_apply(  # WIP
    temp_data: pd.DataFrame,
    blocking_columns: List[str],
    must_not_be_different_columns: List[str],
):
    """Re-block DataFrame on a second column, where we require non-difference rather than equality

    Parameters
    ----------
    temp_data : DataFrame
      Partially blocked DataFrame

    blocking_columns : List[str]
      Columns where we check for equality

    must_not_be_different_columns : List[str]
        Columns where we only check for non-difference

    Returns
    -------
    DataFrame
      Column of scores
    """

    series_block_id = temp_data.groupby(blocking_columns).ngroup()
    temp_data = temp_data[series_block_id.duplicated(keep=False)]

    reconstructed_data = pd.DataFrame(columns=temp_data.columns)
    for block in series_block_id.unique():
        # noinspection PyArgumentList
        current_block = (
            temp_data[series_block_id == block]
            .sort_values(must_not_be_different_columns)
            .copy()
        )
        if (
            len(current_block[current_block[must_not_be_different_columns].notnull()])
            == 0
        ):  # All nulls
            random_string = "".join(
                random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=10)
            )  # As long as the string is not already in the column...
            # There must be a better way to do it...
            current_block[must_not_be_different_columns] = (
                current_block[must_not_be_different_columns]
                .astype(str)
                .fillna(random_string)
            )
        else:
            current_block[must_not_be_different_columns] = (
                current_block[must_not_be_different_columns].astype(str).ffill()
            )
        if len(reconstructed_data) == 0:
            reconstructed_data = current_block
        else:
            reconstructed_data = pd.concat([reconstructed_data, current_block])

    return reconstructed_data


def block_overlap(groups: Iterable, overlap: int = 1) -> Coords:
    """Block a DataFrame based on overlap accross columns

    Parameters
    ----------
    groups : Iterable
      Output of a groupby

    overlap : int
      Minimum passing overlap

    Returns
    -------
    Coords
      Pairs obtained by blocking
    """
    coords = {
        frozenset(pair) for group_list in groups for pair in combinations(group_list, 2)
    }

    if overlap > 1:
        coords = [  # In this specific case, we want to keep duplicates to track the number of occurences of each pair
            frozenset(pair)
            for group_list in groups
            for pair in combinations(group_list, 2)
        ]
        # Filter pairs that fulfill the minimum overlap condition
        occurences_dict = Counter(coords)
        coords = {
            p for p in occurences_dict if occurences_dict[p] >= overlap
        }  # The collection of pairs that fulfill the overlap condition

    return coords


def add_motives_to_coords(coords: Coords, explanations: Set[str]) -> CoordsMotives:
    """Block a DataFrame based on overlap accross columns

    Parameters
    ----------
    coords : Coords
      Coords obtained by blocking

    explanations : Set[str]
      Set of explanations

    Returns
    -------
    CoordsMotives
      Pairs obtained by blocking

    Examples
    --------
    >>> add_motives_to_coords({
        frozenset({1, 4}),
        frozenset({8, 11}),
        frozenset({2, 5}),
        frozenset({10, 13}),
        frozenset({3, 8}),
        frozenset({3, 11}),
    }, {"Same 'City'"}')
    {
        frozenset({1, 4}): {"Same 'City'"},
        frozenset({8, 11}): {"Same 'City'"},
        frozenset({2, 5}): {"Same 'City'"},
        frozenset({10, 13}): {"Same 'City'"},
        frozenset({3, 8}): {"Same 'City'"},
        frozenset({3, 11}): {"Same 'City'"},
    }
    """
    return {pair: explanations for pair in coords}
