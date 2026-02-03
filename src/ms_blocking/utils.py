import numpy as np
import ast
import re
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
import random
from collections import Counter

from itertools import combinations
from typing import List, Set, Iterable, Dict, Collection, Any


class EquivalenceMotive:
    def __init__(self, blocking_columns):
        self.blocking_columns = blocking_columns

    def __eq__(self, other):
        return self.blocking_columns == other.blocking_columns

    def __repr__(self):
        return ", ".join(
            [f"Same '{column_name}'" for column_name in self.blocking_columns]
        )


class OverlapMotive:
    def __init__(self, blocking_columns, overlap=1, word_level=False):
        self.blocking_columns = blocking_columns
        self.overlap = overlap
        self.word_level = word_level

    def __eq__(self, other):
        return (
            self.blocking_columns == other.blocking_columns
            and self.overlap == other.overlap
            and self.word_level == other.word_level
        )

    def __repr__(self):
        return ", ".join(
            [
                f">={self.overlap}{' word_level' if self.word_level else ''} overlap in '{column_name}'"
                for column_name in self.blocking_columns
            ]
        )


Columns = List[str]
Pair = Collection[int]
Motive = EquivalenceMotive | OverlapMotive
CoordsBasic = Set[Pair]
CoordsMotives = Dict[Pair, List[Motive]]
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
                (coords_1[pair] + coords_2[pair])
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
            pair: (coords_1[pair] + coords_2[pair])
            for y in (coords_1, coords_2)
            for pair in y.keys()
            if (pair in coords_1 and pair in coords_2)
        }
    else:
        return coords_1.intersection(coords_2)


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


def add_motives_to_coords(
    coords: Coords, explanations: List[Motive]
) -> Dict[Pair, List[Motive]]:
    """Block a DataFrame based on overlap accross columns

    Parameters
    ----------
    coords : Coords
      Coords obtained by blocking

    explanations : Set[EquivalenceMotive|OverlapMotive]
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


def solve_motives(motives: List[Motive]) -> List[Motive]:
    """Remove duplicated and redundant motives from a list of motives

    Redundant motives refer to OverlapMotives on the same column(s) but with different overlap or word-level condition

    Parameters
    ----------
    motives : List[Motive]
      Coords obtained by blocking

    Returns
    -------
    List[Motive]
      Pairs obtained by blocking

    Examples
    --------
    >>> solve_motives([OverlapMotive(['websites'], 1), OverlapMotive(['websites'], 2), OverlapMotive(['websites'], 2, word_level=False)])
    OverlapMotive(['websites'], 2, word_level=False)
    """
    if not motives:
        raise ValueError("Motives must not be empty")

    final_motives = [motives[0]]
    for motive in motives[1:]:
        if motive not in final_motives:
            final_motives.append(motive)
            if type(motive) is OverlapMotive:
                # Look for redundant motives
                for motive_to_compare in final_motives[:-1]:
                    if (
                        type(motive_to_compare) is OverlapMotive
                    ):  # With EquivalenceMotive, equality check suffices
                        if (
                            motive.blocking_columns
                            == motive_to_compare.blocking_columns
                        ):
                            if motive.word_level == motive_to_compare.word_level:
                                # Replace Blocker with the one with bigger overlap
                                if motive.overlap < motive_to_compare.overlap:
                                    final_motives.remove(motive)
                                    final_motives.append(motive_to_compare)
                                elif motive.overlap > motive.overlap:
                                    final_motives.remove(motive_to_compare)
                                    final_motives.append(motive)
                            elif motive.overlap == motive_to_compare.overlap:
                                # Replace Blocker with the one with stricter word/element-level condition
                                if (
                                    motive.word_level
                                    and not motive_to_compare.word_level
                                ):
                                    final_motives.remove(motive)
                                    final_motives.append(motive_to_compare)
                                elif (
                                    not motive.word_level
                                    and motive_to_compare.word_level
                                ):
                                    final_motives.remove(motive_to_compare)
                                    final_motives.append(motive)

    return final_motives
