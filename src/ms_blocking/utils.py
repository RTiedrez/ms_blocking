import ast
import re
from collections import Counter
from itertools import combinations
from typing import Any, Collection, Dict, Iterable, List, Set

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class EquivalenceMotive:
    def __init__(self, blocking_column: str):
        if not isinstance(blocking_column, str):
            raise TypeError("blocking_column for Motive must be a string")
        self.blocking_column = blocking_column

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EquivalenceMotive | OverlapMotive):
            raise TypeError("Can only compare Motives")
        return self.blocking_column == other.blocking_column

    def __str__(self):
        return f"Same '{self.blocking_column}'"

    def __repr__(self):
        return f"EquivalenceMotive('{self.blocking_column}')"


class OverlapMotive:
    def __init__(
        self, blocking_column: str, overlap: int = 1, word_level: bool = False
    ):
        if not isinstance(blocking_column, str):
            raise TypeError("blocking_column for Motive must be a string")
        if not isinstance(overlap, int):
            raise TypeError("overlap must be an int")
        if not isinstance(word_level, bool):
            raise TypeError("word_level must be a boolean")
        self.blocking_column = blocking_column
        self.overlap = overlap
        self.word_level = word_level

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EquivalenceMotive | OverlapMotive):
            raise TypeError("Can only compare Motives")
        return (
            self.blocking_column == other.blocking_column
            and self.overlap == other.overlap
            and self.word_level == other.word_level
        )

    def __str__(self):
        return f">={self.overlap}{' word-level' if self.word_level else ''} overlap in '{self.blocking_column}'"

    def __repr__(self):
        return f"OverlapMotive('{self.blocking_column}', {self.overlap}{', word_level=True' if self.word_level else ''})"


Columns = List[str]
Pair = Collection[int]
Motive = EquivalenceMotive | OverlapMotive
CoordsBasic = Set[Pair]
CoordsMotives = Dict[Pair, List[Motive]]
Coords = CoordsBasic | CoordsMotives

_PUNCT_RE = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]')
_SPACE_RE = re.compile(r"\s+")


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

    if pd.isna(text) if not isinstance(text, (list, tuple, set)) else False:
        return text

    if isinstance(text, str):
        return normalize_function(text)
    elif isinstance(text, (list, tuple, set)):
        # Filter out None and NaN values from the list first
        filtered_text = [s for s in text if s is not None and not pd.isna(s)]
        return [normalize_function(item) for item in filtered_text]
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
        raise TypeError("Argument must a list-like object")


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
    if isinstance(coords_1, dict) and isinstance(coords_2, dict):  # We have motives
        return {
            pair: (
                coords_1[pair] + coords_2[pair]
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
    if isinstance(coords_1, dict) and isinstance(coords_2, dict):  # We have motives
        return {
            pair: coords_1[pair] + coords_2[pair]
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
      Whether to return a list of all words within s instead of a list of each comma-separated element;
      Note that if passed a string that does not represent a list, this argument will be ignored and the function
      will return a list of each word in the string

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

    if isinstance(s, list):  # If we already have a list
        if (
            len(s) == 1 and str(s[0]).startswith("[") and str(s[0]).endswith("]")
        ):  # In case we have a stringified list INSIDE a normal list
            s = s[0]
        else:
            # Filter None and NaN
            return [item for item in s if item is not None and not pd.isna(item)]

    try:
        if pd.isna(s):
            return []
    except ValueError:  # We have an array
        pass

    s_str = str(s).strip()

    if not s_str:
        return []

    if s_str.startswith("[") and s_str.endswith("]"):  # Stringified list?
        try:
            parts = ast.literal_eval(s_str)
        # except ValueError:  # doesn't seem to be a stringified list
        #    parts = s_str.split("', '")
        except SyntaxError:  # In case we have a string surrounded by brackets
            parts = s_str.split()
    else:
        parts = s_str.split()

    # Filter None and NaN before converting to string
    filtered_parts = [p for p in parts if p is not None and not pd.isna(p)]

    cleaned_items = [str(part).strip().strip("''") for part in filtered_parts]

    if word_level:
        return [w for item in cleaned_items for w in item.split() if len(w) > 0]
    else:
        return [item for item in cleaned_items if len(item) > 0]


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
      A list of Motives whose length should be smaller or equal to the original list of motives

    Examples
    --------
    >>> solve_motives([OverlapMotive('websites', 1), OverlapMotive('websites', 2), OverlapMotive('websites', 2, word_level=False)])
    [OverlapMotive(['websites'], 2, word_level=False)]
    """
    if not motives:
        raise ValueError("Motives must not be empty")

    # split_motives = []
    # for motive in motives:
    #    split_motives += split_motive(motive)

    final_motives = [
        motive for motive in motives if isinstance(motive, EquivalenceMotive)
    ]  # With EquivalenceMotive, equality check suffices
    overlap_motives = [
        motive for motive in motives if isinstance(motive, OverlapMotive)
    ]
    overlap_columns = [motive.blocking_column for motive in overlap_motives]

    for column in overlap_columns:
        overlap_motives_for_column = [
            motive for motive in overlap_motives if motive.blocking_column == column
        ]

        # Select Blocker with stricter word/element-level condition
        word_level_motives_for_column = [
            motive for motive in overlap_motives_for_column if motive.word_level
        ]
        not_word_level_motives_for_column = [
            motive for motive in overlap_motives_for_column if not motive.word_level
        ]

        # Find biggest overlap among the non-word_level ones
        if not_word_level_motives_for_column:
            max_overlap_not_word_level_for_column = max(
                not_word_level_motives_for_column, key=lambda m: m.overlap
            )
            max_overlap_not_word_level_for_column_overlap = (
                max_overlap_not_word_level_for_column.overlap
            )
        else:
            max_overlap_not_word_level_for_column = []
            max_overlap_not_word_level_for_column_overlap = (
                0  # Will never be used, left for linter
            )

        # Now find biggest overlap among the word_level ones
        if word_level_motives_for_column:
            max_overlap_word_level_for_column = max(
                word_level_motives_for_column, key=lambda m: m.overlap
            )
            max_overlap_word_level_for_column_overlap = (
                max_overlap_word_level_for_column.overlap
            )
            if not_word_level_motives_for_column:
                # If there is already an OverlapMotive on same column with equal or greater overlap but not word_level, discard it
                if (
                    max_overlap_word_level_for_column_overlap
                    <= max_overlap_not_word_level_for_column_overlap
                ):
                    max_overlap_word_level_for_column = []
        else:
            max_overlap_word_level_for_column = []

        if max_overlap_not_word_level_for_column:
            max_overlap_not_word_level_for_column = [
                max_overlap_not_word_level_for_column
            ]
        if max_overlap_word_level_for_column:
            max_overlap_word_level_for_column = [max_overlap_word_level_for_column]
        final_motives += (
            max_overlap_word_level_for_column + max_overlap_not_word_level_for_column
        )

    # Remove duplicates
    final_motives_no_duplicates = []
    for motive in final_motives:
        if motive not in final_motives_no_duplicates:
            final_motives_no_duplicates.append(motive)
    return final_motives_no_duplicates
