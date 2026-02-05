import pandas as pd
import pytest

import ms_blocking.ms_blocking as msb
from ms_blocking.datasets import get_users
from ms_blocking.ms_blocking import AndNode


@pytest.fixture
def attribute_city_links():
    return {
        frozenset({1, 4}),
        frozenset({8, 11}),
        frozenset({2, 5}),
        frozenset({10, 13}),
        frozenset({3, 8}),
        frozenset({3, 11}),
    }


@pytest.fixture
def overlap_websites_links():
    return {
        frozenset({1, 4}),
        frozenset({6, 10}),
        frozenset({10, 13}),
        frozenset({6, 13}),
        frozenset({1, 6}),
        frozenset({1, 10}),
    }


@pytest.fixture
def overlap_2_websites_links():
    return {frozenset({6, 10})}


@pytest.fixture
def overlap_websites_merge_blocks():
    return [0, 0, 1, 1, 1, 2, 2, 2]


@pytest.fixture
def attribute_city_age_links():
    return {frozenset({1, 4}), frozenset({2, 5}), frozenset({8, 11})}


@pytest.fixture
def union_attribute_city_overlap_websites_links():
    return {
        frozenset({1, 4}),
        frozenset({8, 11}),
        frozenset({6, 10}),
        frozenset({2, 5}),
        frozenset({10, 13}),
        frozenset({3, 8}),
        frozenset({3, 11}),
        frozenset({6, 13}),
        frozenset({1, 6}),
        frozenset({3, 11}),
        frozenset({1, 10}),
    }


@pytest.fixture
def intersection_attribute_city_overlap_websites_links():
    return {frozenset({1, 4}), frozenset({10, 13})}


@pytest.fixture
def attribute_city_sort_false_blocks():
    return [0, 1, 2, 0, 1, 2, 3, 2, 3]


@pytest.fixture
def attribute_city_keep_ungrouped_rows_false():
    return [0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7, 8]


@pytest.fixture
def attribute_city_motives_true_block():
    return {
        frozenset({3, 8}): [msb.EquivalenceMotive("City")],
        frozenset({1, 4}): [msb.EquivalenceMotive("City")],
        frozenset({8, 11}): [msb.EquivalenceMotive("City")],
        frozenset({3, 11}): [msb.EquivalenceMotive("City")],
        frozenset({2, 5}): [msb.EquivalenceMotive("City")],
        frozenset({10, 13}): [msb.EquivalenceMotive("City")],
    }


@pytest.fixture
def attribute_city_motives_true_add():
    return [
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
        ["Same 'City'"],
    ]


@pytest.fixture
def attribute_city_show_as_pairs_true_id():
    return {(1, 4), (2, 5), (3, 11), (8, 3), (8, 11), (10, 13)}


@pytest.fixture
def attribute_city_show_as_pairs_true_columns():
    return ["id_l", "Name_l", "id_r", "Name_r", "_block"]


@pytest.fixture
def city_age_name_websites_pipelining_id():
    return [1, 4, 2, 5, 8, 11]


@pytest.fixture
def city_age_websites_pipelining_motives():
    return [
        {"Same 'City'", "Same 'Age'", ">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {"Same 'City'", "Same 'Age'", ">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {"Same 'City'", "Same 'Age'"},
        {"Same 'City'", "Same 'Age'"},
        {">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
        {">=1 overlap in 'websites'"},
    ]


@pytest.fixture
def city_age_websites_pipelining_scores():
    return [3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]


@pytest.fixture
def city_age_websites_pipelining_scores_not_show_as_pairs():
    return [3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]


@pytest.fixture
def city_age_not_different():
    return {frozenset({1, 4}), frozenset({8, 11}), frozenset({2, 5})}


@pytest.fixture
def name_and_city_age_not_different():
    return {frozenset({8, 11}), frozenset({2, 5})}


def test_simple_attribute_equivalence_blocking(attribute_city_links):
    """Test that AB blocking does work as intended"""
    expected = attribute_city_links
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    actual = city_blocker.block(get_users())
    assert actual == expected, (
        "Blocking on City should return {{1, 4}, {8, 11}, {2, 5}, {10, 13}, {3, 8}, {3, 11}}"
    )


def test_simple_overlap_blocking(overlap_websites_links):
    """Test that OB blocking does work as intended"""
    expected = overlap_websites_links
    websites_blocker = msb.OverlapBlocker(["websites"])
    actual = websites_blocker.block(get_users())
    assert actual == expected, (
        "Blocking on websites with overlap 1 should return {{1, 4}, {6, 10}, {10, 13}, {6, 13}, {1, 6}, {1, 10}}"
    )


def test_different_overlap(overlap_2_websites_links):
    """Test that changing overlap does work as intended"""
    expected = overlap_2_websites_links
    websites_blocker_2 = msb.OverlapBlocker(["websites"], 2)
    actual = websites_blocker_2.block(get_users())
    assert actual == expected, (
        "Blocking on websites with overlap 2 should return {{6,10}}"
    )


def test_merge_blocks(overlap_websites_merge_blocks):
    """Test that merge_blocks=False does work as intended"""
    expected = overlap_websites_merge_blocks
    websites_blocker = msb.OverlapBlocker(["websites"])
    links = websites_blocker.block(get_users())
    actual = msb.add_blocks_to_dataset(get_users(), links, merge_blocks=False)[
        "_block"
    ].to_list()
    assert actual == expected, (
        "Blocking on websites should return [0, 0, 0, 1, 1, 2, 2, 2]"
    )


def test_double_attribute_equivalence_blocking(attribute_city_age_links):
    """Test that AB blocking does work as intended with 2 columns as argument"""
    expected = attribute_city_age_links
    city_and_age_blocker = msb.AttributeEquivalenceBlocker(["City", "Age"])
    actual = city_and_age_blocker.block(get_users())
    assert actual == expected, (
        "Blocking on City and Age should return {{1, 4}, {2, 5}, {8, 11}}"
    )


def test_normalize_strings_false():
    """Test that normalize_strings=False does work as intended"""
    expected = set()
    normalized_name_blocker = msb.AttributeEquivalenceBlocker(
        ["Name"], normalize_strings=False
    )
    actual = normalized_name_blocker.block(get_users())
    assert actual == expected, (
        "Blocking on Name with normalize_strings=False should return an empty set"
    )


def test_union(union_attribute_city_overlap_websites_links):
    """Test that union does work as intended"""
    expected = union_attribute_city_overlap_websites_links
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    actual = (city_blocker | websites_blocker).block(get_users())
    assert actual == expected, (
        "Blocking on City OR websites should return {{1, 4}, {8, 11}, {6, 10}, {2, 5}, {10, 13}, {3, 8}, {3, 11}, {6, 13}, {1, 6}, {3, 11}, {1, 10}}"
    )


def test_intersection(intersection_attribute_city_overlap_websites_links):
    """Test that intersection does work as intended"""
    expected = intersection_attribute_city_overlap_websites_links
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    actual = (city_blocker & websites_blocker).block(get_users())
    assert actual == expected, (
        "Blocking on City AND websites should return {{1, 4}, {10, 13}}"
    )


def test_sort_false(attribute_city_sort_false_blocks):
    """Test that sort=False does work as intended"""
    expected = attribute_city_sort_false_blocks
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users())
    actual = msb.add_blocks_to_dataset(get_users(), links, sort=False)[
        "_block"
    ].to_list()
    assert actual == expected, (
        "Blocking on websites and adding blocks with sort=False should return [0, 1, 2, 0, 1, 2, 3, 2, 3]"
    )


def test_keep_ungrouped_rows_false(attribute_city_keep_ungrouped_rows_false):
    """Test that keep_ungrouped_rows=False does work as intended"""
    expected = attribute_city_keep_ungrouped_rows_false
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users())
    actual = msb.add_blocks_to_dataset(get_users(), links, keep_ungrouped_rows=True)[
        "_block"
    ].to_list()
    assert actual == expected, (
        "Blocking on Name with normalize_strings=False should return [0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7, 8]"
    )


def test_motives_when_blocking(attribute_city_motives_true_block):
    """Test that keep_ungrouped_rows=False does work as intended"""
    expected = attribute_city_motives_true_block
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    actual = city_blocker.block(get_users(), motives=True)
    assert actual == expected


def test_motives_when_adding_to_dataframe(attribute_city_motives_true_add):
    """Test that motives=True does work as intended"""
    expected = attribute_city_motives_true_add
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users(), motives=True)
    actual = msb.add_blocks_to_dataset(get_users(), links, motives=True)[
        "_motive"
    ].to_list()
    assert actual == expected


def test_show_as_pairs(attribute_city_show_as_pairs_true_id):
    """Test that test_show_as_pairs=True does work as intended"""
    expected = attribute_city_show_as_pairs_true_id
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users(), motives=True)
    blocked_df = msb.add_blocks_to_dataset(get_users(), links, show_as_pairs=True)
    id_ls = blocked_df["id_l"].to_list()
    id_rs = blocked_df["id_r"].to_list()
    actual = set(zip(id_ls, id_rs))
    assert actual == expected


def test_output_columns(attribute_city_show_as_pairs_true_columns):
    """Test that test_show_as_pairs=True does work as intended"""
    expected = attribute_city_show_as_pairs_true_columns
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users(), motives=True)
    actual = msb.add_blocks_to_dataset(
        get_users(), links, show_as_pairs=True, output_columns=["id", "Name"]
    ).columns.to_list()
    assert actual == expected


def test_pipelining(city_age_name_websites_pipelining_id):
    """Test that pipelining does work as intended"""
    expected = city_age_name_websites_pipelining_id
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    name_blocker = msb.AttributeEquivalenceBlocker(["Name"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    final_blocker = (city_blocker & age_blocker) | (name_blocker & websites_blocker)
    links = final_blocker.block(get_users(), motives=True)
    actual = msb.add_blocks_to_dataset(get_users(), links)["id"].to_list()
    assert actual == expected


def test_pipelining_motives(city_age_websites_pipelining_motives):
    """Test that pipelining does work as intended regarding motives"""
    expected = city_age_websites_pipelining_motives
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    final_blocker = (city_blocker & age_blocker) | websites_blocker
    links = final_blocker.block(get_users(), motives=True)
    motives = msb.add_blocks_to_dataset(  # Use set to ignore ordering
        get_users(), links, show_as_pairs=True, motives=True, merge_blocks=False
    )["_motive"].to_list()
    actual = [set(motive) for motive in motives]
    assert actual == expected


def test_pipelining_scores(city_age_websites_pipelining_scores):
    """Test that scoring does work as intended"""
    expected = city_age_websites_pipelining_scores
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    final_blocker = (city_blocker & age_blocker) | websites_blocker
    links = final_blocker.block(get_users(), motives=True)
    report = msb.add_blocks_to_dataset(
        get_users(),
        links,
        show_as_pairs=True,
        motives=True,
        merge_blocks=False,
        score=True,
    )
    actual = sorted(report["_score"], reverse=True)
    assert actual == expected


def test_pipelining_scores_without_show_as_pairs(
    city_age_websites_pipelining_scores_not_show_as_pairs,
):
    """Test that scoring does work as intended"""
    expected = city_age_websites_pipelining_scores_not_show_as_pairs
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    final_blocker = (city_blocker & age_blocker) | websites_blocker
    links = final_blocker.block(get_users(), motives=True)
    report = msb.add_blocks_to_dataset(
        get_users(),
        links,
        show_as_pairs=False,
        motives=True,
        merge_blocks=False,
        score=True,
    )
    actual = sorted(report["_score"], reverse=True)
    assert actual == expected


def test_merge_blockers_aa():
    """Test that merging blockers does work as intended"""
    expected = msb.AttributeEquivalenceBlocker(["City", "Age"])
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    actual = city_blocker & age_blocker
    assert actual == expected


def test_merge_blockers_oo():
    """Test that merging blockers does work as intended"""
    expected = msb.OverlapBlocker(["websites"])
    websites_blocker_1 = msb.OverlapBlocker(["websites"])
    websites_blocker_2 = msb.OverlapBlocker(["websites"])
    actual = websites_blocker_1 & websites_blocker_2
    assert actual == expected
    expected = msb.OverlapBlocker(["Name", "websites"])
    name_overlap_blocker = msb.OverlapBlocker(["Name"])
    actual = name_overlap_blocker & websites_blocker_1
    assert actual == expected


def test_merge_blockers_oa():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["City"], ["websites"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    actual = websites_blocker & city_blocker
    assert actual == expected


def test_merge_blockers_ao():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["City"], ["websites"])
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    actual = city_blocker & websites_blocker
    assert actual == expected


def test_merge_blockers_mm():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["City", "Age"], ["websites"])
    mixed_blocker_1 = msb.MixedBlocker(["City"], ["websites"])
    mixed_blocker_2 = msb.MixedBlocker(["Age"], ["websites"])
    actual = mixed_blocker_1 & mixed_blocker_2
    assert actual == expected


def test_merge_blockers_ma():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["City", "Age"], ["websites"])
    mixed_blocker = msb.MixedBlocker(["City"], ["websites"])
    city_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    actual = mixed_blocker & city_blocker
    assert actual == expected


def test_merge_blockers_mo():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["City"], ["websites"])
    mixed_blocker = msb.MixedBlocker(["City"], ["websites"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    actual = mixed_blocker & websites_blocker
    assert actual == expected


def test_merge_blockers_am():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["City", "Age"], ["websites"])
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    mixed_blocker = msb.MixedBlocker(["Age"], ["websites"])
    actual = city_blocker & mixed_blocker
    assert actual == expected


def test_merge_blockers_om():
    """Test that merging blockers does work as intended"""
    expected = msb.MixedBlocker(["Age"], ["websites"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    mixed_blocker = msb.MixedBlocker(["Age"], ["websites"])
    actual = websites_blocker & mixed_blocker
    assert actual == expected


def test_merge_blockers_and():
    """Test that merging blockers does work as intended"""
    expected = msb.AndNode(
        msb.AttributeEquivalenceBlocker(["City"]),
        msb.AttributeEquivalenceBlocker(["Age"], must_not_be_different=["Name"]),
    )
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker_name_not_different = msb.AttributeEquivalenceBlocker(
        ["Age"], must_not_be_different=["Name"]
    )
    actual = city_blocker & age_blocker_name_not_different
    assert actual == expected


def test_block_andnode(name_and_city_age_not_different):
    """Test that the optimizations of AndNode do work as intended"""
    expected = name_and_city_age_not_different
    name_blocker = msb.AttributeEquivalenceBlocker(["Name"])
    city_blocker_age_not_different = msb.AttributeEquivalenceBlocker(
        ["City"], must_not_be_different=["Age"]
    )
    actual = (name_blocker & city_blocker_age_not_different).block(get_users())
    assert actual == expected


def test_must_not_be_different(city_age_not_different):
    """Test that must_not_be_different does work as intended"""
    expected = city_age_not_different
    city_blocker_not_different_age = msb.AttributeEquivalenceBlocker(
        ["City"], must_not_be_different=["Age"]
    )
    actual = city_blocker_not_different_age.block(get_users())
    assert actual == expected
    mixed_age_blocker_not_different_name = msb.MixedBlocker(
        ["Age"], ["websites"], must_not_be_different=["Name"], normalize_strings=False
    )
    actual = mixed_age_blocker_not_different_name.block(get_users())
    assert actual == set()
    mixed_age_blocker_not_different_name_normalized = msb.MixedBlocker(
        ["Age"], ["websites"], must_not_be_different=["Name"]
    )
    links = mixed_age_blocker_not_different_name_normalized.block(get_users())
    actual = msb.add_blocks_to_dataset(get_users(), links)["Name"].to_list()
    assert actual == []


def test_no_links_a():
    """Test that AttributeEquivalenceBLocker gracefully outputs an empty set when no pairs are found"""
    expected = set()
    expected_motives = dict()
    id_blocker = msb.AttributeEquivalenceBlocker(["id"])
    actual = id_blocker.block(get_users())
    assert actual == expected
    actual_motives = id_blocker.block(get_users(), motives=True)
    assert actual_motives == expected_motives


def test_no_links_o():
    """Test that OverlapBlocker gracefully outputs an empty set when no pairs are found"""
    expected = set()
    expected_motives = dict()
    websites_blocker_3 = msb.OverlapBlocker(["websites"], 3)
    actual = websites_blocker_3.block(get_users())
    assert actual == expected
    actual_motives = websites_blocker_3.block(get_users(), motives=True)
    assert actual_motives == expected_motives


def test_no_links_m():
    """Test that MixedBlocker gracefully outputs an empty set when no pairs are found"""
    expected = set()
    expected_motives = dict()
    id_blocker = msb.AttributeEquivalenceBlocker(["id"])
    websites_blocker_3 = msb.OverlapBlocker(["websites"], 3)
    actual = (id_blocker & websites_blocker_3).block(get_users())
    assert actual == expected
    actual_motives = id_blocker.block(get_users(), motives=True)
    assert actual_motives == expected_motives


def test_no_links_add_blocks_to_dataframe():
    """Test that add_blocks_to_dataframe gracefully outputs an empty DataFrame when no pairs were found"""
    expected = pd.DataFrame(columns=["id", "Name", "City", "Age", "websites", "_block"])
    expected_show_as_pairs = pd.DataFrame(
        columns=[
            "id_l",
            "Name_l",
            "City_l",
            "Age_l",
            "websites_l",
            "id_r",
            "Name_r",
            "City_r",
            "Age_r",
            "websites_r",
            "_block",
        ]
    )
    expected_motives = pd.DataFrame(
        columns=["id", "Name", "City", "Age", "websites", "_motive", "_block"]
    )
    id_blocker = msb.AttributeEquivalenceBlocker(["id"])
    links = id_blocker.block(get_users())
    actual = msb.add_blocks_to_dataset(get_users(), links)
    assert (actual == expected).all().all()
    actual_show_as_pairs = msb.add_blocks_to_dataset(
        get_users(), links, show_as_pairs=True
    )
    assert (actual_show_as_pairs == expected_show_as_pairs).all().all()
    links_motives = id_blocker.block(get_users(), motives=True)
    actual_motives = msb.add_blocks_to_dataset(get_users(), links_motives, motives=True)
    assert (actual_motives == expected_motives).all().all()


def test_generic_blockernode_methods():
    expected = msb.AttributeEquivalenceBlocker(["City"])
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    actual = city_blocker | city_blocker
    assert actual == expected
    name_blocker = msb.AttributeEquivalenceBlocker(["Name"])
    actual = AndNode(city_blocker, name_blocker)
    assert actual != 42
    name_blocker = msb.AttributeEquivalenceBlocker(["Name"])
    expected = f"OrNode{{{msb.AttributeEquivalenceBlocker(['City'])}, {msb.AttributeEquivalenceBlocker(['Name'])}}}"
    actual = str(city_blocker | name_blocker)
    assert actual == expected
    mixed_city_websites_blocker = msb.MixedBlocker(["City"], ["websites"])
    assert mixed_city_websites_blocker != 42


def test_errors_with_constructors():
    with pytest.raises(ValueError):
        msb.AttributeEquivalenceBlocker(["City"], ["City"])
    with pytest.raises(ValueError):
        msb.AttributeEquivalenceBlocker(["City"], ["Age", "Name"])
    with pytest.raises(ValueError):
        msb.OverlapBlocker(["websites"], -1)
    with pytest.raises(ValueError):
        msb.MixedBlocker(["City"], [], ["City"], overlap=-1)
    with pytest.raises(ValueError):
        msb.MixedBlocker(["websites"], [], ["Age", "Name"], overlap=-1)
    with pytest.raises(ValueError):
        msb.MixedBlocker([], ["websites"], overlap=-1)
    with pytest.raises(ValueError):
        msb.MixedBlocker(["City"], [], ["Age", "Name"])
    with pytest.raises(ValueError):
        msb.MixedBlocker(["City"], [], ["City"])
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        msb.EquivalenceMotive(42)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        msb.OverlapMotive(42)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        msb.OverlapMotive("City", "42")
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        msb.OverlapMotive("City", 1, 42)


def test_comparing_motives():
    city_motive = msb.EquivalenceMotive("City")
    websites_motive = msb.OverlapMotive("websites")
    with pytest.raises(TypeError):
        assert city_motive == 42
    with pytest.raises(TypeError):
        assert websites_motive == 42
    assert city_motive != websites_motive
    assert websites_motive != city_motive


def test_repr_motives():
    city_motive = msb.EquivalenceMotive("City")
    websites_motive = msb.OverlapMotive("websites")
    assert city_motive.__repr__() == "EquivalenceMotive('City')"
    assert websites_motive.__repr__() == "OverlapMotive('websites', 1)"


def test_edge_normalize():
    expected = ["abricot", "banane", "couscous", 42]
    actual = msb.normalize(["__Abricot", "Banane", "(Couscous)", 42])
    assert actual == expected
    assert msb.normalize([None, None]) == []
    assert msb.normalize(["__Abricot", None]) == ["abricot"]


def test_edge_parse_list():
    assert msb.parse_list([["a", "b", "c d e"]]) == ["a", "b", "c d e"]
    assert msb.parse_list(['["a", "b", "c"]']) == ["a", "b", "c"]
    assert msb.parse_list(["a", "b", "c d e"]) == ["a", "b", "c d e"]
    assert msb.parse_list("[-42-]") == ["[-42-]"]
    assert msb.parse_list('["a", "b", "c d e"]', word_level=True) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]
    assert msb.parse_list(["a", "b", None]) == ["a", "b"]
    assert msb.parse_list([None, None, None]) == []
    assert msb.parse_list(None) == []
    assert msb.parse_list([]) == []
    assert msb.parse_list("") == []


def test_errors_with_functions():
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users())
    links_motives = city_blocker.block(get_users(), motives=True)
    data_broken_index = get_users()
    data_broken_index.index = [1] * len(data_broken_index)
    data__motive = get_users().rename({"Age": "_motive"}, axis=1)
    data__score = get_users().rename({"Age": "_score"}, axis=1)
    data__block = get_users().rename({"Age": "_block"}, axis=1)

    with pytest.raises(TypeError):
        msb.add_blocks_to_dataset(get_users(), links, motives=True)
    with pytest.raises(ValueError):
        msb.add_blocks_to_dataset(
            get_users(), links, show_as_pairs=True, keep_ungrouped_rows=True
        )
    with pytest.raises(ValueError):
        msb.add_blocks_to_dataset(data_broken_index, links)
    with pytest.raises(TypeError):
        msb.add_blocks_to_dataset(get_users(), links, score=True)
    with pytest.raises(ValueError):
        msb.add_blocks_to_dataset(data__motive, links_motives, motives=True)
    with pytest.raises(ValueError):
        msb.add_blocks_to_dataset(data__score, links_motives, motives=True, score=True)
    with pytest.raises(ValueError):
        msb.add_blocks_to_dataset(data__block, links)
    with pytest.raises(ValueError):
        msb.add_blocks_to_dataset(data__block, links)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        msb.flatten(42)
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        msb.solve_motives([])


def test_passing_strings_to_constructors(city_age_name_websites_pipelining_id):
    """Test that passing strings to constructors instead of lists does work as intended"""
    expected = city_age_name_websites_pipelining_id
    city_blocker = msb.AttributeEquivalenceBlocker("City", "Age")
    age_blocker = msb.AttributeEquivalenceBlocker("Age")
    name_blocker = msb.AttributeEquivalenceBlocker("Name")
    websites_blocker = msb.OverlapBlocker("websites")
    final_blocker = (city_blocker & age_blocker) | (name_blocker & websites_blocker)
    links = final_blocker.block(get_users(), motives=True)
    actual = msb.add_blocks_to_dataset(get_users(), links)["id"].to_list()
    assert actual == expected
    some_mixed_blocker = msb.MixedBlocker("City", "websites", "id")
    actual = some_mixed_blocker.block(get_users(), motives=True)
    assert actual == dict()


def test_return_empty():
    nope_blocker_ae = msb.AttributeEquivalenceBlocker(
        "Name", "Age", normalize_strings=False
    )
    actual = nope_blocker_ae.block(get_users())
    assert actual == set()
    actual = nope_blocker_ae.block(get_users(), motives=True)
    assert actual == dict()

    nope_blocker_overlap = msb.OverlapBlocker("id")
    actual = nope_blocker_overlap.block(get_users())
    assert actual == set()
    actual = nope_blocker_overlap.block(get_users(), motives=True)
    assert actual == dict()

    actual = (nope_blocker_ae & nope_blocker_overlap).block(get_users(), motives=True)
    assert actual == dict()

    assert list(msb.solve_connected_components_from_coords({})) == []


def test_add_empty_coords_to_dataframe():
    expected = pd.DataFrame(
        columns=["id", "Name", "City", "Age", "websites", "_motive", "_score", "_block"]
    )
    actual = msb.add_blocks_to_dataset(get_users(), dict(), motives=True, score=True)
    assert (actual == expected).all().all()


def test_generate_blocking_report(city_age_websites_pipelining_motives):
    expected_block = [0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4]
    expected_motives = city_age_websites_pipelining_motives
    expected_score = [3, 1, 1, 3, 1, 1, 1, 1, 2, 2, 1, 1, 1]
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    final_blocker = (city_blocker & age_blocker) | websites_blocker
    links = final_blocker.block(get_users(), motives=True)
    actual = msb.generate_blocking_report(get_users(), links)
    actual_block = actual["_block"].to_list()
    actual_motive = [set(motive) for motive in actual["_motive"].to_list()]
    actual_score = actual["_score"].to_list()
    assert expected_block == actual_block
    assert expected_motives == actual_motive
    assert expected_score == actual_score


def test_empty_after_must_not_be_different():
    mixed_age_blocker_not_different_age = msb.AttributeEquivalenceBlocker(
        ["Age"], ["Name"]
    )
    actual = mixed_age_blocker_not_different_age.block(get_users())
    actual_motives = mixed_age_blocker_not_different_age.block(
        get_users(), motives=True
    )
    assert actual == {frozenset({8, 11}), frozenset({2, 5})}
    assert actual_motives == {
        frozenset({8, 11}): [msb.EquivalenceMotive("Age")],
        frozenset({2, 5}): [msb.EquivalenceMotive("Age")],
    }


def test_solve_motives():
    motive_ae_city = msb.EquivalenceMotive("City")
    assert all([x in [motive_ae_city] for x in msb.solve_motives([motive_ae_city, motive_ae_city])]) and len([motive_ae_city]) == len(msb.solve_motives([motive_ae_city, motive_ae_city]))
    motive_overlap_websites_1 = msb.OverlapMotive("websites", 1)
    motive_overlap_websites_2 = msb.OverlapMotive("websites", 2)
    motive_overlap_websites_1_word_level = msb.OverlapMotive("websites", 1, word_level=True)
    motive_overlap_websites_2_word_level = msb.OverlapMotive("websites", 2, word_level=True)
    assert all([x in [motive_overlap_websites_2] for x in msb.solve_motives([motive_overlap_websites_1, motive_overlap_websites_2])]) and len([motive_overlap_websites_2]) == len(msb.solve_motives([motive_overlap_websites_1, motive_overlap_websites_2]))
    assert all([x in [motive_overlap_websites_1, motive_overlap_websites_2_word_level] for x in msb.solve_motives([motive_overlap_websites_1, motive_overlap_websites_2_word_level])]) and len([motive_overlap_websites_1, motive_overlap_websites_2_word_level]) == len(msb.solve_motives([motive_overlap_websites_1, motive_overlap_websites_2_word_level]))
    assert all([x in [motive_overlap_websites_2] for x in msb.solve_motives([motive_overlap_websites_2, motive_overlap_websites_2_word_level])]) and len([motive_overlap_websites_2]) == len(msb.solve_motives([motive_overlap_websites_2, motive_overlap_websites_2_word_level]))
    assert all([x in [motive_overlap_websites_2_word_level] for x in msb.solve_motives([motive_overlap_websites_1_word_level, motive_overlap_websites_2_word_level])]) and len([motive_overlap_websites_2_word_level]) == len(msb.solve_motives([motive_overlap_websites_1_word_level, motive_overlap_websites_2_word_level]))