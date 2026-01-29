import pandas as pd
import pytest

import ms_blocking.ms_blocking as msb
from ms_blocking.datasets import get_users


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
def attribute_name_normalize_strings_links():
    return set()


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
        frozenset({3, 8}): {"Same 'City'"},
        frozenset({1, 4}): {"Same 'City'"},
        frozenset({8, 11}): {"Same 'City'"},
        frozenset({3, 11}): {"Same 'City'"},
        frozenset({2, 5}): {"Same 'City'"},
        frozenset({10, 13}): {"Same 'City'"},
    }


@pytest.fixture
def attribute_city_motives_true_add():
    return [{"Same 'City'"}] * 9


@pytest.fixture
def attribute_city_show_as_pairs_true_id():
    return {(1, 4), (2, 5), (3, 11), (8, 3), (8, 11), (10, 13)}


@pytest.fixture
def attribute_city_show_as_pairs_true_columns():
    return ["id_l", "Name_l", "id_r", "Name_r", "block"]


@pytest.fixture
def city_age_name_websites_pipelining_id():
    return [1, 4, 2, 5, 8, 11]


@pytest.fixture
def city_age_websites_pipelining_motives():
    return [
        frozenset({"Same 'Age'", "Same 'City'", ">=1 overlap in 'websites'"}),
        frozenset({"Same 'Age'", "Same 'City'", ">=1 overlap in 'websites'"}),
        frozenset({"Same 'Age'", "Same 'City'", ">=1 overlap in 'websites'"}),
        frozenset({"Same 'Age'", "Same 'City'", ">=1 overlap in 'websites'"}),
        frozenset({"Same 'Age'", "Same 'City'", ">=1 overlap in 'websites'"}),
        frozenset({"Same 'Age'", "Same 'City'", ">=1 overlap in 'websites'"}),
        frozenset({">=1 overlap in 'websites'"}),
        frozenset({">=1 overlap in 'websites'"}),
        frozenset({"Same 'Age'", "Same 'City'"}),
        frozenset({"Same 'Age'", "Same 'City'"}),
        frozenset({">=1 overlap in 'websites'"}),
        frozenset({">=1 overlap in 'websites'"}),
        frozenset({">=1 overlap in 'websites'"}),
    ]


@pytest.fixture
def city_age_websites_pipelining_scores():
    return [3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1]


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
        "block"
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


def test_normalize_strings_false(attribute_name_normalize_strings_links):
    """Test that normalize_strings=False does work as intended"""
    expected = attribute_name_normalize_strings_links
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
        "block"
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
        "block"
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
        "motive"
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


def test_generate_blocking_report(attribute_city_show_as_pairs_true_id):
    """Test that generate_blocking_report does work as intended"""
    expected = attribute_city_show_as_pairs_true_id
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    links = city_blocker.block(get_users(), motives=True)
    blocked_df = msb.add_blocks_to_dataset(get_users(), links, show_as_pairs=True)
    id_ls = blocked_df["id_l"].to_list()
    id_rs = blocked_df["id_r"].to_list()
    actual = set(zip(id_ls, id_rs))
    assert actual == expected


def test_pipelining_motives(city_age_websites_pipelining_motives):
    """Test that pipelining does work as intended regarding motives"""
    expected = city_age_websites_pipelining_motives
    city_blocker = msb.AttributeEquivalenceBlocker(["City"])
    age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
    websites_blocker = msb.OverlapBlocker(["websites"])
    final_blocker = (city_blocker & age_blocker) | websites_blocker
    links = final_blocker.block(get_users(), motives=True)
    actual = msb.add_blocks_to_dataset(
        get_users(), links, show_as_pairs=True, motives=True, merge_blocks=False
    )["motive"].to_list()
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
        get_users(), links, show_as_pairs=True, motives=True, merge_blocks=False
    )
    actual = sorted(msb.scoring(report), reverse=True)
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
    expected = pd.DataFrame(columns=["id", "Name", "City", "Age", "websites", "block"])
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
            "block",
        ]
    )
    expected_motives = pd.DataFrame(
        columns=["id", "Name", "City", "Age", "websites", "motive", "block"]
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
