import time

import pytest

import fallback
import index as ix
import parse
from conftest import SAMPLE_CORPUS


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fb_index")
    ix.save(d, ix.build(parse.walk(SAMPLE_CORPUS)))
    return d


def test_single_rare_id_fixed_in(index_dir):
    assert fallback.lookup(index_dir, "Which firmware version fixed ticket ORR-1847?") == \
        ("4.3.2", ["support/bug_database.csv"])


def test_other_attribute_of_same_record(index_dir):
    assert fallback.lookup(index_dir, "What is the severity of ticket ORR-1847?") == \
        ("S2", ["support/bug_database.csv"])


def test_no_rare_id_refuses(index_dir):
    assert fallback.lookup(index_dir, "What is the maximum junction temperature of the TQ-40?") == ("", [])


def test_rare_id_without_attribute_word_refuses(index_dir):
    assert fallback.lookup(index_dir, "Tell me about ticket ORR-1847") == ("", [])


def test_unknown_id_refuses(index_dir):
    assert fallback.lookup(index_dir, "Which firmware version fixed ticket ORR-9999?") == ("", [])


def test_two_rare_ids_refuse(index_dir):
    assert fallback.lookup(index_dir, "Which version fixed ORR-1847 and ORR-1802?") == ("", [])


def test_missing_index_dir(tmp_path):
    assert fallback.lookup(tmp_path / "nope", "Which firmware version fixed ticket ORR-1847?") == ("", [])


def test_corrupt_index_never_raises(tmp_path):
    (tmp_path / "files.json").write_text("{not json")
    (tmp_path / "records.jsonl").write_text("garbage\n")
    assert fallback.lookup(tmp_path, "Which firmware version fixed ticket ORR-1847?") == ("", [])


def test_deadline_exceeded(index_dir):
    assert fallback.lookup(index_dir, "Which firmware version fixed ticket ORR-1847?",
                           deadline_ts=time.time() - 1) == ("", [])


def test_future_deadline_ok(index_dir):
    assert fallback.lookup(index_dir, "Which firmware version fixed ticket ORR-1847?",
                           deadline_ts=time.time() + 5)[0] == "4.3.2"
