import json
import os
import pytest
from logic.src.utils.target.matcher import (
    PolicyFilter,
    _parse_slug,
    slug_matches_filter,
    display_name_matches_filter,
)
from logic.src.utils.target.remover import (
    remove_from_json_file,
    remove_from_jsonl_file,
    remove_checkpoint_files,
    remove_fill_history_files,
    remove_targeted_runs,
)

def test_parse_slug():
    # Test lookahead_aco_hh_custom_ftsp_emp
    p = _parse_slug("lookahead_aco_hh_custom_ftsp_emp")
    assert p["distribution"] == "emp"
    assert p["improver"] == "ftsp"
    assert p["ms_strategy"] == "lookahead"
    assert p["constructor"] == "aco_hh_custom"

    # Test another combination
    p = _parse_slug("last_minute_cf70_sans_new_ftsp_gamma1")
    assert p["distribution"] == "gamma1"
    assert p["improver"] == "ftsp"
    assert p["ms_strategy"] == "last_minute_cf70"
    assert p["constructor"] == "sans_new"

    # Edge cases - no distribution or improver
    p = _parse_slug("regular_gurobi_none_none")
    assert p["distribution"] == ""
    assert p["improver"] == "none"
    assert p["ms_strategy"] == "regular"
    assert p["constructor"] == "gurobi_none"

    # Empty/unexpected input
    p = _parse_slug("invalid_slug")
    assert p["constructor"] == "invalid_slug"

def test_slug_matches_filter():
    filt = PolicyFilter(distributions=["emp"], constructors=["aco_hh"], ms_strategies=["lookahead"])
    assert slug_matches_filter("lookahead_aco_hh_custom_ftsp_emp", filt) is True
    assert slug_matches_filter("lookahead_aco_hh_custom_ftsp_gamma1", filt) is False
    assert slug_matches_filter("lookahead_gurobi_ftsp_emp", filt) is False

    # exact match test
    filt_exact = PolicyFilter(constructors=["aco_hh_custom"], exact_match=True)
    assert slug_matches_filter("lookahead_aco_hh_custom_ftsp_emp", filt_exact) is True
    filt_exact_fail = PolicyFilter(constructors=["aco_hh"], exact_match=True)
    assert slug_matches_filter("lookahead_aco_hh_custom_ftsp_emp", filt_exact_fail) is False

def test_display_name_matches_filter():
    filt = PolicyFilter(distributions=["emp"], constructors=["aco_hh"], ms_strategies=["lookahead"], improvers=["ftsp"])
    assert display_name_matches_filter("Lookahead + ACO_HH_CUSTOM + Ftsp (Emp)", filt) is True
    assert display_name_matches_filter("Lookahead + Gurobi + Ftsp (Emp)", filt) is False

    # Test with empty fields
    filt_empty = PolicyFilter()
    assert display_name_matches_filter("Lookahead + ACO_HH_CUSTOM + Ftsp (Emp)", filt_empty) is True

def test_remove_from_json_file(tmp_path):
    path = tmp_path / "test.json"
    data = {
        "lookahead_aco_hh_custom_ftsp_emp": {"metric": 1.0},
        "regular_gurobi_none_gamma1": {"metric": 2.0},
    }
    with open(path, "w") as f:
        json.dump(data, f)

    filt = PolicyFilter(distributions=["emp"])

    # Dry run
    removed = remove_from_json_file(str(path), filt, dry_run=True)
    assert removed == ["lookahead_aco_hh_custom_ftsp_emp"]
    with open(path, "r") as f:
        loaded = json.load(f)
    assert "lookahead_aco_hh_custom_ftsp_emp" in loaded

    # Live run
    removed = remove_from_json_file(str(path), filt, dry_run=False)
    assert removed == ["lookahead_aco_hh_custom_ftsp_emp"]
    with open(path, "r") as f:
        loaded = json.load(f)
    assert "lookahead_aco_hh_custom_ftsp_emp" not in loaded
    assert "regular_gurobi_none_gamma1" in loaded

    # Non-existent file
    assert remove_from_json_file("nonexistent.json", filt) == []

    # Invalid JSON file
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("invalid json")
    assert remove_from_json_file(str(bad_json), filt) == []

def test_remove_from_jsonl_file(tmp_path):
    path = tmp_path / "test.jsonl"
    lines = [
        "GUI_DAY_LOG_START:Lookahead + ACO_HH_CUSTOM + Ftsp (Emp),0,1,{}\n",
        "GUI_DAY_LOG_START:Regular + Gurobi + None (Gamma1),0,1,{}\n",
        "NON_STANDARD_LINE\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)

    filt = PolicyFilter(distributions=["emp"])

    # Dry run
    removed = remove_from_jsonl_file(str(path), filt, dry_run=True)
    assert removed == ["Lookahead + ACO_HH_CUSTOM + Ftsp (Emp)"]
    with open(path, "r") as f:
        loaded = f.readlines()
    assert len(loaded) == 3

    # Live run
    removed = remove_from_jsonl_file(str(path), filt, dry_run=False)
    assert removed == ["Lookahead + ACO_HH_CUSTOM + Ftsp (Emp)"]
    with open(path, "r") as f:
        loaded = f.readlines()
    assert len(loaded) == 2
    assert "Lookahead + ACO_HH_CUSTOM + Ftsp (Emp)" not in loaded[0]

    # Non-existent file
    assert remove_from_jsonl_file("nonexistent.jsonl", filt) == []

def test_remove_checkpoint_files(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Create mock checkpoint files
    f1 = ckpt_dir / "checkpoint_lookahead_aco_hh_custom_ftsp_emp_day1.pkl"
    f2 = ckpt_dir / "checkpoint_regular_gurobi_none_gamma1_day2.pkl"
    f3 = ckpt_dir / "other_file.txt"
    f1.write_text("dummy")
    f2.write_text("dummy")
    f3.write_text("dummy")

    filt = PolicyFilter(distributions=["emp"])

    # Dry run
    removed = remove_checkpoint_files(str(ckpt_dir), filt, dry_run=True)
    assert str(f1) in removed
    assert str(f2) not in removed
    assert f1.exists()

    # Live run
    removed = remove_checkpoint_files(str(ckpt_dir), filt, dry_run=False)
    assert str(f1) in removed
    assert not f1.exists()
    assert f2.exists()
    assert f3.exists()

    # Non-existent directory
    assert remove_checkpoint_files("nonexistent_dir", filt) == []

def test_remove_fill_history_files(tmp_path):
    fh_dir = tmp_path / "fill_history"
    fh_dir.mkdir()

    emp_dir = fh_dir / "emp"
    emp_dir.mkdir()
    gamma_dir = fh_dir / "gamma1"
    gamma_dir.mkdir()

    f1 = emp_dir / "lookahead_aco_hh_custom_ftsp_emp42_sample0.xlsx"
    f2 = gamma_dir / "regular_gurobi_none_gamma142_sample1.xlsx"
    f1.write_text("dummy")
    f2.write_text("dummy")

    filt = PolicyFilter(distributions=["emp"])

    # Dry run
    removed = remove_fill_history_files(str(fh_dir), filt, dry_run=True)
    assert str(f1) in removed
    assert str(f2) not in removed
    assert f1.exists()

    # Live run
    removed = remove_fill_history_files(str(fh_dir), filt, dry_run=False)
    assert str(f1) in removed
    assert not f1.exists()
    assert f2.exists()

    # Non-existent directory
    assert remove_fill_history_files("nonexistent_dir", filt) == []

def test_remove_targeted_runs(tmp_path):
    # Setup a complete results structure
    results_dir = tmp_path / "riomaior_100"
    results_dir.mkdir()

    # 1. log_*.json
    log_json = results_dir / "log_lookahead_aco_hh_custom_ftsp_emp_31N.json"
    log_json.write_text(json.dumps({"lookahead_aco_hh_custom_ftsp_emp": {}}))

    # 2. jsonl files
    log_jsonl = results_dir / "log_realtime_emp.jsonl"
    log_jsonl.write_text("GUI_DAY_LOG_START:Lookahead + ACO_HH_CUSTOM + Ftsp (Emp),0,1,{}\n")

    # 3. checkpoints
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_file = ckpt_dir / "checkpoint_lookahead_aco_hh_custom_ftsp_emp_day1.pkl"
    ckpt_file.write_text("dummy")

    # 4. fill history
    fh_dir = results_dir / "fill_history"
    fh_dir.mkdir()
    emp_dir = fh_dir / "emp"
    emp_dir.mkdir()
    fh_file = emp_dir / "lookahead_aco_hh_custom_ftsp_emp42_sample0.xlsx"
    fh_file.write_text("dummy")

    filt = PolicyFilter(distributions=["emp"])

    # Dry run
    removed = remove_targeted_runs(str(results_dir), filt, dry_run=True, verbose=True)
    assert len(removed) == 4
    assert log_json.exists()
    assert ckpt_file.exists()
    assert fh_file.exists()

    # Live run
    removed = remove_targeted_runs(str(results_dir), filt, dry_run=False, verbose=True)
    assert len(removed) == 4
    assert not log_json.exists()
    assert not ckpt_file.exists()
    assert not fh_file.exists()

    # Missing results dir
    assert remove_targeted_runs("nonexistent_dir", filt) == []
