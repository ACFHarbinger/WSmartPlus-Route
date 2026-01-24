"""
End-to-end smoke tests for CLI commands.
"""

import subprocess
import sys

import pytest


@pytest.mark.e2e
def test_cli_help():
    """Test that main.py --help runs successfully."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout or "Usage:" in result.stdout


@pytest.mark.e2e
@pytest.mark.parametrize("problem", ["vrpp"])
def test_cli_gen_data_smoke(tmp_path, problem):
    """Smoke test for data generation."""
    output_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "gen_data",
            "--dataset_type",
            "test_simulator",
            "--problem",
            problem,
            "--data_distributions",
            "unif",
            "--graph_sizes",
            "10",
            "--dataset_size",
            "2",
            "--data_dir",
            str(output_dir),
            "--name",
            "smoke_test",
            "-f",
        ],
        capture_output=True,
        text=True,
    )

    # Check for success
    if result.returncode != 0:
        with open("cli_gen_data_error.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        assert (
            result.returncode == 0
        ), f"gen_data failed with return code {result.returncode}. See cli_gen_data_error.log"

    # Expected filename for test_simulator: {area}{size}_{dist}_{name}_N{dataset_size}_seed{seed}.pkl
    # area defaults to 'riomaior', seed defaults to 42
    # riomaior10_unif_smoke_test_N2_seed42.pkl
    expected_file = output_dir / "riomaior10_unif_smoke_test_N2_seed42.pkl"
    assert expected_file.exists(), f"Generated data file not found at {expected_file}"


@pytest.mark.e2e
def test_cli_train_lightning_smoke():
    """Smoke test for training loop."""
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "train_lightning",
            "env.name=vrpp",
            "env.num_loc=10",
            "train.n_epochs=1",
            "train.batch_size=2",
            "train.eval_batch_size=2",
            "train.train_data_size=10",
            "train.val_data_size=10",
            "wandb_mode=offline",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0


@pytest.mark.e2e
def test_cli_eval_smoke():
    """Smoke test for evaluation."""
    # This assumes a model exists or we just test the command structure
    # Since we might not have a trained model, we check if it fails gracefully
    # or runs a dummy eval if possible.
    # For now, let's just check help/arg parsing
    result = subprocess.run(
        [sys.executable, "main.py", "eval", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
