"""
File System Utility Module.

This module provides high-level utilities for managing the WSmart+ project's
file system, including:
1. Cryptographic operations: Key generation, encryption, and decryption.
2. Entry updates: Batch processing of file contents based on patterns.
3. Entry deletion: Cleanup of logs, models, datasets, and cache directories.

It acts as a CLI entry point for administrative and data-management tasks.
"""

import argparse
import os
import shutil
import sys
import traceback
from typing import Any, Dict

from logic.src.cli import ConfigsParser, add_files_args, validate_file_system_args

from .utils.crypto_utils import (
    decrypt_file_data,
    encrypt_file_data,
    generate_key,
    load_key,
)
from .utils.definitions import ROOT_DIR
from .utils.io_utils import (
    confirm_proceed,
    preview_changes,
    preview_file_changes,
    preview_file_statistics,
    preview_pattern_files_statistics,
    process_file,
    process_file_statistics,
    process_pattern_files,
    process_pattern_files_statistics,
)


def perform_cryptographic_operations(opts: Dict[str, Any]) -> None:
    """
    Perform cryptographic tasks based on the provided options.

    Tasks include:
    - Key generation: If no input path is provided, generates a new symmetric key.
    - Encryption: Encrypts a file using a stored key and verifies the result.

    Args:
        opts (Dict[str, Any]): Configuration options containing paths, key names,
            and encryption parameters.

    Raises:
        Exception: If any cryptographic operation fails.
    """
    try:
        if "input_path" not in opts or opts["input_path"] is None:
            _, _ = generate_key(
                opts["salt_size"],
                opts["key_length"],
                opts["hash_iterations"],
                opts["symkey_name"],
                opts["env_file"],
            )
        else:
            key = load_key(opts["symkey_name"], opts["env_file"])
            inpath = os.path.join(ROOT_DIR, opts["input_path"])
            outpath = (
                os.path.join(ROOT_DIR, opts["output_path"])
                if "output_path" in opts and opts["output_path"]
                else os.path.join(ROOT_DIR, opts["input_path"] + ".enc")
            )
            _ = encrypt_file_data(key, inpath, outpath)
            dec_data = decrypt_file_data(key, outpath)
            with open(inpath, "r") as gp_file:
                data = gp_file.read()
            assert dec_data == data
    except Exception as e:
        raise Exception(f"failed to perform cryptographic operations due to {repr(e)}")


def update_file_system_entries(opts: Dict[str, Any]) -> int:
    """
    Apply content updates or statistical processing to files/directories.

    Logic:
    1. Identifies the target entry (file or directory).
    2. Optional preview: Shows changes/statistics and asks for confirmation.
    3. Execution: Processes files matching a pattern or a specific target path.

    Args:
        opts (Dict[str, Any]): Configuration options containing target paths,
            filename patterns, update operations, and preview flags.

    Returns:
        int: 1 if successful, 0 if cancelled by user.

    Raises:
        ValueError: If target path does not exist.
        Exception: If file processing fails.
    """
    try:
        target_path = os.path.join(ROOT_DIR, opts["target_entry"])
        process_stats = "stats_function" in opts and opts["stats_function"] is not None
        if os.path.isdir(target_path):
            if opts["update_preview"]:
                if process_stats:
                    preview_pattern_files_statistics(
                        target_path,
                        opts["filename_pattern"],
                        opts["output_filename"],
                        opts["output_key"],
                        opts["stats_function"],
                    )
                else:
                    preview_changes(
                        target_path,
                        opts["output_key"],
                        opts["filename_pattern"],
                        opts["update_operation"],
                        opts["update_value"],
                        opts["input_keys"],
                    )
                if not confirm_proceed(operation_name="directory update"):
                    print("Operation cancelled by user.")
                    return 0
            if process_stats:
                process_pattern_files_statistics(
                    target_path,
                    opts["filename_pattern"],
                    opts["output_filename"],
                    opts["output_key"],
                    opts["stats_function"],
                )
            else:
                process_pattern_files(
                    target_path,
                    opts["filename_pattern"],
                    opts["output_key"],
                    opts["update_operation"],
                    opts["update_value"],
                    opts["input_keys"],
                )
        elif os.path.isfile(target_path):
            if opts["update_preview"]:
                if process_stats:
                    preview_file_statistics(
                        target_path,
                        opts["output_filename"],
                        opts["output_key"],
                        opts["stats_function"],
                    )
                else:
                    preview_file_changes(
                        target_path,
                        opts["output_key"],
                        opts["update_operation"],
                        opts["update_value"],
                        opts["input_keys"],
                    )
                if not confirm_proceed(operation_name="file update"):
                    print("Operation cancelled by user.")
                    return 0
            if process_stats:
                process_file_statistics(
                    target_path,
                    opts["output_filename"],
                    opts["output_key"],
                    opts["stats_function"],
                )
            else:
                process_file(
                    target_path,
                    opts["output_key"],
                    opts["update_operation"],
                    opts["update_value"],
                    opts["input_keys"],
                )
        else:
            raise ValueError(f"no file or directory found with path '{target_path}'")
        return 1
    except Exception as e:
        raise Exception(f"failed to update file system entries due to {repr(e)}")


def delete_file_system_entries(opts: Dict[str, Any]) -> int:
    """
    Batch delete project-related directories based on provided flags.

    Supported categories:
    - WandB logs, training logs, model outputs.
    - Datasets, evaluation results, test sim outputs/checkpoints.
    - Local cache directories.

    Args:
        opts (Dict[str, Any]): Configuration options containing boolean flags for
            each directory category and directory names.

    Returns:
        int: 0 after completion or cancellation.

    Raises:
        Exception: If listing or deletion operations fail.
    """
    try:
        directories_to_delete = []
        if opts.get("wandb"):
            wandb_path = os.path.join(ROOT_DIR, "wandb")
            if os.path.exists(wandb_path):
                directories_to_delete.append(("wandb logs", wandb_path))

        if opts.get("log"):
            log_path = os.path.join(ROOT_DIR, opts["log_dir"])
            if os.path.exists(log_path):
                directories_to_delete.append(("train logs", log_path))

        if opts.get("output"):
            output_path = os.path.join(ROOT_DIR, opts["output_dir"])
            if os.path.exists(output_path):
                directories_to_delete.append(("model outputs", output_path))

        if opts.get("data"):
            data_path = os.path.join(ROOT_DIR, opts["data_dir"])
            if os.path.exists(data_path):
                directories_to_delete.append(("datasets", data_path))

        if opts.get("eval"):
            eval_path = os.path.join(ROOT_DIR, opts["eval_dir"])
            if os.path.exists(eval_path):
                directories_to_delete.append(("evaluation results", eval_path))

        if opts.get("test_sim"):
            test_sim_path = os.path.join(ROOT_DIR, "assets", opts["test_sim_dir"])
            if os.path.exists(test_sim_path):
                directories_to_delete.append(("test sim outputs", test_sim_path))

        if opts.get("test_checkpoint"):
            test_sim_checkpoint_path = os.path.join(ROOT_DIR, "assets", opts["test_sim_checkpoint_dir"])
            if os.path.exists(test_sim_checkpoint_path):
                directories_to_delete.append(("test sim checkpoints", test_sim_checkpoint_path))

        if opts.get("cache"):
            cache_path1 = os.path.join(ROOT_DIR, "cache")
            cache_path2 = os.path.join(ROOT_DIR, "notebooks", "cache")
            if os.path.exists(cache_path1):
                directories_to_delete.append(("main cache", cache_path1))
            if os.path.exists(cache_path2):
                directories_to_delete.append(("notebooks cache", cache_path2))

        if not directories_to_delete:
            print("No directories exist for deletion based on the provided options.")
            return 0

        if opts.get("delete_preview"):
            print("\nThe following directories exist and will be deleted:")
            print("-" * 60)
            for i, (description, path) in enumerate(directories_to_delete, 1):
                print(f"{i}. {description}:")

            if not confirm_proceed(operation_name="deletion"):
                print("Deletion cancelled by user.")
                return 0

        print("\nDeleting directories...")
        success_count = 0
        for description, path in directories_to_delete:
            try:
                shutil.rmtree(path, ignore_errors=True)
                if not os.path.exists(path):
                    print(f"- Successfully deleted: {description}")
                    success_count += 1
                else:
                    print(f"- Failed to delete: {description}")
            except Exception as e:
                print(f"- Error deleting {description}: {e}")

        print(f"\nDeletion completed: {success_count}/{len(directories_to_delete)} directories removed successfully.")
        return 0
    except Exception as e:
        raise Exception(f"failed to delete file system entries due to {repr(e)}")


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="File System Utility Runner (update/delete/cryptography)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_files_args(parser)
    try:
        parsed_args = parser.parse_process_args(sys.argv[1:], "file_system")
        comm, opts = validate_file_system_args(parsed_args)

        if comm == "update":
            update_file_system_entries(opts)
        elif comm == "delete":
            delete_file_system_entries(opts)
        else:
            assert comm == "cryptography"
            perform_cryptographic_operations(opts)

    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        parser.print_help()
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(str(e), file=sys.stderr)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
