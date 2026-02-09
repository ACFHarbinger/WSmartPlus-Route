"""
Worker for asynchronous data loading and processing.
"""

import collections.abc as abc
from typing import Any, List

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot


class DataLoadWorker(QObject):
    """
    Background worker that loads data files (CSV, XLSX, PKL) into Pandas DataFrames.
    Includes heuristic splitting for VRPP/WCVRP problem datasets.
    """

    # Signal to emit the loaded data as a thread-safe list of dicts
    data_loaded = Signal(object)
    error_occurred = Signal(str)

    def _process_data_to_dfs(self, raw_data: Any) -> List[pd.DataFrame]:
        """
        Recursively processes raw data into a flat list of DataFrames.

        Args:
            raw_data: Raw data loaded from file (DataFrame, NumPy array, or list).

        Returns:
            List of Pandas DataFrames extracted from the raw data.
        """
        dfs: List[pd.DataFrame] = []

        if isinstance(raw_data, pd.DataFrame):
            # Transpose DataFrames loaded directly from non-pkl files if needed
            dfs.append(raw_data)

        # --- NEW LOGIC: Direct DataFrame conversion for Lists/Sequences (ragged support) ---
        elif isinstance(raw_data, (list, abc.Sequence)) and not isinstance(raw_data, (str, bytes, np.ndarray)):
            # Try to create a DataFrame directly from the sequence (List of Records)
            # This handles list of tuples (VRPP datasets) and ragged lists gracefully.
            try:
                # We specifically use list() to ensure generators are consumed if any,
                # but mostly to pass to DataFrame constructor which handles list-of-lists/tuples well.
                df = pd.DataFrame(list(raw_data))
                if not df.empty:
                    dfs.append(df)
                    return dfs
            except Exception:
                # If direct conversion fails (unlikely for lists), fall back to numpy logic
                pass

            # Fallback to existing NumPy logic if DataFrame creation didn't return (e.g. error)
            # COPIED EXISTING LOGIC BELOW (slightly refactored flow)
            try:
                array_data = np.array(raw_data)
            except Exception:
                array_data = np.array(raw_data, dtype=object)

            if array_data.dtype == object and array_data.ndim == 1:
                # Recurse for 1D object arrays
                for item in raw_data:
                    dfs.extend(self._process_data_to_dfs(item))
            else:
                dfs.extend(self._process_data_to_dfs(array_data))

        # --- Unified Logic for Array/Sequence Data ---
        elif isinstance(raw_data, np.ndarray):
            if raw_data.ndim >= 3:
                # Iterate over axis 0: shape (N, Rows, Cols)
                num_slices = raw_data.shape[0]
                for i in range(num_slices):
                    # Take the i-th slice and squeeze out the axis of length 1 (axis 0)
                    slice_data = raw_data[i, ...].squeeze()

                    if slice_data.ndim == 2:
                        # Data is now 2D (Rows x Cols)
                        # --- MODIFICATION: APPLY TRANSPOSE (.T) HERE ---
                        dfs.append(pd.DataFrame(slice_data).T)
                    else:
                        raise TypeError(
                            f"Slice {i} of the array did not result in 2D data: "
                            f"{slice_data.shape}. Cannot convert to a table."
                        )

            elif raw_data.ndim == 2:
                # Standard 2D array case (1 table)
                # --- MODIFICATION: APPLY TRANSPOSE (.T) HERE ---
                dfs.append(pd.DataFrame(raw_data).T)
            else:
                raise TypeError(f"Unsupported NumPy array dimension ({raw_data.ndim}) found: {raw_data.shape}")

        else:
            # Catch any other unexpected types
            raise TypeError(f"Unsupported object type found: {type(raw_data)}")

        return dfs

    @Slot(str)
    def load_data_file(self, file_path: str) -> None:
        """
        Loads data in worker thread, converts to DataFrames, and emits.

        Args:
            file_path: Path to the data file to load (CSV, XLSX, or PKL).
        """
        try:
            dfs_to_emit: List[pd.DataFrame] = []

            if file_path.endswith(".csv"):
                # We assume CSV/XLSX are correctly oriented and should NOT be transposed by default.
                dfs_to_emit.append(pd.read_csv(file_path))
            elif file_path.endswith(".xlsx"):
                dfs_to_emit.append(pd.read_excel(file_path))
            elif file_path.endswith(".pkl"):
                raw_data = pd.read_pickle(file_path)
                # This call handles the transposition internally
                dfs_to_emit = self._process_data_to_dfs(raw_data)
            else:
                raise ValueError("Unsupported file type.")

            # Check for VRPP/WCVRP structure (Heuristic: 1 DataFrame with exactly 4 columns)
            # Structure: (depot, loc, waste, max_waste)
            if len(dfs_to_emit) == 1:
                df = dfs_to_emit[0]
                if df.shape[1] == 4:
                    # Heuristic: Check if column 1 looks like locations (list/array)
                    # and column 0 looks like depot.
                    # We can be aggressive here since the user specifically requested this for these files.

                    try:
                        # Explode into 4 separate named tables
                        vrpp_dfs = []

                        # 0: Depot
                        # Expand list column to separate columns
                        depot_df = pd.DataFrame(df[0].tolist(), index=df.index)
                        vrpp_dfs.append(("Depots", depot_df))

                        # 1: Locs
                        # This is Ragged (list of lists of coords)
                        # We keep it as is (Rows=Sample, Cols=Nodes?? No, list in cell)
                        # OR we try to expand?
                        # User wants "one table for locs".
                        # If we expand `df[1].tolist()`, we get N samples x MaxNodes columns.
                        # This is much more useful for analysis than a column of lists.
                        # pd.DataFrame(list_of_lists) automatically handles raggedness (NaN padding).
                        loc_df = pd.DataFrame(df[1].tolist(), index=df.index)
                        # Rename columns to Node_0, Node_1 ...
                        loc_df.columns = [f"Node_{c}" for c in loc_df.columns]
                        vrpp_dfs.append(("Locations", loc_df))

                        # 2: Fill Values
                        # User Request: D fill value tables of N x V size.
                        # Data structure in column 2 is List[D][V] per row.

                        # Get D from the first row
                        first_fill = df.iloc[0, 2]
                        if isinstance(first_fill, list) and len(first_fill) > 0:
                            # Check if it's nested (Multi-day) or flat (Single-day/Legacy?)
                            # If it's list of lists, D = len. If list of scalars, D=1 (or treated as Day 0).
                            # generate_problem_data.py seems to output (N, D, V).
                            # distinct check:
                            if isinstance(first_fill[0], list):
                                num_days = len(first_fill)
                                for d in range(num_days):
                                    # Extract Day d for all rows
                                    # We can use apply, but that's slow? List comprehension is fine.
                                    day_data = [row[d] for row in df[2]]  # list of list of V
                                    day_df = pd.DataFrame(day_data, index=df.index)
                                    day_df.columns = [f"Bin_{c}" for c in day_df.columns]
                                    vrpp_dfs.append((f"Fill Values (Day {d + 1})", day_df))
                            else:
                                # Flat list (maybe D=1 collapsed or old format?)
                                fill_df = pd.DataFrame(df[2].tolist(), index=df.index)
                                fill_df.columns = [f"Bin_{c}" for c in fill_df.columns]
                                vrpp_dfs.append(("Fill Values", fill_df))
                        else:
                            # Empty or weird
                            fill_df = pd.DataFrame(df[2].tolist(), index=df.index)
                            vrpp_dfs.append(("Fill Values", fill_df))

                        # 3: Max Waste
                        # Scalar or list?
                        # Usually scalar per sample, or list if per-bin?
                        # `np.full(dataset_size, MAX_WASTE)` in generator implies scalar.
                        # But `dataset['max_waste']` might be tensor.
                        # Let's try to expand. If scalar, it's 1 column.
                        max_waste_data = df[3].tolist()
                        # Check if items are lists or scalars
                        if len(max_waste_data) > 0 and isinstance(max_waste_data[0], (list, np.ndarray)):
                            mw_df = pd.DataFrame(max_waste_data, index=df.index)
                        else:
                            mw_df = pd.DataFrame(max_waste_data, index=df.index, columns=["Max Waste"])

                        vrpp_dfs.append(("Max Waste", mw_df))

                        # Override the emit list with our new split tables
                        # We need to convert DFs to dicts.
                        thread_safe_list = [(name, d.to_dict(orient="list")) for name, d in vrpp_dfs]
                        self.data_loaded.emit(thread_safe_list)
                        return  # Done, skip default emit

                    except Exception as e:
                        # Fallback if heuristic fails or splitting crashes
                        print(f"VRPP splitting heuristic failed: {e}")

            # Default Emission (original logic + naming)
            # Convert all DataFrames in the list to a list of thread-safe dicts for emission
            # Use default names "Table 1" handled by receiver if name is missing,
            # but we can explicitly pass None key.
            thread_safe_list = [d.to_dict(orient="list") for d in dfs_to_emit]

            self.data_loaded.emit(thread_safe_list)

        except Exception as e:
            self.error_occurred.emit(f"Error loading {file_path}: {str(e)}")
