import pandas as pd
import numpy as np 
import collections.abc as abc

from PySide6.QtCore import QObject, Signal, Slot


class DataLoadWorker(QObject):
    # Signal to emit the loaded data as a thread-safe list of dicts
    data_loaded = Signal(object) 
    error_occurred = Signal(str)

    def _process_data_to_dfs(self, raw_data):
        """Recursively processes raw data (DataFrame, NumPy array, or converted list) into a flat list of DataFrames."""
        dfs = []

        if isinstance(raw_data, pd.DataFrame):
            # Transpose DataFrames loaded directly from non-pkl files if needed
            dfs.append(raw_data)
            
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
                        raise TypeError(f"Slice {i} of the array did not result in 2D data: {slice_data.shape}. Cannot convert to a table.")

            elif raw_data.ndim == 2:
                # Standard 2D array case (1 table)
                # --- MODIFICATION: APPLY TRANSPOSE (.T) HERE ---
                dfs.append(pd.DataFrame(raw_data).T)
            else:
                raise TypeError(f"Unsupported NumPy array dimension ({raw_data.ndim}) found: {raw_data.shape}")

        # --- CRITICAL FIX: Convert list/sequence to NumPy array first ---
        elif isinstance(raw_data, abc.Sequence) and not isinstance(raw_data, str):
            # Convert the list (or other sequence) into a NumPy array.
            array_data = np.array(raw_data)
            
            # If the resulting array is a 1D object array (often due to irregular contents like mixed-size arrays), 
            # we must iterate over the original list items.
            if array_data.dtype == object and array_data.ndim == 1:
                for item in raw_data:
                    dfs.extend(self._process_data_to_dfs(item))
            else:
                 # Otherwise, the conversion resulted in a proper multi-dimensional array, 
                 # so we process the resulting array.
                 dfs.extend(self._process_data_to_dfs(array_data))
        # --- END CRITICAL FIX ---

        else:
            # Catch any other unexpected types
            raise TypeError(f"Unsupported object type found: {type(raw_data)}")
            
        return dfs

    @Slot(str)
    def load_data_file(self, file_path):
        """Loads data in the worker thread, converts slices to DataFrames, then to a thread-safe list of dicts, and emits it."""
        try:
            dfs_to_emit = []
            
            if file_path.endswith('.csv'):
                # We assume CSV/XLSX are correctly oriented and should NOT be transposed by default.
                dfs_to_emit.append(pd.read_csv(file_path))
            elif file_path.endswith('.xlsx'):
                dfs_to_emit.append(pd.read_excel(file_path))
            elif file_path.endswith('.pkl'):
                raw_data = pd.read_pickle(file_path)
                # This call handles the transposition internally
                dfs_to_emit = self._process_data_to_dfs(raw_data)
            else:
                raise ValueError("Unsupported file type.")
            
            # Convert all DataFrames in the list to a list of thread-safe dicts for emission
            thread_safe_list = [df.to_dict(orient='list') for df in dfs_to_emit]
            
            self.data_loaded.emit(thread_safe_list)
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading {file_path}: {str(e)}")