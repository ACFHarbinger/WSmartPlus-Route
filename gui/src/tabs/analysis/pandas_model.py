"""
Custom Qt table model for displaying Pandas DataFrames.
"""

from PySide6.QtCore import QAbstractTableModel, Qt


class PandasModel(QAbstractTableModel):
    """
    A read-only table model that allows viewing a Pandas DataFrame in a QTableView.
    """

    def __init__(self, data):
        """
        Initialize the model with a DataFrame.

        Args:
            data (pd.DataFrame): The data to display.
        """
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        """Return the number of rows in the DataFrame."""
        return self._data.shape[0]

    def columnCount(self, parent=None):
        """Return the number of columns in the DataFrame."""
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):  # type: ignore[attr-defined]
        """
        Retrieve data for a specific index and role.

        Args:
            index (QModelIndex): The index of the item.
            role (Qt.ItemDataRole): The role being requested.

        Returns:
            Any: The data at the index, or None if invalid.
        """
        if index.isValid() and role == Qt.DisplayRole:  # type: ignore[attr-defined]
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):  # type: ignore[override]
        """
        Retrieve header data for columns.

        Args:
            col (int): Column index.
            orientation (Qt.Orientation): Horizontal or Vertical.
            role (Qt.ItemDataRole): Data role.

        Returns:
            Any: The column name for horizontal headers.
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:  # type: ignore[attr-defined]
            return self._data.columns[col]
        return None
