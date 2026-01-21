"""
Unit tests for data extraction and pre-processing in the wsmart_bin_analysis module.
"""

import numpy as np
import pandas as pd

from ..Deliverables.extract import (
    container_global_sorted_wrapper,
    import_same_file,
    pre_process_data,
)


class TestExtract:
    """
    Test suite for data extraction tools, including SARIMA pre-processing and file imports.
    """

    def test_pre_process_data(self):
        """Test pre-processing of raw fill and collection dataframes into aligned structures."""
        # Create raw dataframes imitating what comes from CSV
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        df_fill = pd.DataFrame(
            {
                "DateStr": dates.strftime("%Y-%m-%d %H:%M:%S"),
                "BinID": [101] * 10,
                "Level": np.arange(10) * 10,
            }
        )

        df_coll = pd.DataFrame({"DateColl": dates[::2].strftime("%Y-%m-%d %H:%M:%S"), "BinID": [101] * 5})

        fill, collect, info = pre_process_data(
            df_fill=df_fill,
            df_collection=df_coll,
            id_header_fill="BinID",
            date_header_fill="DateStr",
            date_format_fill="%Y-%m-%d %H:%M:%S",
            fill_header_fill="Level",
            id_header_collect="BinID",
            date_header_collect="DateColl",
            date_format_collect="%Y-%m-%d %H:%M:%S",
            start_date="01/01/2019",  # Cover the range
            end_date="01/01/2021",
        )

        assert "ID" in fill.columns
        assert "ID" in collect.columns
        assert "Fill" in fill.columns
        assert "Date" in fill.columns
        assert "Date" in collect.columns
        assert 101 in fill["ID"].values
        assert 101 in collect["ID"].values
        assert not info.empty

    def test_import_same_file(self, tmp_path):
        """Test importing data where fill and collection records share the same file."""
        # Create a dummy CSV
        p = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "Date": ["2020-01-01", "2020-01-02"],
                "Fill": [50, 60],
                "IsColl": [np.nan, 1],
            }
        )
        df.to_csv(p, index=False)

        fill_df, rec_df = import_same_file(
            src_fill="data.csv",
            collect_id_header="IsColl",
            path=str(tmp_path),
            print_first_line=False,
        )

        assert len(fill_df) == 2
        assert len(rec_df) == 1
        assert rec_df["IsColl"].iloc[0] == 1

    def test_container_global_sorted_wrapper(self, sample_df_fill, sample_df_collection, sample_info):
        """Test the wrapper that instantiates multiple Container objects from dataframes."""
        # They need to be aligned and cleaned first usually, but let's try with fixtures
        # Ensure IDs match
        sample_df_fill["ID"] = 1
        sample_df_collection["ID"] = 1
        sample_info["ID"] = 1

        res, ids = container_global_sorted_wrapper(sample_df_fill, sample_df_collection, sample_info)

        assert 1 in ids
        assert 1 in res
        assert res[1] is not None
