"""
A bundle of tools to pre-process diferent import sources
used thoughout the project. After pre-processing the data, create container
objects that you can be easily saved and loaded to separate csv files and compute predictions or correct
collections as you go.
"""

from typing import cast

import pandas as pd

from .container import Container


def pre_process_data(
    df_fill: pd.DataFrame,
    df_collection: pd.DataFrame,
    id_header_fill: str,
    date_header_fill: str,
    date_format_fill: str,
    fill_header_fill: str,
    id_header_collect: str,
    date_header_collect: str,
    date_format_collect: str,
    start_date: str = "01/01/2020",
    end_date: str = "01/01/2025",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pre-processes the data-frames obtained from the import function to obtain
    a separate but equal representation for every container independant of the source

    It selects appropriate columns and renames them, drops duplicate data,
    asserts the necessary datatype and keeps the intersection between the existing data
    between fill data and collection data.

    The info dictionary will keep the reamining columns with the value presented ont he first line
    of the dataframe.

    Colection and fill will sorted dataframes which can interated through. As the ids where intersected,
    one can iterate through both grouby by their order without needing to earch for a specific id
    because they would be aligned.

    Parameters
    ----------

    df_fill : pd.Dataframe
        Crude fill data pandas dataframe after importing csv data.
    df_collection: pd.Dataframe
        Crude collection data pandas dataframe after importing csv data.
    date_header_fill: string
        Name of the column with the date timestamp in the orginal df with fill info.
    date_format_fill: string
        Date timestamp format in the original df with fill info e.g. "%Y-%m-%d %H:%M:%S.%f"
        for 2020-05-01 23:06:28.000.
    fill_header_fill: string
        Name of the column with the fill data the orginal df with fill info.
    date_header_collect: string
        Name of the column with the date timestamp in the orginal df with collection info.
    date_format_collect: string
        Date timestamp format in the original df with collection info e.g,
    round: bool
        Weather to to round the fill information to the floor of the corresponding hour.
        This is important because driver readings have the same timestamp has collections and
        reveal the fill level before collection. Se to False for Sensor data.
    end/start_date: string with format %d/%m/%Y.
        Period to be analysed. First day is inclusive staring at 00:01 last day is exclusive.
        Set to 01/01/2020 - 01/01/2025 by default
    """
    # Renaming
    rename_keys_fill = {
        date_header_fill: "Date",
        id_header_fill: "ID",
        fill_header_fill: "Fill",
    }
    rename_keys_collect = {date_header_collect: "Date", id_header_collect: "ID"}

    df_fill.rename(columns=rename_keys_fill, errors="raise", inplace=True)
    df_collection.rename(columns=rename_keys_collect, errors="raise", inplace=True)

    # Keep just the instersection
    id = df_fill["ID"].drop_duplicates()
    id = id[id.isin(df_collection["ID"])]
    df_fill = cast(pd.DataFrame, df_fill[df_fill["ID"].isin(id)])
    df_collection = cast(pd.DataFrame, df_collection[df_collection["ID"].isin(id)])

    # Take care of datatypes
    pd.options.mode.chained_assignment = None

    df_fill["Date"] = pd.to_datetime(df_fill["Date"], format=date_format_fill, errors="raise")
    df_collection["Date"] = pd.to_datetime(df_collection["Date"], format=date_format_collect, errors="raise")

    pd.options.mode.chained_assignment = "warn"

    df_fill.loc[:, "ID"] = df_fill["ID"].astype(int, errors="raise")
    df_collection.loc[:, "ID"] = df_collection["ID"].astype(int, errors="raise")
    df_fill.loc[:, "Fill"] = df_fill["Fill"].astype(int, errors="raise")

    # get characterizing data
    info = df_collection.drop_duplicates(subset="ID")
    info.drop(["Date"], axis=1)

    df_fill = df_fill.loc[:, ["Date", "ID", "Fill"]]
    df_collection = df_collection.loc[:, ["Date", "ID"]]

    df_collection.dropna(inplace=True)
    df_fill.dropna(inplace=True)

    # Select required window
    df_collection = cast(
        pd.DataFrame,
        df_collection[
            (df_collection["Date"] > pd.to_datetime(start_date, format="%d/%m/%Y"))
            & (df_collection["Date"] < pd.to_datetime(end_date, format="%d/%m/%Y"))
        ],
    )
    df_fill = cast(
        pd.DataFrame,
        df_fill[
            (df_fill["Date"] > pd.to_datetime(start_date, format="%d/%m/%Y"))
            & (df_fill["Date"] < pd.to_datetime(end_date, format="%d/%m/%Y"))
        ],
    )

    # sort by ID and by Date in each ID
    df_fill.sort_values(["ID", "Date"], inplace=True)
    df_collection.sort_values(["ID", "Date"], inplace=True)

    # drop_duplicates()
    df_fill.drop_duplicates(inplace=True)
    df_collection.drop_duplicates(inplace=True)

    # keep just the date intersection
    fill = df_fill.groupby("ID")
    collect = df_collection.groupby("ID")

    df_date_intersect = pd.DataFrame(
        {
            "f1": fill["Date"].first(),
            "c1": collect["Date"].first(),
            "flast": fill["Date"].last(),
            "clast": collect["Date"].last(),
        }
    )

    df_date_intersect["upper_bound"] = df_date_intersect[["flast", "clast"]].min(axis=1)
    df_date_intersect["lower_bound"] = df_date_intersect[["f1", "c1"]].max(axis=1)

    def filter_by_bounds(group_df):
        """
        Filter rows in the group based on calculated date bounds.

        Args:
            group_df (pd.DataFrame): The dataframe group to filter.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        bounds = df_date_intersect.loc[group_df.name]  # Get bounds for the current group
        return group_df[(group_df["Date"] >= bounds["lower_bound"]) & (group_df["Date"] <= bounds["upper_bound"])]

    fill = fill.apply(filter_by_bounds).reset_index(drop=True)
    collect = collect.apply(filter_by_bounds).reset_index(drop=True)

    # reset as integers because soem are NaNs now
    idsf = fill["ID"].to_numpy().astype(int)
    idsc = collect["ID"].to_numpy().astype(int)

    fill.drop("ID", axis=1, inplace=True)
    collect.drop("ID", axis=1, inplace=True)

    fill["ID"] = idsf
    collect["ID"] = idsc

    # Filter DataFrames to keep only common IDs
    fill = fill[fill["ID"].isin(collect["ID"])].dropna(subset=["ID"])
    collect = collect[collect["ID"].isin(fill["ID"])].dropna(subset=["ID"])
    return fill, collect, info


def import_separate_file(
    src_fill: list[str],
    src_collect: list[str],
    sep_f: str = ",",
    sep_c: str = ",",
    path: str = "",
    print_firt_line=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    import method for when the collections and fill is in separate files.
    Used to import driver and sensor data from Rio maior, and 2020/2021 data from Figeuria da Foz
    The sources must be in a .csv format
    The output is the imported joined dataframes from each source file
    The pre-processing is done in the pre-processing function

    Set print_first_line to False not to print helpful infomration

    Parameters
    ----------
    src_fill: list[strtins]
        List of name of file strings to import fill data from
    src_collect: list[strtins]
        List of name of file strings to import collection data from
    path: str
        path where the files are. The deafult is set to ""
    sep_f: str
        Separation character for the csv of fill data. Default is ','
    sep_c: str
        Separation character for the csv of collect data. Default is ','
    print_first_line: bool
        Weather to print the first line of each dataframe to infer keys
        and formats for the next steps
    """
    if len(path) > 0:
        if path[-1] != "/":
            path = path + "/"
        if path[0] != "/":
            path = "/" + path

    ##get data
    fill_df = [pd.read_csv(path + name, sep=sep_f, encoding_errors="replace") for name in src_fill]
    rec_df = [pd.read_csv(path + name, sep=sep_c, encoding_errors="replace") for name in src_collect]

    fill_df = pd.concat(fill_df, ignore_index=True)
    rec_df = pd.concat(rec_df, ignore_index=True)
    if print_firt_line:
        print("FILL DF LINE 1: ")
        print(fill_df.iloc[0])
        print("REC DF LINE 1: ")
        print(rec_df.iloc[0])
    return fill_df, rec_df


def import_same_file(
    src_fill: str,
    collect_id_header: str,
    sep: str = ",",
    path: str = "",
    print_first_line=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    import method for when the collections and fill is in the same file files.
    this will separate collections and fill inforation in separate dataframes
    Parameters
    ----------
    src_fill: str
        List of name of file strings to import fill data from
    collect_id_header:str
        Header key of the column indicating weater the entry is a colletion or not. Should be Nan
        in reading events and not NaN in collection  event
    sep: str
        Separation character for the csv. Default is ','.
    path: str
        path where the files are. The deafult is set to ""
    print_first_line: bool
        Weather to print the first line of each dataframe to infer keys
        and formats for the next steps
    """
    if len(path) > 0:
        if path[-1] != "/":
            path = path + "/"
        if path[0] != "/":
            path = "/" + path

    fill_df: pd.DataFrame = pd.read_csv(path + src_fill, sep=sep, encoding_errors="replace")
    assert collect_id_header in list(
        fill_df.keys()
    ), f"'{collect_id_header}' is not a valid key. Available keys are: {list(fill_df.keys())}"

    rec_df = cast(pd.DataFrame, fill_df[~fill_df[collect_id_header].isna()])
    rec_df = rec_df.copy(deep=True)
    if print_first_line:
        print("FILL DF LINE 1: ")
        print(fill_df.iloc[0])
        print("REC DF LINE 1: ")
        print(rec_df.iloc[0])
    return fill_df, rec_df


def container_global_sorted_wrapper(
    fill_: pd.DataFrame, collect_: pd.DataFrame, info: pd.DataFrame
) -> tuple[dict[int, Container], list[int]]:
    """
    Wrapper to iterate through containers dataframes and create a list of container classes and preforms initial
    pre-processing.
    Assumes fill and collect have the container ids that are sorted.

    Parameters
    ----------
    fill: DataFrameGroupBy
        groupby object with fill info for each container
    collect: DataFrameGroupBy
        groupby obejct with collection info
    info: pd.Dataframe
        dataframe relative other information regarding each container
    path: str
        path string where to save container info

    Returns
    -------
    res: dict
        returns a dictionary  with keys id and container objects.
    ids: list[int]
        returns the list of ids contemplated in the dictionary
    """
    fill = fill_.groupby("ID")
    collect = collect_.groupby("ID")

    res = {}
    for fi in fill:
        c = Container(fi[1], collect.get_group(fi[0]), info.loc[info["ID"] == fi[0], :])
        try:
            c.mark_collections()
            res[fi[0]] = c
        except Exception:
            pass
    return res, list(res.keys())
