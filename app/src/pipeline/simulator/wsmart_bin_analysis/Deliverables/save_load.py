import pandas as pd

from .container import Container


def save_container_structured(id:int, container:Container, ver:str = None, path:str = None, names:list[str] = None):
    """
    Wrapper to save container information. Names are generated automatically unless 
    specified in the names parameter for which those names are used

    Parameters
    ----------
    container:Container.Container
        container object to be saved
    ver:str, optional
        Aditional distinctive name to add for different 
        versions of container information is set to "" by default
    path:str, optional
        path wehre the files are to be saved. Is set to "" by defualt. 
    name: str, optional
        Names to use instead of the names generator. 
        All should come already with the .csv atached in the ends
    """
    names_, path = verify_names(id, ver, path, names)
    df, recs, info = container.get_vars()

    df.to_csv( path + names_[0], index = True, index_label = recs.index.name)
    recs.to_csv( path + names_[1], index = True, index_label = recs.index.name)
    info.to_csv( path + names_[2])  

def load_container_structured(id:int = None, ver:str = None, path:str = None, names = None) -> Container:
    """
    Load Container structured information froma a csv file that has been created 
    by a save_container function

    Parameters
    ----------
    id:int
        container id to be load
    ver:str
        Aditional distinctive name for name generator to added.
        Is set to "" by default
    path:str
        path wehre the files are to be saved. Is set to "" by defualt. 
    name: str
        Names to use instead of the names generator. 
        All should come already with the .csv atached in the end
    
    Returns
    -------
    res: tuple[int, container]
        returns a tuple with int and continaer object
    """
    names_, path = verify_names(id, ver, path, names)

    df_fill = pd.read_csv(path + names_[0], header = 0)
    df_collection = pd.read_csv(path + names_[1], header = 0)
    info = pd.read_csv(path + names_[2])

    pd.options.mode.chained_assignment = None 
    df_fill['Date'] = pd.to_datetime(df_fill['Date'])
    df_collection['Date'] = pd.to_datetime(df_collection['Date'])
    pd.options.mode.chained_assignment = 'warn'

    c = Container(my_df = df_fill, my_rec = df_collection, info = info)
    return c

def save_id_containers(id_list:list[int], path = None, name = None):
    if path != None:
        if path[-1] != '/':
            path = path + '/'
        if path[0] != '/':
            path = '/' + path
    else:
        path = ""
    
    if name == None:
        name = "ids.csv"
    
    df = pd.Series(id_list, name = "Ids")

    # Save the DataFrame as a CSV file
    df.to_csv(path + name, index=False)

def load_id_containers(path = None, name = "ids.csv") -> list[int]:
    # Adjust path formatting
    if path != None:
        if path[-1] != '/':
            path = path + '/'
        if path[0] != '/':
            path = '/' + path
    else:
        path = ""
    
    # Load the CSV file into a pandas Series
    df = pd.read_csv(path + name)
    df['Ids'] = df['Ids'].astype(int)
    
    # Return the list of IDs as a list
    return df['Ids'].tolist()

def save_rate_series(id:int, container:Container, rate_type:str, freq:str, path:str = None, names:list[str] = None):
    """
    Wrapper to save a Container's rate Time Series. Aditional distinctive name to in automatic naming is set to
    rate_type + rate

    Parameters
    ----------
    container:Container
        Container object o save
    rate_type:str,
        Can be 'mean' or 'crude' to extract the mean monotic aproximation or the crude rate
        (with possible neagtive values). 
    freq:str,
        frequency string indicator. For daily values set to '1D'. For another frequencies check
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    path:str
        path wehre the files are to be saved. Is set to "" by defualt. 
    name: str
        Names to use instead of the names generator. 
        All should come already with the .csv atached in the end
    """
    names_, path = verify_names(id, "rate_"+rate_type, path, names)
    assert rate_type in ['mean', 'crude'] 
    "rate_type can only be 'mean' or 'crude' "

    if rate_type == 'mean':
        rate = container.get_monotonic_mean_rate(freq=freq)
    elif rate_type == 'crude':
        rate = container.get_crude_rate(freq=freq)
    else:
        raise  "rate_type can only be 'mean' or 'crude' "
    rate.to_csv( path + names_[0], index = True, index_label = rate.index.name)

def load_info(id:int = None, ver:str = None, path:str = None, name:str = None) -> pd.DataFrame:
    """
    Load just the Container info fies fro, a csv file that has been created 
    by a save_container function

    Parameters
    ----------
    id:int
        container id to be load
    ver:str
        Aditional distinctive name for name generator to added.
        Is set to "" by default
    path:str
        path wehre the files are to be saved. Is set to "" by defualt. 
    name: str
        Names to use instead of the names generator. 
        All should come already with the .csv atached in the end
    
    Returns
    -------
    res: tuple[int, container]
        returns a tuple with int and continaer object
    """
    names = [None, None, name] if name != None else None
    names_, path = verify_names(id, ver, path, names)
    info = pd.read_csv(path + names_[2])
    return info


def load_rate_series(id:int, rate_type:str, path:str = None, name:str = None) -> dict[int, pd.DataFrame, pd.DataFrame]:
    """
    Load Container structured information froma a csv file that has been created 
    by a save_container function

    Parameters
    ----------
    id:int
        container id to be load
    rate_type:str,
        Can be 'mean' or 'crude' to extract the mean monotic aproximation or the crude rate
        (with possible neagtive values). 
    path:str
        path wehre the files are to be saved. Is set to "" by defualt. 
    name: str
        Names to use instead of the names generator. 
        All should come already with the .csv atached in the end
    """ 
    names = [name, None, None] if name != None else None
    names_, path = verify_names(id, "rate_"+ rate_type, path, names)

    df_rate = pd.read_csv(path + names_[0], header = 0, index_col = 0)
    df_rate.index = pd.to_datetime(df_rate.index)
    return {'id': id, 'data': df_rate}

def load_rate_global_wrapper(rate_list:list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data: list[pd.Series] = []
    info: list[pd.Series] = []
    for tup in rate_list:
        assert isinstance(tup['data']['Rate'], pd.Series),  f"The with id {tup['id']} is not a pandas Series"

        data = data + [tup['data']['Rate']]
        data[-1].name = tup['id']
    df = pd.DataFrame(data[0]).join(data[1:], how='outer',validate='m:m') 
    return df

def verify_names(id:int = None, ver:str = None, path:str = None, names:list[str] = None) -> tuple[list[str], str]:
    """
    Helper function to check containers load/save inputs
    """
    if ver == None:
        ver = ""

    if path != None:
        if path[-1] != '/':
            path = path + '/'
        if path[0] != '/':
            path = '/' + path
    else:
        path = ""
    
    if names != None:
        assert len(names) == 3 
        f"Three files are required to load a container. {len(names)} were specified."
    else:
        assert id != None, "A container id is required to be specified tO use default naming"

        names_ = container_names(id = id, ver=ver)
    return names_, path

def container_names(id:int, ver:str) -> list[str]:
    """
    Helper function to automaically generate containers names
    
    Parameters
    ----------
    id:int
        Container id 
    ver:str
        Aditional distinctive name to add for different 
        versions of container information is set to "" by default
    """
    return ["Container_" + str(id) +"_fill" + ver + ".csv", \
           "Container_" + str(id) +"_recs" + ver + ".csv", \
           "Container_" + str(id) +"_info" + ver + ".csv"]
