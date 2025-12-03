import os
import json
import torch
import numpy as np
import pandas as pd

from .loader import load_area_and_waste_type_params
from src.utils.definitions import (
    MAX_WASTE, MAX_LENGTHS,
    EARTH_RADIUS, EARTH_WMP_RADIUS,
)
from src.utils.graph_utils import adj_to_idx, get_edge_idx_dist, get_adj_knn


def sort_dataframe(df, metric_tosort, ascending_order=True):
    df = df.sort_values(by=metric_tosort, ascending=ascending_order)
    columns = [metric_tosort] + [col for col in df.columns if col != metric_tosort]
    return df[columns]


# Process (and sample from) Pandas Dataframes
def get_df_types(df, prec="32"):
    df_types = dict(df.dtypes)
    for key, val in df_types.items():
        if key == 'ID':
            new_type = f'int{prec}'
        elif "obj" in str(val):
            new_type = "string" 
        else:
            new_type = str(val)[:-2] + prec
        df_types[key] = new_type
    return df_types


def setup_df(depot, df, col_names, index_name="#bin"):
    df = df.loc[:, col_names]
    df.loc[-1] = depot.loc[0, col_names].values
    df.index = df.index + 1
    df = df.sort_index()
    if index_name is None:
        df = df.sort_values(by='ID').reset_index(drop=True).astype(get_df_types(df))
        #df = df.sort_index().reset_index().astype(get_df_types(df)).drop('index', axis=1)    
    else:
        df = df.sort_values(by='ID').reset_index().astype(get_df_types(df))
        df = df.rename(columns={'index': index_name})
        df[index_name] = df[index_name].astype(df['ID'].dtype)
    return df


def sample_df(df, n_elems, depot=None, output_path=None):
    df = df.sample(n=n_elems)
    df_types = get_df_types(df)
    if depot is not None:
        df.loc[0] = depot
    if output_path is not None:
        if os.path.isfile(output_path):
            with open(output_path) as fp:
                data = json.load(fp)
            data.append(df.sort_index().index.tolist())
        else:
            data = [df.sort_index().index.tolist()]
        with open(output_path, 'w') as fp:
            json.dump(data, fp)
    df = df.sort_values(by='ID').reset_index(drop=True).astype(df_types)
    return df


# Setup data for simulation(s)
def process_indices(df, indices):
    if indices is None:
        df = df.copy()
    else:
        if 'index' in df.columns or 'ID' in df.columns:
            df = df.iloc[indices]
            df = df.sort_values(by='ID').reset_index(drop=True).astype(get_df_types(df))
        else:
            df = df[df.columns[indices]]
    return df


def process_data(data, bins_coordinates, depot, indices=None):
    new_data = process_indices(data, indices)
    coords = process_indices(bins_coordinates, indices)
    coords = setup_df(depot, coords, ['ID', 'Lat', 'Lng'])
    new_data = setup_df(depot, new_data, ['ID', 'Stock', 'Accum_Rate'])
    return new_data, coords


def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin((lng2 - lng1)/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c


def process_coordinates(coords, method, col_names=['Lat', 'Lng']):
    assert method in ['mmn', 'mun', 'smsd', 'ecp', 'utmp', 'wmp', 'hdp', 'c3d', 's4d']

    IS_PANDAS = col_names is not None
    lat = coords[col_names[0]] if IS_PANDAS else coords[:, :, 0]
    lng = coords[col_names[1]] if IS_PANDAS else coords[:, :, 1]
    if method == 'c3d': # Conversion to 3D Cartesian coordinates
        latr = np.radians(lat)
        lngr = np.radians(lng)
        x_axis = EARTH_RADIUS * np.cos(latr) * np.cos(lngr)
        y_axis = EARTH_RADIUS * np.cos(latr) * np.sin(lngr)
        z_axis = EARTH_RADIUS * np.sin(latr)
        if IS_PANDAS:
            x_axis = (x_axis - x_axis.min())/(x_axis.max() - x_axis.min())
            y_axis = (y_axis - y_axis.min())/(y_axis.max() - y_axis.min())
            z_axis = (z_axis - z_axis.min())/(z_axis.max() - z_axis.min())
            depot = np.array([x_axis.iloc[0], y_axis.iloc[0], z_axis.iloc[0]])
            loc = np.array([[x, y, z] for x, y, z in zip(x_axis.iloc[1:], y_axis.iloc[1:], z_axis.iloc[1:])])
        else:
            coords_3d = np.stack((x_axis, y_axis, z_axis), axis=-1)
            min_arr = np.min(coords_3d, axis=1, keepdims=True)
            max_arr = np.max(coords_3d, axis=1, keepdims=True)
            coords_3d = (coords_3d - min_arr) / (max_arr - min_arr)
            depot = coords_3d[:, 0, :]
            loc = coords_3d[:, 1:, :]
    elif method == 's4d': # Conversion to 4D spherical coordinates
        latr = np.radians(lat)
        lngr = np.radians(lng)
        lats, latc = np.sin(latr), np.cos(latr)
        lngs, lngc = np.sin(lngr), np.cos(lngr)
        if IS_PANDAS:
            lats = (lats - lats.min())/(lats.max() - lats.min())
            latc = (latc - latc.min())/(latc.max() - latc.min())
            lngs = (lngs - lngs.min())/(lngs.max() - lngs.min())
            lngc = (lngc - lngc.min())/(lngc.max() - lngc.min())
            depot = np.array([lats.iloc[0], latc.iloc[0], lngs.iloc[0], lngc.iloc[0]])
            loc = np.array([[x, y, z, w] for x, y, z, w in zip(lats.iloc[1:], latc.iloc[1:], lngs.iloc[1:], lngc.iloc[1:])])
        else:
            coords4d = np.stack([lats, latc, lngs, lngc], axis=-1)
            min_arr = np.min(coords4d, axis=1, keepdims=True)
            max_arr = np.max(coords4d, axis=1, keepdims=True)
            coords4d = (coords4d - min_arr) / (max_arr - min_arr)
            depot = coords4d[:, 0, :]
            loc = coords4d[:, 1:, :]
    else:
        if method == 'mun': # Mean (μ) normalization
            if IS_PANDAS:
                lat = (lat - lat.mean())/(lat.max() - lat.min())
                lng = (lng - lng.mean())/(lng.max() - lng.min())
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                mean_arr = np.mean(coords, axis=1, keepdims=True)
                coords = (coords - mean_arr) / (max_arr - min_arr)
        elif method == 'smsd': # Standardization (using μ and σ)
            if IS_PANDAS:
                lat = (lat - lat.mean()) / lat.std()
                lng = (lng - lng.mean()) / lng.std()
            else:
                coords = coords[:, :, [1, 0]]
                mean_arr = np.mean(coords, axis=1, keepdims=True)
                std_arr = np.std(coords, axis=1, keepdims=True)
                coords = (coords - mean_arr) / std_arr
        elif method == 'ecp': # Equidistant cylindrical (aka Equirectangular/Geographic) projection
            if IS_PANDAS:
                center_meridian = (lng.max() + lng.min()) / 2
                per_func = lambda lat, percent: np.percentile(lat, percent)
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                center_meridian = (max_arr + min_arr) / 2
                per_func = lambda lat, percent: np.percentile(lat, percent, axis=1, keepdims=True)
            lat_lower, lat_upper = per_func(lat, 10), per_func(lat, 90)
            center_parallel = (lat_upper + lat_lower) / 2
            offset = (lat_upper - lat_lower) / 2
            pscale = (np.cos(np.radians(center_parallel - offset)) + np.cos(np.radians(center_parallel + offset))) / 2
            lat = EARTH_RADIUS * (np.radians(lat) - np.radians(center_parallel))
            lng = EARTH_RADIUS * (np.radians(lng) - np.radians(center_meridian)) * pscale
        elif method == 'utmp': # Universal Transverse Mercator projection
            raise NotImplementedError
            transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_epsg(5018), always_xy=True)
            if IS_PANDAS:
                lng, lat = transformer.transform(lng.to_numpy(), lat.to_numpy())
                lat, lng = pd.Series(lat), pd.Series(lng)
            else:
                lng_flat, lat_flat = transformer.transform(lng.ravel(), lat.ravel())
                coords = np.stack((lat_flat.reshape(lat.shape), lng_flat.reshape(lng.shape)), axis=-1)
        elif method == 'wmp': # World Mercator projection
            lng = EARTH_WMP_RADIUS * np.radians(lng)
            lat = EARTH_WMP_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
            if not IS_PANDAS:
                coords = np.stack((lat, lng), axis=-1)
        elif method == 'hdp': # Haversine distance projection
            if IS_PANDAS:
                lat_max, lat_min = lat.max(), lat.min()
                lng_max, lng_min = lng.max(), lng.min()
                max_func = lambda h1, h2: max(h1, h2)
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                lng_min, lat_min = np.split(min_arr, 2, axis=-1)
                lng_max, lat_max = np.split(max_arr, 2, axis=-1)
                max_func = lambda h1, h2: np.max(np.concatenate((h1, h2)), axis=0, keepdims=True)
            mid_lat = (lat_max + lat_min) / 2
            mid_lng = (lng_max + lng_min) / 2
            max_distance = EARTH_RADIUS * np.pi / 180 * max_func(haversine_distance(lat_min, lng_min, lat_max, lng_max),
                                                                 haversine_distance(lat_min, lng_max, lat_max, lng_min))
            lat = (lat - mid_lat) / max_distance
            lng = (lng - mid_lng) / max_distance
        else:
            assert method == "mmn" # Min-Max normalization
            if IS_PANDAS:
                lat = (lat - lat.min())/(lat.max() - lat.min())
                lng = (lng - lng.min())/(lng.max() - lng.min())
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                coords = (coords - min_arr) / (max_arr - min_arr)
        if IS_PANDAS:
            depot = np.array([lng.iloc[0], lat.iloc[0]])
            loc = np.array([[x, y] for x, y in zip(lng.iloc[1:], lat.iloc[1:])])
        else:
            depot = coords[:, 0, :]
            loc = coords[:, 1:, :]
    return (depot, loc)


def process_model_data(coordinates, dist_matrix, device, method, configs, 
                    edge_threshold, edge_method, area, waste_type, adj_matrix=None):
    problem_size = len(dist_matrix) - 1
    depot, loc = process_coordinates(coordinates, method)
    model_data = {
        'loc': torch.as_tensor(loc, dtype=torch.float32),
        'depot': torch.as_tensor(depot, dtype=torch.float32),
        'waste': torch.zeros(problem_size)
    }
    if configs['problem'] in ['vrpp', 'wcrp']:
        #cw_dict = {'waste': configs['w_waste'], 'length': configs['w_length'], 'overflows': configs['w_overflows'], 'lost': configs['w_lost']}
        model_data['max_waste'] = torch.as_tensor(MAX_WASTE, dtype=torch.float32)
    else:
        assert configs['problem'] == 'op'
        #cw_dict = {'prize': configs['w_prize']}
        model_data['max_length'] = torch.as_tensor(MAX_LENGTHS[problem_size], dtype=torch.float32)

    if 'model' in configs and configs['model'] in ['tam']:
        model_data['fill_history'] = torch.zeros((1, configs['graph_size'], configs['temporal_horizon']))

    if edge_threshold > 0 and edge_threshold < 1:
        if edge_method == 'dist':
            edges = torch.tensor(adj_to_idx(adj_matrix, negative=False)) if adj_matrix is not None \
                    else torch.tensor(get_edge_idx_dist(dist_matrix[1:, 1:], edge_threshold))
        else:
            assert edge_method == 'knn'
            #adj_matrix = get_adj_osm(coordinates, problem_size, args, negative=False)
            edges = torch.from_numpy(adj_matrix) if adj_matrix is not None \
                else torch.from_numpy(get_adj_knn(dist_matrix[1:, 1:], edge_threshold, negative=False))
            #edges = torch.tensor(adj_to_idx(neg_adj_matrix)).to(device, dtype=torch.long)
        dtype = torch.float32 if 'encoder' in configs and configs['encoder'] in ['gac', 'tgc'] else torch.bool
        edges = edges.unsqueeze(0).to(device, dtype=dtype)
    else:
        edges = None
    
    VEHICLE_CAPACITY, REVENUE_KG, DENSITY, COST_KM, VOLUME = load_area_and_waste_type_params(area, waste_type)
    BIN_CAPACITY = VOLUME * DENSITY
    profit_vars = {
        'cost_km': COST_KM,
        'revenue_kg': REVENUE_KG,
        'bin_capacity': BIN_CAPACITY
    }
    return ({key: val.unsqueeze(0) for key, val in model_data.items()}, 
        (edges, torch.from_numpy(dist_matrix).float().to(device)), profit_vars)


def create_dataframe_from_matrix(matrix):
    # Extrai o último elemento de cada sub-array da matriz
    enchimentos = [row[-1] for row in matrix]

    # Gera os índices (bins) como a sequência de números, igual ao índice de cada linha
    ids_rota = np.arange(len(matrix))

    # Cria o DataFrame com as colunas '#bin', 'Stock' e 'Accum_Rate'
    data = pd.DataFrame({
        '#bin': ids_rota,  # O índice de cada linha é utilizado como 'bin'
        'Stock': enchimentos,  # Últimos elementos de cada sub-array
        'Accum_Rate': np.zeros(len(ids_rota))  # Inicializa 'Accum_Rate' com 0
    })
    return data


def convert_to_dict(bins_coordinates):
    # Cria um dicionário vazio
    coordinates_dict = {}
    
    # Itera sobre as linhas do DataFrame
    for _, row in bins_coordinates.iterrows():
        # Extrai os valores ID, Lat e Lng
        bin_id = row['ID']
        lat = np.float64(row['Lat'])
        lng = np.float64(row['Lng'])
        
        # Adiciona o par chave-valor no dicionário
        coordinates_dict[bin_id] = (lat, lng)
    return coordinates_dict
