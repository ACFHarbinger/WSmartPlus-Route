import os
import math
import numpy as np
import googlemaps
import osmnx as ox
import geopandas as gpd

from typing import Iterable
from copy import deepcopy
from geopy.distance import geodesic
from dotenv import dotenv_values
from abc import ABC, abstractmethod
from tqdm import tqdm
from networkx import MultiDiGraph

from logic.src.utils.definitions import ROOT_DIR, EARTH_RADIUS
from logic.src.utils.graph_utils import get_edge_idx_dist, get_adj_knn, idx_to_adj
from logic.src.utils.crypto_utils import decrypt_file_data, load_key


class DistanceStrategy(ABC):
    @abstractmethod
    def calculate(self, coords, **kwargs) -> np.ndarray:
        pass

    def _eval_kwarg(self, kwarg, kwargs):
        return True if kwarg in kwargs and kwargs[kwarg] is not None else False

class GoogleMapsStrategy(DistanceStrategy):
    def calculate(self, coords, **kwargs) -> np.ndarray:
        assert self._eval_kwarg('env_filename', kwargs)
        env_path = os.path.join(ROOT_DIR, "env", kwargs['env_filename'])
        config = dotenv_values(env_path)
        api_key = config.get("GOOGLE_API_KEY", '')
        
        if api_key == '' and self._eval_kwarg('symkey_name', kwargs):
            assert self._eval_kwarg('gapik_file', kwargs)
            sym_key = load_key(kwargs['symkey_name'], kwargs['env_filename'])
            api_key = decrypt_file_data(sym_key, kwargs['gapik_file'])
        elif api_key == '' and self._eval_kwarg('gapik_file', kwargs):
            with open(kwargs['gapik_file'], 'r') as gapik_file:
                api_key = gapik_file.read()
        else:
            assert api_key is not None and api_key != '', "Google API key not found."
            
        gmaps = googlemaps.Client(key=api_key)
        size = len(coords)
        distance_matrix = np.zeros((size, size))
        
        FREE_SIZE = 10
        src = dst = coords[['Lat', 'Lng']].values.tolist()
        
        # Batch processing
        for id_i in range(0, size, FREE_SIZE):
            for id_j in range(0, size, FREE_SIZE):
                origins = src[id_i:id_i+FREE_SIZE]
                dests = dst[id_j:id_j+FREE_SIZE]
                response = gmaps.distance_matrix(origins, dests, mode='driving', units='metric')
                for row_id, row in enumerate(response['rows']):
                    for col_id, elem in enumerate(row['elements']):
                        if 'distance' in elem:
                            distance_matrix[id_i+row_id, id_j+col_id] = elem['distance']['value']/1000
                            
        return distance_matrix

class GeoPandasStrategy(DistanceStrategy):
    def calculate(self, coords, **kwargs) -> np.ndarray:
        # World Geodetic System (https://epsg.io/4326)
        gdf = gpd.GeoDataFrame(
            coords, 
            crs='EPSG:4326', 
            geometry=gpd.points_from_xy(coords['Lng'], coords['Lat'])
        )
        size = len(coords)
        distance_matrix = np.zeros((size, size))
        
        for id_row, row in gdf.iterrows():
            distance_matrix[id_row] = gdf['geometry'].distance(row['geometry']) * 100
            
        return distance_matrix

class OSMStrategy(DistanceStrategy):
    def calculate(self, coords, **kwargs) -> np.ndarray:
        if 'graph' in kwargs and isinstance(kwargs['graph'], MultiDiGraph):
            GG = kwargs['graph']
        elif self._eval_kwarg('download_method', kwargs) and kwargs['download_method'] == 'bbox':
            bounding_box = (coords['Lat'].max(), coords['Lat'].min(), coords['Lng'].max(), coords['Lng'].min())
            GG = ox.graph_from_bbox(bounding_box, network_type='drive')
        else:
            GG = None

        size = len(coords)
        distance_matrix = np.zeros((size, size))
        
        # Iterative calculation
        for id_i, row_i in tqdm(coords.iterrows(), total=size, desc="Outer Loop", disable=not kwargs.get('verbose', False)):
            for id_j, row_j in tqdm(coords.iterrows(), total=size, desc="Inner Loop", leave=False, disable=not kwargs.get('verbose', False)):
                if id_i != id_j:
                    coords_i = (row_i['Lat'], row_i['Lng'])
                    coords_j = (row_j['Lat'], row_j['Lng']) 
                    
                    G = GG if GG is not None else ox.graph_from_point(coords_i, dist=10000, network_type='drive')        
                    bin_i = ox.distance.nearest_nodes(G, coords_i[1], coords_i[0])
                    bin_j = ox.distance.nearest_nodes(G, coords_j[1], coords_j[0])
                    try:
                        length = ox.shortest_path(G, bin_i, bin_j, weight='length')
                        if length:
                            distance_matrix[id_i, id_j] = sum(length)/10_000_000_000
                        else:
                            distance_matrix[id_i, id_j] = float('inf') # Or 0?
                    except Exception:
                        distance_matrix[id_i, id_j] = float('inf')
                        
        return distance_matrix

class IterativeDistanceStrategy(DistanceStrategy):
    @abstractmethod
    def calculate_pair(self, coords_i, coords_j) -> float:
        pass

    def calculate(self, coords, **kwargs) -> np.ndarray:
        size = len(coords)
        distance_matrix = np.zeros((size, size))
        
        verbose = kwargs.get('verbose', False)
        rows = list(coords.iterrows())
        for id_i, row_i in tqdm(rows, total=size, desc="Outer Loop", disable=not verbose):
             coords_i = (row_i['Lat'], row_i['Lng'])
             for id_j, row_j in tqdm(rows, total=size, desc="Inner Loop", leave=False, disable=not verbose):
                 if id_i != id_j:
                     coords_j = (row_j['Lat'], row_j['Lng'])
                     distance_matrix[id_i, id_j] = self.calculate_pair(coords_i, coords_j)
                     
        return distance_matrix

class GeodesicStrategy(IterativeDistanceStrategy):
    def calculate_pair(self, coords_i, coords_j) -> float:
        return geodesic(coords_i, coords_j).km

class HaversineStrategy(IterativeDistanceStrategy):
    def calculate_pair(self, coords_i, coords_j) -> float:
        coords_i_rad = (math.radians(coords_i[0]), math.radians(coords_i[1]))
        coords_j_rad = (math.radians(coords_j[0]), math.radians(coords_j[1]))
        dlat = coords_j_rad[0] - coords_i_rad[0]
        dlng = coords_j_rad[1] - coords_i_rad[1]
        a = math.sin(dlat/2)**2 + math.cos(coords_i_rad[0]) * math.cos(coords_j_rad[0]) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return c * EARTH_RADIUS

class EuclideanStrategy(IterativeDistanceStrategy):
    def calculate_pair(self, coords_i, coords_j) -> float:
        dist = 86.51 * 1.58 * math.sqrt((coords_i[0]-coords_j[0])**2+(coords_i[1]-coords_j[1])**2)
        return round(dist, 10)

def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin((lng2 - lng1)/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c

def compute_distance_matrix(coords, method, **kwargs):
    STRATEGIES = {
        'gmaps': GoogleMapsStrategy,
        'gpd': GeoPandasStrategy,
        'osm': OSMStrategy,
        'gdsc': GeodesicStrategy,
        'hsd': HaversineStrategy,
        'ogd': EuclideanStrategy
    }
    
    assert method in STRATEGIES.keys(), f"Method {method} not supported. usage: {list(STRATEGIES.keys())}"
    
    # Evaluate keyword arguments helper
    def eval_kwarg(kwarg, kwargs): 
        return True if kwarg in kwargs and kwargs[kwarg] is not None else False

    # Caching Logic
    size = len(coords)
    to_save = False
    
    if eval_kwarg('dm_filepath', kwargs):
        filename_only = os.path.basename(kwargs['dm_filepath']) == kwargs['dm_filepath'] and not os.path.isabs(kwargs['dm_filepath'])
        matrix_path = os.path.join(ROOT_DIR, "data", "wsr_simulator", "distance_matrix", kwargs['dm_filepath']) if filename_only \
            else kwargs['dm_filepath']
            
        if os.path.isfile(matrix_path):
            distance_matrix = np.loadtxt(matrix_path, delimiter=',')[1:, 1:]
            if eval_kwarg('focus_idx', kwargs):
                idx = kwargs['focus_idx'][0] if isinstance(kwargs['focus_idx'][0], Iterable) else kwargs['focus_idx']
                idx = np.array([-1] + idx) + 1
                return distance_matrix[idx[:, None], idx]
            else:
                return distance_matrix
        else:
            # Prepare for saving
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True) # Ensure dir exists
            with open(matrix_path, mode='w', newline='') as matrix_f:
                matrix_f.write(",".join(map(str, coords['ID'].to_numpy()))+'\n')
            to_save = True
    
    # Strategy Execution
    strategy_cls = STRATEGIES[method]
    strategy = strategy_cls()
    
    kwargs['verbose'] = to_save 
    
    distance_matrix = strategy.calculate(coords, **kwargs)
    
    # Saving Result
    if to_save:
        with open(matrix_path, mode='a', newline='') as matrix_f: # Append mode since header written
             for row in distance_matrix:
                 matrix_f.write(",".join(map(str, row))+'\n')
                 
    return distance_matrix


def apply_edges(dist_matrix, edge_thresh, edge_method):
    def _make_path(start, end):
        if next_node[start, end] == -1:
            return []  # No path exists
        path = [start]
        while start != end:
            start = next_node[start, end]
            if start == -1:  # Ensure no infinite loops
                return []
            path.append(start)
        return path
    dist_matrix_edges = deepcopy(dist_matrix)
    if edge_thresh > 0 and edge_method == 'dist':
        adj_matrix = idx_to_adj(get_edge_idx_dist(dist_matrix_edges[1:, 1:], edge_thresh))
    elif edge_thresh > 0 and edge_method == 'knn':
        adj_matrix = get_adj_knn(dist_matrix_edges[1:, 1:], edge_thresh, negative=False)
    else:
        adj_matrix = None
    if adj_matrix is not None:
        n_vertices = len(dist_matrix_edges)
        dist_matrix_edges[adj_matrix == 0] = math.inf
        np.fill_diagonal(dist_matrix_edges, np.zeros((n_vertices, n_vertices)))
        next_node = np.full((n_vertices, n_vertices), -1, dtype=int)
        for i in range(n_vertices):
            for j in range(n_vertices):
                if adj_matrix[i][j]:
                    next_node[i][j] = j
                if i == j:
                    next_node[i][j] = i
        for k in range(n_vertices):
            for i in range(n_vertices):
                for j in range(n_vertices):
                    if dist_matrix_edges[i, k] + dist_matrix_edges[k, j] < dist_matrix_edges[i, j]:
                        dist_matrix_edges[i, j] = dist_matrix_edges[i, k] + dist_matrix_edges[k, j]
                        next_node[i, j] = next_node[i, k]

        shortest_paths = {
            (i, j): _make_path(i, j) for i in range(n_vertices) for j in range(n_vertices) if i != j
        }
    else:
        shortest_paths = None
    return dist_matrix_edges, shortest_paths, adj_matrix


def get_paths_between_states(n_bins, shortest_paths=None):
    paths_between_states = []
    for ii in range(0,n_bins):
        paths_between_states.append([])
        for jj in range(n_bins):
            if shortest_paths is None or ii == jj:
                paths_between_states[ii].append([ii, jj])
            else:
                paths_between_states[ii].append(shortest_paths[(ii, jj)])
    return paths_between_states

