import os
import json
import pandas as pd
import app.src.utils.definitions as udef


# Load data
def load_indices(filename, n_samples, n_nodes, data_size, lock=None):
    graphs_file_path = os.path.join(udef.ROOT_DIR, "data", "wsr_simulator", 'bins_selection', filename)
    if os.path.isfile(graphs_file_path):
        if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
        try:
            with open(graphs_file_path) as fp:
                indices = json.load(fp)
        finally:
            if lock is not None: lock.release()
            if len(indices) == 1 and n_samples > 1: indices *= n_samples
    else:
        df = pd.Series(range(data_size))
        indices = []
        for _ in range(n_samples):
            data = df.sample(n=n_nodes).to_list() #pd.concat([pd.Series([0]), df.sample(n=n_nodes)])
            data.sort()
            while len(indices) > 0 and data in indices:
                data = df.sample(n=n_nodes).to_list() #pd.concat([pd.Series([0]), df.sample(n=n_nodes)]).to_list()
                data.sort()
            indices.append(data)

        if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
        try:
            with open(graphs_file_path, 'w') as fp:
                fp.write("[\n")
                fp.write(",\n".join(json.dumps(idx) for idx in indices))  # Serialize each list properly
                fp.write("\n]")
                #json.dump(indices, fp)
        finally:
            if lock is not None: lock.release()
    return indices


def load_depot(data_dir, area='Rio Maior'):
    src_area = area.translate(str.maketrans('', '', '-_ ')).lower()
    facilities = pd.read_csv(os.path.join(data_dir, 'coordinates', 'Facilities.csv'))
    depot_df = facilities[facilities['Sigla'] == udef.MAP_DEPOTS[src_area]] \
                        .loc[:, ['Lat', 'Lng']] \
                        .reset_index(drop=True)
    depot_df.insert(0, 'ID', [0])
    new_cols = pd.DataFrame({'Stock': [0], 'Accum_Rate': [0]})
    return pd.concat([depot_df, new_cols], axis=1)


def load_simulator_data(data_dir, number_of_bins, area='Rio Maior', waste_type=None, lock=None):
    def _preprocess_county_date(data):
        data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")
        data = data.set_index('Date')
        data = data.round()
        data.columns = data.columns.astype(int)
        return data
    
    def _preprocess_county_data(data):
        def __get_stock(col):
            positive_values = col[col >= 1e-32].dropna() # Find the first index where value is >= 1e-32
            if not positive_values.empty:
                return positive_values.iloc[0] 
            return 0 # If no positive data found (or if all data is negative/0)
        # Get waste accumulation rate (mean) and stock (first value >= 0) of bins
        accum_rate = data.clip(lower=0).fillna(0).mean()
        stock = data.apply(__get_stock, axis=0)
        new_data = pd.DataFrame(data.columns, columns=['ID'])
        new_data['Stock'] = new_data['ID'].map(stock)
        new_data['Accum_Rate'] = new_data['ID'].map(accum_rate)
        new_data[['Stock', 'Accum_Rate']] = ((new_data[['Stock', 'Accum_Rate']] - new_data[['Stock', 'Accum_Rate']].min()) / 
                                            (new_data[['Stock', 'Accum_Rate']].max() - new_data[['Stock', 'Accum_Rate']].min()))
        return new_data

    src_area = area.translate(str.maketrans('', '', '-_ ')).lower()
    wtype = waste_type if waste_type is None else udef.WASTE_TYPES[waste_type]
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        if src_area == 'mixrmbac':
            if number_of_bins <= 20:
                data = pd.read_excel(os.path.join(data_dir, 'bins_waste', 'StockAndAccumulationRate - small.xlsx'))
                bins_coordinates = pd.read_excel(os.path.join(data_dir, 'coordinates', 'Coordinates - small.xlsx'))
            elif number_of_bins <= 50 and number_of_bins > 20:
                data = pd.read_excel(os.path.join(data_dir, 'bins_waste', 'StockAndAccumulationRate - 50bins.xlsx')) 
                bins_coordinates = pd.read_excel(os.path.join(data_dir, 'coordinates', 'Coordinates - 50bins.xlsx'))
            else:
                assert number_of_bins <= 225, f"Number of bins for area {src_area} must be <= 225"
                data = pd.read_excel(os.path.join(data_dir, 'bins_waste', 'StockAndAccumulationRate.xlsx'))
                bins_coordinates = pd.read_excel(os.path.join(data_dir, 'coordinates', 'Coordinates.xlsx'))
        elif src_area == 'riomaior':
            data = _preprocess_county_date(pd.read_csv(os.path.join(data_dir, 'bins_waste', f"old_out_crude_rate[{src_area}].csv")))
            assert number_of_bins <= 317, f"Number of bins for area {src_area} must be <= 317"
            coords_tmp = pd.read_csv(os.path.join(data_dir, 'coordinates', f'old_out_info[{src_area}].csv'))
            coords_tmp = coords_tmp.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lng'})
            if wtype: coords_tmp = coords_tmp[coords_tmp['Tipo de Residuos'] == wtype]
            bins_coordinates = coords_tmp[['ID', 'Lat', 'Lng']]
            data = _preprocess_county_data(data)
            data = data[data['ID'].isin(bins_coordinates['ID'])]
        elif src_area == 'figueiradafoz':
            data = _preprocess_county_date(pd.read_csv(os.path.join(data_dir, "out_crude_rate[figdafoz].csv")))
            assert number_of_bins <= 1094, f"Number of bins for area {src_area} must be <= 1094"
            coords_tmp = pd.read_csv(os.path.join(data_dir, 'coordinates', f'out_info[figdafoz].csv'))
            coords_tmp = coords_tmp.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lng'})
            if wtype: coords_tmp = coords_tmp[coords_tmp['Tipo de Residuos'] == wtype]
            bins_coordinates = coords_tmp[['ID', 'Lat', 'Lng']]
            data = _preprocess_county_data(data)
            data = data[data['ID'].isin(bins_coordinates['ID'])]
        else:
            assert src_area == 'both', f"Invalid area: {src_area}"
            wsrs_data = pd.read_excel(os.path.join(data_dir, 'bins_waste', 'StockAndAccumulationRate.xlsx'))
            wsba_data = _preprocess_county_data(_preprocess_county_date(
                pd.read_csv(os.path.join(data_dir, 'bins_waste', f"old_out_crude_rate[{src_area}].csv")))
            )
            if number_of_bins <= 57:
                coords_tmp = pd.read_csv(os.path.join(data_dir, 'coordinates', 'intersection.csv'))
                bins_coordinates = coords_tmp.drop('ID317', axis=1).rename(columns={'ID225': 'ID'})
                data = wsrs_data[wsrs_data['ID'].isin(bins_coordinates['ID'])]
            elif number_of_bins <= 371:
                coords_tmp = pd.read_csv(os.path.join(data_dir, 'coordinates', 'merged.csv'))
                data = wsrs_data[wsrs_data['ID'].isin(coords_tmp['ID225'])]
                data_tmp = wsba_data[wsba_data['ID'].isin(coords_tmp.loc[coords_tmp['ID225'].isna(), 'ID317'])]
                data = pd.concat([data, data_tmp], axis=0)
                bins_coordinates = pd.DataFrame({'ID': coords_tmp['ID225'].fillna(coords_tmp['ID317']), 
                                                'Lat': coords_tmp['Lat'], 'Lng': coords_tmp['Lng']})
            elif number_of_bins <= 485:
                coords_tmp = pd.read_csv(os.path.join(data_dir, 'coordinates', 'union.csv'))
                data = wsba_data[wsba_data['ID'].isin(coords_tmp['ID317'])]
                data_tmp = wsrs_data[wsrs_data['ID'].isin(coords_tmp.loc[coords_tmp['ID317'].isna(), 'ID225'])]
                data = pd.concat([data, data_tmp], axis=0)
                bins_coordinates = pd.DataFrame({'ID': coords_tmp['ID317'].fillna(coords_tmp['ID225']), 
                                                'Lat': coords_tmp['Lat'], 'Lng': coords_tmp['Lng']})
            else:
                assert number_of_bins <= 542, f"Number of bins for {src_area} must be <= 542"
                bins_coordinates = pd.read_csv(os.path.join(data_dir, 'coordinates', f'old_out_info[{src_area}].csv'))
                bins_coordinates = bins_coordinates.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lng'})
                bins_coordinates = bins_coordinates[['ID', 'Lat', 'Lng']]
                coords_tmp = pd.read_excel(os.path.join(data_dir, 'Coordinates.xlsx'))
                data = wsba_data[wsba_data['ID'].isin(bins_coordinates['ID'])]
                data_tmp = wsrs_data[wsrs_data['ID'].isin(coords_tmp['ID'])]

                # Change ID since bin with same ID (but diff coords) exists in out_INFO.csv
                coords_tmp.loc[coords_tmp['ID'] == 1610, 'ID'] = coords_tmp.iloc[-1]['ID'] + 1
                data_tmp.loc[data_tmp['ID'] == 1610, 'ID'] = data_tmp.iloc[-1]['ID'] + 1
                bins_coordinates = pd.concat([bins_coordinates, coords_tmp])
                data = pd.concat([data, data_tmp])
            data = data[data['ID'].isin(bins_coordinates['ID'])]
        bins_coordinates = bins_coordinates[bins_coordinates['ID'].isin(data['ID'])]
    finally:
        if lock is not None: lock.release()
    return data.sort_values(by='ID').reset_index(drop=True), bins_coordinates.sort_values(by='ID').reset_index(drop=True)


def load_area_and_waste_type_params(area, waste_type):
    expenses = 1 # travelling cost per travel unit (in € per KM)
    bin_volume = 2.5 # maximum waste storage capacity per bin (in m^3)
    src_area = area.translate(str.maketrans('', '', '-_ ')).lower()
    if waste_type == 'paper':
        revenue = 0.65 * 250/1000
        if src_area in ['riomaior', 'mixrmbac']:
            density = 21.0
            vehicle_capacity = 4000
        else:
            assert src_area == 'figueiradafoz', "Unknown waste collection area: {}".format(src_area)
            density = 32.0
            vehicle_capacity = 3000
    elif waste_type == 'plastic':
        revenue = 0.65 * 898/1000
        if src_area in ['riomaior', 'mixrmbac']:
            density = 19.0
            vehicle_capacity = 3500
        else:
            assert src_area == 'figueiradafoz', "Unknown waste collection area: {}".format(src_area)
            density = 20.0
            vehicle_capacity = 2500
    else:
        assert waste_type == 'glass', "Unknown waste type: {}".format(waste_type)
        revenue = 0.90 * 84/1000
        if src_area in ['riomaior', 'mixrmbac']:
            density = 190.0
            vehicle_capacity = 9000
        else:
            assert src_area == 'figueiradafoz', "Unknown waste collection area: {}".format(src_area)
            density = 200.0
            vehicle_capacity = 8000
    return (vehicle_capacity, revenue, density, expenses, bin_volume) # KG, $/KG, KG/m^3, $/KM, m^3
