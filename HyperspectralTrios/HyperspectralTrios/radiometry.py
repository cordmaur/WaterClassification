from pathlib import Path
import pandas as pd
import numpy as np
import math
from .common import from_excel_date, plot_reflectances, apply_subplot, create_interpolated_columns, \
    listify, to_excel_date, plot_mean_std_traces

import configparser
from plotly import subplots
import plotly.io as pio

from datetime import timedelta
import shutil


class BaseRadiometry:

    @staticmethod
    def backup(fn):
        if fn.exists():
            if not fn.with_suffix('.bak').exists():
                shutil.copy(fn, fn.with_suffix('.bak'))
        else:
            print(f'File {fn} does not exists to be backed up')

    @staticmethod
    def check_file(path, file_name):
        """Check if a file exists in the given path"""
        path = Path(path)

        if path.exists():
            for f in path.iterdir():
                if file_name in f.name:
                    return f

        return False

    @staticmethod
    def get_file_by_suffix(path, stem, suffixes=None):
        """Get the file that matches the suffix in the order of preference of suffixes"""
        if suffixes is None:
            suffixes = ['.txt', '.mlb']

        for suffix in suffixes:
            f = BaseRadiometry.check_file(path, stem + suffix)
            if f:
                return f

        return False

    @staticmethod
    def open_trios_radiometry(file):
        file = Path(file)
        # print(f'Opening radiometry {file}')
        # Get the data frames from the file
        if file.suffix in ['.txt', '.bak']:
            metadata = pd.read_csv(file, header=None, sep='\t', nrows=18).drop(columns=1)
            rdmtry = pd.read_csv(file, sep='\t', skiprows=19, header=1, parse_dates=True, skip_blank_lines=True)

        elif file.suffix == '.mlb':
            metadata = pd.read_fwf(file, widths=[1, 19, 9, 50], header=None, nrows=18).drop(columns=[0, 2])
            metadata = metadata.rename(columns={1: 0, 3: 2})
            rdmtry = pd.read_fwf(file, skiprows=19, header=1, engine='python', skip_blank_lines=True)
            # return rdmtry, metadata
        else:
            # Suffix doesn't implemented
            print(f'Open radiometry function cannot open {file.suffix} suffix')
            return None, None

        # Adjust metadata (header of the file)
        for i in range(2, len(metadata.columns)):
            metadata[2] = metadata[2] + ' ' + metadata[i + 1].replace(np.nan, '')
            metadata = metadata.drop(columns=i + 1)

        metadata[2] = metadata[2].str.strip()
        metadata.rename(columns={0: 'key', 2: 'value'}, inplace=True)

        # Adjust the radiometry measurements
        rdmtry = rdmtry.drop(columns=['NaN.1', 'NaN.2', 'NaN.3']).rename(columns={'NaN': 'DateTime'})

        # Convert the DateTime from excel to understandable format
        rdmtry['DateTime'] = rdmtry['DateTime'].map(from_excel_date)
        rdmtry.sort_values(by='DateTime', inplace=True)
        rdmtry.set_index('DateTime', inplace=True)

        return rdmtry, metadata

    @staticmethod
    def get_radiances(folder,  prefixes, sep='_', base_name='spectrum_LO', suffixes=None):

        radiances = {}
        metadatas = {}

        for prefix in prefixes:
            # get the filename corresponded by each prefix
            fn = BaseRadiometry.get_file_by_suffix(folder, stem=(prefix + sep + base_name), suffixes=suffixes)

            if fn:
                rdmtry, meta = BaseRadiometry.open_trios_radiometry(fn)

                radiances.update({prefix: rdmtry})
                metadatas.update({prefix: meta})
            else:
                print(f'{prefix}:Raw not found in {folder}')

        return radiances, metadatas


# #############  Radiometry Class  ##################
class Radiometry:

    labels = {'Ed': {'y_axis': "Irradiance (mW/(m^2))",
                     'title': 'Irradiance (Ed)'},
              'Ld': {'y_axis': 'Radiance (mW/(m^2 sr))',
                     'title': 'Radiance (Ld)'},
              'Lu': {'y_axis': 'Radiance (mW/(m^2 sr))',
                     'title': 'Radiance (Lu)'},
              'Rrs': {'y_axis': 'Reflectance (sr^-1)',
                      'title': 'Reflectance (Rrs)'}}

    default_prefixes = ['Rrs', 'Ed', 'Ld', 'Lu']

    def __init__(self, radiances, metadata, interp_radiances, folder=None):
        self.radiances = radiances
        self.metadata = metadata
        self.folder = folder
        self.interp_radiances = interp_radiances

        self._subset = None

    @classmethod
    def from_folder(cls, folder, prefixes=None, sep='_', base_name='spectrum_LO', read_backup=False,
                    load_interpolated=False, base_interpolated_name='_interpolated'):
        """Open the radiometry on a specific folder. If prefixes are not specified, open the basic [Lu, Ld, Ed, Rrs]"""

        # set the prefixes to be loaded. If not informed, use the default prefixes
        prefixes = Radiometry.default_prefixes if prefixes is None else prefixes

        # convert the folder into a Path object
        folder = Path(folder)

        # If read_backup, try to load first the .bak extension
        suffixes = ['.bak', '.txt', '.mlb'] if read_backup else None

        # Read the radiances and the metadata
        radiances, metadatas = BaseRadiometry.get_radiances(folder, prefixes=prefixes, sep=sep,
                                                            base_name=base_name, suffixes=suffixes)

        # create also an interpolated radiances repository (dictionary)
        interp_radiances = {}
        if load_interpolated:
            for prefix in prefixes:
                fn = folder/(prefix + base_interpolated_name)
                fn = fn.with_suffix('.bak') if read_backup else fn.with_suffix('.csv')

                if fn.exists():
                    rdmtry = pd.read_csv(fn, sep=';')
                    rdmtry['DateTime'] = pd.to_datetime(rdmtry['DateTime'])
                    interp_radiances.update({prefix: rdmtry.set_index('DateTime')})

        metadatas.update({'Metadata': cls.open_metadata(folder)})

        rdmtry = cls(radiances, metadatas, interp_radiances, folder)
        return rdmtry

    @staticmethod
    def open_metadata(folder):
        file = Path(folder)/'metadata.txt'
        if not file.exists():
            # print(f'Metadata file not found in {str(folder)}')
            return None
        else:
            config = configparser.ConfigParser()
            config.read(file)
            return config

    def interpolate_radiometries(self, r_types=None, step=1, min_wl=320, max_wl=950):
        r_types = ['Ed', 'Lu', 'Ld'] if r_types is None else listify(r_types)

        for r_type in r_types:
            if r_type is None:
                continue

            rd = create_interpolated_columns(self[r_type],
                                             create_id=False,
                                             step=step,
                                             min_col=min_wl,
                                             max_col=max_wl)

            rd = rd._get_numeric_data()

            rd.columns = rd.columns.map(lambda x: str(x))
            self.interp_radiances.update({r_type: rd})

    def create_reflectance(self, ro=0.028, step=1, min_wl=320, max_wl=950, ed='Ed', lu='Lu', ld='Ld'):

        r_types = [ed, lu, ld]
        self.interpolate_radiometries(r_types=r_types, step=step, min_wl=min_wl, max_wl=max_wl)

        ed = self.interp_radiances[ed]
        lu = self.interp_radiances[lu]
        ld = self.interp_radiances[ld] if ld is not None else 0

        # Calculate the reflectance
        r_rs = (lu - ro * ld) / ed

        r_rs.dropna(axis=0, how='all', inplace=True)
        self.interp_radiances.update({'Rrs': r_rs})

    def get_radiometry(self, r_type, use_subset=False, interpolated=False):

        radiances = self.radiances if not interpolated else self.interp_radiances

        # First, let's check if the radiometry is already loaded
        if r_type not in radiances:
            print(f"Radiometry {r_type}:{'interpolated' if interpolated else 'Raw'} not available."
                  f"Loaded: {list(self.radiances.keys())}")
            return None

        else:
            if use_subset and self.subset is not None:
                return radiances[r_type].loc[self.subset]
            else:
                return radiances[r_type]

    # ##########  PLOTTING METHODS  #############
    def plot_radiometry(self, r_type='Rrs', use_subset=True, interpolated=True, mean=False, std_delta=1., **kwargs):
        subset = self.subset if use_subset else None

        # get the radiometry DataFrame
        df = self.get_radiometry(r_type, interpolated=interpolated, use_subset=use_subset)

        if df is None:
            return

        numeric_columns = df._get_numeric_data().columns

        color = df.index if not mean else None

        fig = plot_reflectances(df, numeric_columns, color=color, colorbar=False)

        # if mean, get the figure with mean and standard deviation
        if mean:
            mean_traces = plot_mean_std_traces(df, numeric_columns, shaded=True, std_delta=std_delta)
            for trace in mean_traces:
                fig.add_trace(trace)

        if r_type in self.labels:
            fig.update_layout(title=self.labels[r_type]['title'],
                              yaxis_title=self.labels[r_type]['y_axis'])

        return fig

    def plot_radiometries(self, r_types=None, cols=2, base_height=400, use_subset=True, interpolated=True, **kwargs):
        """
        Plot all the radiometries that are loaded.
        :param r_types: The radiometry names ex. ['reflectance', 'Ed']. If None plot all loaded radiometries.
        :param cols: Number of columns
        :param base_height: Height for each figure row
        :param use_subset:
        :param interpolated: Indicate if it should plot Interpolated (default) or Raw radiometries.
        :return: Multi-axis figure
        """

        if interpolated:
            r_types = self.interp_radiances.keys() if r_types is None else listify(r_types)
        else:
            r_types = self.radiances.keys() if r_types is None else listify(r_types)

        if len(r_types) == 0:
            print(f'No radiance found to plot')
            return

        # get the number of rows
        n = len(r_types)
        rows = math.ceil(n/cols)

        # get the titles of the graphs
        titles = list(r_types)

        # create the main figure
        fig = subplots.make_subplots(rows=rows, cols=cols,
                                     subplot_titles=titles)

        for idx, name in enumerate(r_types):
            position = ((idx // cols) + 1, (idx % cols) + 1)
            subplot = self.plot_radiometry(name, interpolated=interpolated, use_subset=use_subset, **kwargs)
            if subplot is not None:
                apply_subplot(fig, subplot, position)

        fig.update_layout(height=base_height * rows)
        return fig

    # ##########  DATETIME METHODS  #############
    @property
    def subset(self):
        return self._subset # if self._subset is not None else self.times

    @subset.setter
    def subset(self, values):
        self._subset = values

    @property
    def times(self):
        times = None
        for df in self.radiances.values():
            times = df.index if times is None else times.union(df.index)
        return times

    def times_range(self, string=True):
        times = self.times

        if string:
            return str(times.min()), str(times.max())
        else:
            return times.min(), times.max()

    # ##########  FILTERING METHODS  #############
    def apply_time_filter(self, start, end=None, interval=timedelta(minutes=5), accumulate=False):
        """
        Creates a subset of times that can be used for displaying/exporting the measurements (through the
        flag `use_subset`).
        :param start: time window start datetime in 'yyyy-mm-dd hh:mm:ss' (str) format
        :param end: time window end datetime in 'yyyy-mm-dd hh:mm:ss' (str) format.
        If None, a interval will be applied to the start time.
        :param interval: A timedelta parameter that's used when end is not passed.
        :param accumulate: If accumulate is False, the filter is always applied to the entire times range,
        otherwise, it is being accumulated every time the method is called. This can be used to fine-tune
        the curves to be exported.
        :return: The resulting datetimes found in the database.
        """

        start = pd.to_datetime(start)
        if end is None:
            if interval is None:
                print(f'Either end or interval arguments should be specified')
            elif not isinstance(interval, timedelta):
                print('Interval should be passed as a timedelta variable')
            else:
                end = start + interval
                start = start - interval
        else:
            end = pd.to_datetime(end)

        if accumulate and self.subset is not None:
            self.subset = self.subset[(self.subset >= start) & (self.subset <= end)]
        else:
            self.subset = self.times[(self.times >= start) & (self.times <= end)]

        return self.subset

    def apply_value_filter(self, r_type, min_value=-np.inf, max_value=np.inf, accumulate=False):

        df = self.get_radiometry(r_type, interpolated=True, use_subset=accumulate)

        self.subset = df[(df.max(axis=1) > min_value) & (df.max(axis=1) < max_value)].index
        return self.subset

    # ##########  IN/OUT METHODS  #############
    def _save_radiance(self, r_type, use_subset=True, save_backup=True):
        fn = self.folder / (r_type + '_spectrum_LO.txt')
        meta = self.metadata[r_type].copy()

        measures = self.get_radiometry(r_type, use_subset=use_subset, interpolated=False)

        meta.insert(1, 'Symbol', '=')
        header_txt = meta.to_csv(sep='\t', line_terminator='\n', header=False, index=False)

        # Before writing to txt, convert the dates to Excel format
        measures.index = measures.index.map(to_excel_date)
        measures.insert(0, 'IntegrationTime', int(meta[meta['key']=='IntegrationTime']['value'].values))
        measures.insert(0, 'PositionLongitude', 0)
        measures.insert(0, 'PositionLatitude', 0)

        measures_txt = measures.to_csv(sep='\t', line_terminator='\n', na_rep='NaN')

        for title in ['DateTime', 'PositionLatitude', 'PositionLongitude', 'IntegrationTime', 'Comment', 'IDData']:
            measures_txt = measures_txt.replace(title, 'NaN')

        c_header = [f'c{str(i).zfill(3)}' for i in range(1, len(measures.columns) - 3 + 1)]
        c_header = ['DateTime', 'PositionLatitude', 'PositionLongitude', 'IntegrationTime'] + c_header
        c_header = c_header + ['Comment', 'IDData']

        c_header = '\t'.join(c_header)

        txt = header_txt + '\n' + c_header + '\n' + measures_txt

        if save_backup:
            BaseRadiometry.backup(fn)

        with open(fn, "w") as text_file:
            text_file.write(txt)

    def _save_interpolated_radiance(self, r_type, use_subset=True, save_backup=True,
                                    interpolated_name='_interpolated.csv', sep=';'):

        rd = self.get_radiometry(r_type, use_subset=use_subset, interpolated=True)

        # create the filename
        fn = self.folder/(r_type + interpolated_name)

        # backup
        if save_backup:
            BaseRadiometry.backup(fn)

        # save to csv
        rd.to_csv(fn, sep=sep)

    def save_radiometry(self, r_type, use_subset=True, save_interpolated=True, save_backup=True, sep=';'):

        if r_type in self.radiances:
            self._save_radiance(r_type, use_subset=use_subset, save_backup=save_backup)

        if save_interpolated and r_type in self.interp_radiances:
            self._save_interpolated_radiance(r_type, use_subset=use_subset, save_backup=save_backup, sep=sep)

    def save_radiometries(self, use_subset=True, save_interpolated=True, save_backup=True, sep=';'):

        # get the r_types to be saved (everything)
        r_types = set().union(self.radiances.keys(), self.interp_radiances.keys())

        for r_type in r_types:
            self.save_radiometry(r_type, use_subset=use_subset, save_interpolated=save_interpolated,
                                 save_backup=save_backup, sep=sep)

    def save_radiometries_graph(self, folder=None, use_subset=True, mean=False, **kwargs):
        folder = Path(folder) if folder is not None else self.folder
        fig = self.plot_radiometries(use_subset=use_subset, mean=mean, **kwargs)

        fn = f'Fig_{self.folder.stem}.png'

        print(f'Saving image {fn} into: {folder}')
        pio.write_image(fig, str(folder/fn), width=2000, height=1200, validate=False)#, engine='kaleido')

    # ##########  SPECIAL METHODS  #############
    def __getitem__(self, r_type):
        """
        Return a specific radiometry. By default, the interpolated is returned,
        if it is not found, return the Raw.
        :param r_type: name of the radiometry to be obtained
        :return: DataFrame with the desired radiometry
        """
        if r_type in self.interp_radiances:
            return self.get_radiometry(r_type, interpolated=True)
        elif r_type in self.radiances:
            return self.get_radiometry(r_type, interpolated=False)
        else:
            print(f'No radiometry {r_type} found.')

    def __repr__(self):
        s = f'Class Radiometry\n'
        s += f'Raw radiometries: {list(self.radiances.keys())} \n'
        s += f'Interpolated radiometries: {list(self.interp_radiances.keys())} \n'
        s += f'Folder: {self.folder} \n'
        s += f'Date Range: {self.times_range()} \n'
        # s += f'Subset: '
        # s += f'{len(self.subset)} items' if isinstance(self.subset, list) else f'{self.subset}'
        return s

