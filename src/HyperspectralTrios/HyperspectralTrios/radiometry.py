from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
import math

import plotly.express as px

from HyperspectralTrios.common import get_color, hex_to_rgb
from pandas.core.dtypes.common import is_numeric_dtype

from .common import from_excel_date, apply_subplot, create_interpolated_columns, \
    listify, to_excel_date, plot_mean_std_traces, get_file_by_suffix, check_file

import configparser
from plotly import subplots, graph_objects as go
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
    def get_radiometry_date(d, dt_type='start', source='txt', base_name='Ed spectrum LO', **kwargs):
        if source == 'txt':
            f = get_file_by_suffix(d, base_name, suffixes=['.txt', '.mlb'])
            if f is not None:
                rdmtry, _ = rdmtry, meta = BaseRadiometry.open_trios_radiometry(f)
            else:
                return None

            # get the first cell (first date) or the last if end date
            return rdmtry.iloc[0, 0] if dt_type == 'start' else rdmtry.iloc[-1, 0]
        else:
            print(f"Function get_date only supports .txt source, but {source} found")

    @staticmethod
    def get_number_measurements(d, rdmtry_type='Ed', base_name='spectrum LO', sep=' '):
        f = get_file_by_suffix(d, rdmtry_type + sep + base_name)
        rdmtry, _ = BaseRadiometry.open_trios_radiometry(f)

        return len(rdmtry)

    @staticmethod
    def get_file_by_suffix(path, stem, suffixes=None):
        """Get the file that matches the suffix in the order of preference of suffixes"""
        if suffixes is None:
            suffixes = ['.txt', '.mlb']

        for suffix in suffixes:
            f = check_file(path, stem + suffix)
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
                # print(f'{prefix}:Raw not found in {folder}')
                pass

        return radiances, metadatas

    @staticmethod
    def calc_area(df, bands=None, col_name="area", norm_band=None):
        """Calc the integral of the curve and adds it to a new column.
        norm_band: the normalization band reflectance will be used to subtract the curve
        """
        bands = df.columns[df.columns.str.isdigit()] if bands is None else bands

        values = df.fillna(0)[bands].to_numpy()

        if norm_band is not None:
            norm_vector = df.fillna(0)[norm_band].to_numpy()
            values = values - norm_vector[..., None]

        df[col_name] = np.trapz(values)
        return df

    @staticmethod
    def normalize(df, bands=None, inplace=False, add_columns=False):
        """Normalize the reflectance spectra by dividing all reflectance values by the area under the curve.
        All normalized spectra will have area=1.
        add_columns will add the normalized columns to the data frame"""

        bands = df.columns[df.columns.str.isdigit()] if bands is None else bands

        if 'area' not in df.columns:
            BaseRadiometry.calc_area(df, bands)

        df_norm = df if inplace else df.copy()

        # if add_columns, we will append new columns to the dataframe (inplace or new dataframe)
        new_bands = [f'n{b}' for b in bands] if add_columns else bands

        df_norm[new_bands] = df[bands]/df.area.to_numpy()[..., None]

        return df_norm

    @staticmethod
    def plot_reflectances(df, bands, color=None, hover_vars=None, colormap='viridis', log_color=True,
                          colorbar=False, show_legend=False, discrete=True, line_width=1):
        """
        Plot radiometry curves given a dataframe and the bands (columns).
        :param df: Dataframe with the values
        :param bands: Columns that indicate the wavelenght (X axis)
        :param color: column name or a list of values.
        :param hover_vars:
        :param colormap:
        :param log_color:
        :param colorbar:
        :param show_legend:
        :param discrete: If true, force the use of discreet (not continuous).
        :param line_width:
        :return: A Plotly figure
        """

        # if None has been passed, use gray as color
        if color is None:
            cs = pd.Series('gray', index=df.index)

        # otherwise, create a color series based on a column or a list-like
        else:
            cs = df[color] if isinstance(color, str) else pd.Series(color, index=df.index)

            # if the color is numeric, we will use a continuous colorscale
            if is_numeric_dtype(cs) and not discrete:
                cs = (cs - cs.min()) / (cs - cs.min()).max()
                cs = cs.map(partial(get_color, colorscale_name='Viridis'))

            # otherwise, pass None
            else:
                cs = pd.Series([None] * len(df), index=df.index)

        scatters = []
        for idx in df.index:
            row = df.loc[idx]
            reflectances = row[bands]
            x = reflectances.index
            y = reflectances.values

            hover_text = f'Idx: {idx}<br>'
            for var in listify(hover_vars):
                hover_text += f'{var}: {row[var]}<br>'

            scatters.append(go.Scatter(x=x.astype('float'), y=y,
                                       text=hover_text,
                                       name=str(idx),
                                       line=dict(width=line_width, color=cs[idx]),
                                       showlegend=show_legend
                                       ))

        fig = go.Figure(data=scatters)

        # create the colorbar
        if colorbar and color is not None:
            colorbar_trace = go.Scatter(x=[None],
                                        y=[None],
                                        mode='markers',
                                        marker=dict(
                                            colorscale=colormap,
                                            showscale=True,
                                            cmin=0,
                                            cmax=1,
                                            colorbar=dict(xanchor="left", title='', thickness=30,
                                                          tickvals=[0, (0 + 1) / 2, 1],
                                                          ticktext=[0, (0 + 1) / 2, 1],
                                                          len=1, y=0.5
                                                          ),
                                        ),
                                        hoverinfo='none'
                                        )

            fig.add_trace(colorbar_trace)

        fig.update_layout(
            showlegend=True,
            title="Full spectra",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Radiometry",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="RebeccaPurple"))

        return fig

    @staticmethod
    def plot_mean_reflectances(df, group_column, wls, std_delta=1., opacity=0.2, shaded=True, showlegend=True):

        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24

        groups = df[group_column].unique()
        groups.sort()

        mean = df.groupby(by=group_column)[wls].mean()

        if shaded:
            std = df.groupby(by=group_column)[wls].std()
            upper = mean + std*std_delta
            lower = mean - std*std_delta
        else:
            upper = lower = None

        fig = go.Figure()

        for color_idx, group in enumerate(groups):
            y = mean.loc[group]
            fig.add_trace(go.Scatter(x=wls, y=y, name=f'Cluster {group}', line_color=colors[color_idx],
                                     showlegend=showlegend))

            transparent_color = f"rgba{(*hex_to_rgb(colors[color_idx]), opacity)}"
            if shaded:
                y_up = upper.loc[group]
                y_low = lower.loc[group]
                fig.add_trace(go.Scatter(x=wls, y=y_up, showlegend=False, mode=None,
                                         fillcolor=transparent_color,
                                         line=dict(width=0.1, color=transparent_color)
                                         ))
                fig.add_trace(go.Scatter(x=wls, y=y_low, fill='tonexty', showlegend=False, mode=None,
                                         fillcolor=transparent_color,
                                         line=dict(width=0.1, color=transparent_color)))

        fig.update_xaxes(title='Wavelength (nm)')
        fig.update_yaxes(title='Reflectance (Rrs)')
        return fig

    @staticmethod
    def calc_df_grouped_stats(df, groupby, variables, nameslist, funcslist):
        "Calculate the statistics of a dataframe, grouped by a field, given a list of variables and aggregate \nfunctions"
        stats = pd.DataFrame()

        # convert variables to a list
        variables = [variables] if isinstance(variables, str) else variables

        # loop through the desired statistics
        for name, func in zip(nameslist, funcslist):

            # create the renaming dictionary
            renaming = {var:f'{var}_{name}' for var in variables}

            stats = pd.concat([stats, func(df.groupby(by=groupby)[variables]).rename(columns=renaming)], axis=1)

        return stats


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

        # check if the folder exists
        if not folder.exists() or not folder.is_dir():
            print(f'Folder {str(folder)} does not exists')
            return

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

            rd = create_interpolated_columns(self.get_radiometry(r_type, interpolated=False),
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
                result = radiances[r_type].loc[self.subset]
                result.index.name = radiances[r_type].index.name
                return result
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

        color = df.index.astype('int32') if not mean else None

        fig = BaseRadiometry.plot_reflectances(df, numeric_columns, color=color, discrete=False, colorbar=False)

        # if mean, get the figure with mean and standard deviation
        if mean:
            mean_traces = plot_mean_std_traces(df, numeric_columns, shaded=True, std_delta=std_delta)
            for trace in mean_traces:
                fig.add_trace(trace)

        if r_type in self.labels:
            fig.update_layout(title=self.labels[r_type]['title'],
                              yaxis_title=self.labels[r_type]['y_axis'])

        return fig

    def plot_radiometries(self, r_types=None, cols=2, base_height=400, use_subset=True, interpolated=True,
                          mean=False, **kwargs):
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
            print(f"No {'interpolated' if interpolated else 'raw'} radiances were found to plot.")
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
            subplot = self.plot_radiometry(name, interpolated=interpolated, use_subset=use_subset, mean=mean, **kwargs)
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

    def apply_value_filter(self, r_type, min_value=-np.inf, max_value=np.inf, accumulate=False, interpolated=False):

        df = self.get_radiometry(r_type, interpolated=interpolated, use_subset=accumulate)

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

    def create_measurement_name(self):
        return self.times_range()[0].replace('-', '').replace(' ', '-').replace(':', '')[:13]

    def adjust_time(self, t_delta):
        """Adjust the datetime of the radiances by a timedelta. It is necessary to recreate reflectance afterwards"""
        for r_type in self.radiances:
            self.radiances[r_type].set_index(self.radiances[r_type].index + t_delta, inplace=True)

        for r_type in self.interp_radiances:
            self.interp_radiances[r_type].set_index(self.interp_radiances[r_type].index + t_delta, inplace=True)

        if self._subset is not None:
            self.subset = self.subset + t_delta

        new_folder = self.folder.with_name(self.create_measurement_name())
        self.folder.rename(new_folder)
        self.folder = new_folder

    def save_radiometry(self, r_type, use_subset=True, save_interpolated=True, save_backup=True, sep=';'):

        if r_type in self.radiances:
            self._save_radiance(r_type, use_subset=use_subset, save_backup=save_backup)

        if save_interpolated:
            if r_type not in self.interp_radiances:
                self.interpolate_radiometries()
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


