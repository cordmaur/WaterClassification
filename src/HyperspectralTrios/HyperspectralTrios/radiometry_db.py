import pandas as pd
import plotly.express as px
import os
import configparser
from pathlib import Path
from functools import partial

# other functions defined within the package
from .common import listify, get_file_datetime, copy_dir, get_complete_dirs, check_file, all_wls
from .radiometry import Radiometry, BaseRadiometry
from .common_excel import append_df_to_excel


class RadiometryDB:
    """Class responsible for loading a Radiometry Database and manipulating it."""

    config = {
        "control_file": 'BD-GET-Summary.xlsx',
        "mean_interpolated": 'Mean_interpolated.csv',
        "trios_output": 'Trios_Output.csv',
        "summary": '_summary.csv',
        "interpolated_name": '_interpolated.csv'
    }

    workbooks = {
        "data": {'name': 'GETDB-Summary', 'index': 'Id'},
        "stations": {'name': 'Stations', 'index': 'StationId'},
        "summertime": {'name': 'SummerTime', 'index': 'Id'},
        "granulometry": {'name': 'Granulometry', 'index': 'Id'},
        "kd": {'name': 'Kd', 'index': 'Id'}
    }

    all_radiometries = ['Rrs', 'Ed', 'Ld', 'Lu']

    default_attrs = ['Area', 'Station', 'Measurement', 'Start_Date', 'SPM', 'Status', 'Description']

    def __init__(self, path, config=None):
        """Open the radiometry DB located in specified folder"""
        self.path = Path(path)

        # update the main configuration of the class with the passed parameters
        if config is not None:
            self.config.update(config)

        # load main spreadsheets. They will be stored as instance attributes.
        # The attributes names and workbooks correspondence are defined in the workbooks dictionary
        self.load_control_file()

        # Join main data with the stations
        self.data = self.data.join(self.stations.set_index('Station_Name'), on='Station_Name')

        self.rdmtries = None

    # Loading functions
    def load_control_file(self):
        """Load the control file  and return data, stations and summertime dataframes.
        The configuration parameters shall be passed as a dict-like"""
        if check_file(self.path, self.config['control_file']):
            # Load all the workbooks defined in the class dictionary and add them as attributes to the class
            for workbook in self.workbooks:
                wb = pd.read_excel(self.path / self.config['control_file'],
                                   RadiometryDB.workbooks[workbook]['name'], engine='openpyxl')
                self.__setattr__(workbook, wb.set_index(RadiometryDB.workbooks[workbook]['index']))

            # data = pd.read_excel(path / config['control_file'], config['main_workbook'], engine='openpyxl')\
            #     .set_index('Id')
            # stations = pd.read_excel(path / config['control_file'], config['stations_workbook'], engine='openpyxl')\
            #     .set_index('StationId')
            # summertime = pd.read_excel(path / config['control_file'], config['summertime_workbook'], engine='openpyxl')
        else:
            print(f"Not possible to open control file {self.config['control_file']} at {self.path}")

    @staticmethod
    def _load_mean_radiometries(df, db_anchor, r_types='Rrs', path_column='Relative_Path', attrs=None,
                                mean_fname='Mean_interpolated.csv'):
        attrs = RadiometryDB.default_attrs if attrs is None else attrs

        db_anchor = Path(db_anchor)

        # convert the r_types into a list to be passed to the mapping function
        r_types = listify(r_types)

        def map_radiometry(row, rtypes, _attrs):
            rdmtry = pd.read_csv(db_anchor / row[path_column] / mean_fname, sep=';', index_col=0)

            # check if the r_type exists in the radiometry mean file
            for r_type in r_types:
                if not (r_type in rdmtry.index):
                    print(f'{r_type} not found in {row[path_column]}')

                if rdmtry.index.to_list().count(r_type) > 1:
                    print(f'More than one [{r_type}] in {row[path_column]}')

            return [row[_attrs].append(rdmtry.loc[r_type]) if r_type in rdmtry.index else None
                    for r_type in rtypes]

        idx_name = df.index.name
        df = df.reset_index(drop=False, inplace=False)
        result_df = df.apply(partial(map_radiometry, rtypes=r_types, _attrs=[idx_name]+attrs), axis=1)

        # as the result comes in list, we have to split it first
        result_df = pd.DataFrame(result_df.tolist())

        # create a dictionary with a dataframe for each interpolated radiometry
        result_dic = {r_type: pd.DataFrame(result_df[i].tolist()).dropna(axis=1, how='all').drop(columns='DateTime')
                      for i, r_type in enumerate(r_types)}

        # adjust index
        for r_type in r_types:
            result_dic[r_type].set_index('Id', inplace=True)

        return result_dic

    @staticmethod
    def plot_ids(df, ids, hover_vars=None, color=None, bands=all_wls, title=''):
        sub_df = df.loc[ids]
        color = sub_df.index if color is None else color

        return BaseRadiometry.plot_reflectances(sub_df,
                                                bands,
                                                color=color,
                                                hover_vars=hover_vars,
                                                colormap='viridis',
                                                log_color=True,
                                                colorbar=False,
                                                show_legend=True,
                                                discrete=True,
                                               line_width=2
                                                )

    def load_radiometry(self, id=None, folder=None, area=None, station=None, measurement=None,
                        load_interpolated=True, read_backup=False):
        """Load a specific radiometry from the database and returns a HyperspectralTrios class
        The load radiometry function can work with either:
        - database Id
        - description: area, station and measurement ( in the format YYYYMMDD-hhmm)
        - folder: if we know the folder (absolute or relative) of the measurement in the database
        """
        if id is not None:
            path = self.path/self.data.loc[id]['Relative_Path']

        elif folder is not None:
            folder = Path(folder)
            path = folder if folder.is_absolute() else self.path/folder

        elif (area is not None) and (station is not None) and (measurement is not None):
            path = self.path/area/station/measurement

        else:
            print('Not enough info to load the radiometry. Check input arguments.')
            return None

        return Radiometry.from_folder(path,
                                      load_interpolated=load_interpolated,
                                      read_backup=read_backup)

    def load_radiometries(self, r_types=None, attrs=None, norm=False, funcs=None, qry=None):
        """
        Load the mean radiometries from all points in the database. It uses the summary csv's that have been
        previously created using create_summary_radiometries. The results are written as DataFrames into a dictionary.
        :param r_types: The radiometries to be loaded (ex. 'Rrs', 'Ed', 'Ld', 'Lu')
        :param attrs: Additional attributes to be loaded in the dataframe. Defaults to:
        ['Area', 'Station', 'Lat', 'Long', 'GMT', 'SummerTime', 'OBS', 'Relative_Path', 'Start_Date', 'End_Date',
        'Project', 'Description']
        :param funcs: List of functions to be applied to the resulting DataFrames
        :param norm: Include normalized bands with prefix 'n'
        :param qry: Additional query to select only desired status
        :return: A dictionary of radiometries. Each entry is a Dataframe with corresponding measurements.
        """
        attrs = ['Status', 'SPM', 'Area', 'Station', 'Lat', 'Long', 'GMT', 'SummerTime', 'OBS', 'Relative_Path',
                 'Start_Date', 'End_Date', 'Project', 'Description'] if attrs is None else listify(attrs)

        r_types = RadiometryDB.all_radiometries if r_types is None else listify(r_types)

        self.rdmtries = {r_type: pd.read_csv(self.path/f"{r_type}{self.config['summary']}", index_col='Id')
                         for r_type in r_types}

        for r_type in r_types:
            # make a join to add the attributes
            self.rdmtries[r_type] = self.rdmtries[r_type].join(self.data[attrs], lsuffix='_2')

            # create the normalized bands (with suffix)
            if norm:
                df_norm = BaseRadiometry.normalize(self.rdmtries[r_type])
                bands = df_norm.columns[df_norm.columns.str.isdigit()]
                df_norm.rename(columns={band: f'n{band}' for band in bands}, inplace=True)

                self.rdmtries[r_type] = pd.concat([self.rdmtries[r_type], df_norm[[f'n{b}' for b in bands]]], axis=1)

            # apply post-processing functions
            if funcs:
                funcs = listify(funcs)
                for func in funcs:
                    func(self.rdmtries[r_type])

            if qry:
                self.rdmtries[r_type].query(qry, inplace=True)

        print(f'Radiometries {r_types} loaded in dictionary .rdmtries')

    # Properties
    @property
    def control_file(self): return self.path/self.config['control_file']

    def status(self):
        """Output the overall status of the database (number of records, update dates, etc.) """
        print(10*'-' + ' DataBase Status ' + 10*'-')
        print(f'Base Folder: {self.path}')
        print(f'Control File: {self.control_file.name} (Modified: {get_file_datetime(self.control_file)})')

        print('\n')
        print('Summary files:')

        # Get the summary files and plot their modified dates
        files = [f for f in self.path.glob('*' + self.config['summary'])]
        for file in files:
            print(f'{file.name}: Modified on {get_file_datetime(file)}')

        # Specifies the files to be included in the status
        files = {'Reflectance': 'Rrs_interpolated.csv',
                 'Mean Reflectance': 'Mean_interpolated.csv',
                 'OSOAA Reflectance': 'Rrs_OSOAA_interpolated.csv',
                 'Metadata': 'metadata.txt',
                 'Graph': '*.png'
                 }

        print('\n')
        print('Integrity/date table:')

        # Create a dictionary to receive the results from each check
        series = {}

        # Create a series with the absolute directories to iter
        paths = self.data['Relative_Path'].map(lambda x: self.path/x)

        # For each file to be checked, call the get_file_datetime function
        for name, file in files.items():
            series[name] = paths.map(lambda x: get_file_datetime(x/ file, ft='%Y-%m-%d')).value_counts()

        # Return a pandas dataframe with the results
        return pd.DataFrame(series).sort_index(ascending=False)

    def summary(self, by='Area', plot=False, bars=None, **kwargs):
        """Create a summary DataFrame to present all data stored in the database
        :param bars: The name of the bars to include in the plot.
        This can be 'SPM', 'POC', 'DOC', 'Chl-a', 'Granul', 'Kd'
        :param plot: Instead of returning a DataFrame, return a plotly bar fig
        """
        bars = ['SPM', 'POC', 'DOC', 'Chl-a', 'Granul', 'Kd'] if bars is None else bars

        # make a joined summary DataFrame
        summary = self.data.copy()

        summary = summary.join(self.granulometry['D10']).join(self.kd[650])
        summary.rename(columns={'D10': 'Granul', 650: 'Kd'}, inplace=True)

        summary = summary.groupby(by=by).count()[bars]

        summary.sort_values(by=bars, ascending=False, inplace=True)
        if plot:
            return px.bar(summary, y=bars, barmode='group', **kwargs)
        else:
            return summary

    # Internal Functions
    def get_measurement_id(self, station_name, measurement):
        measurement = self.data[(self.data['Station_Name'] == station_name) &
                                (self.data['Measurement'] == measurement)]

        return int(measurement.index.values) if len(measurement) > 0 else None

    def get_station_id(self, station_name):
        station = self.stations[self.stations['Station_Name'] == station_name]
        return int(station.index.values) if len(station) > 0 else None

    def insert_station(self, row, overwrite=False):
        idx = self.get_station_id(row['Station_Name'])

        if idx is not None:
            if overwrite:
                print(f"*** Overwriting existing station {row['Station_Name']}(Id: {idx})")
                self.stations.loc[idx] = row
            else:
                print(f"*** Station {row['Station_Name']}(Id: {idx}) already exists. Skipping it.")
        else:
            row.name = self.stations.index.max() + 1
            print(f'Appending new station - Id={row.name}')
            self.stations = self.stations.append(row)

    def insert_measurement(self, row, overwrite=False):
        idx = self.get_measurement_id(row['Station_Name'], row['Measurement'])

        if idx is not None:
            if overwrite:
                print(f'*** Overwriting existing record. Id={idx}')
                self.data.loc[idx] = row
            else:
                print(f"*** Record Id={idx} already exists. Skipping it.")

        else:
            row.name = self.data.index.max() + 1
            print(f'Appending new data. Id={row.name}')
            self.data = self.data.append(row)

        return idx

    # Processing functions
    def loop_database(self, funcs, qry_str=None):
        """Loop the database and applies several callback functions to each row.
        The subset can be specified by a query string like: Id > 400 or SPM < 8.0 for example."""
        # if filters is not None:
        #     filters = listify(filters)
        #     dfs = [self.data[self.data['Station_Name'].str.contains(f, na='')] for f in filters]
        #
        #     df = pd.concat(dfs)
        # else:
        #     df = self.data
        df = self.data if qry_str is None else self.data.query(qry_str)

        funcs = listify(funcs)

        for idx, row in df.iterrows():
            if pd.isnull(row['Relative_Path']):
                print(f"Missing Relative Path for row: {row.name}")
                continue

            out_path = self.path/row['Relative_Path']
            if not out_path.exists():
                print(f'Folder {out_path} not found. Skipping')
                continue

            for func in funcs:
                func(row)

        print(f'Affected IDs: {df.index.to_list()}')

    def create_summary_radiometries(self, r_types=None, attrs=None):
        """Create the files xxx summary.csv for the r_types indicated. These files are necessary to load
        the mean radiometries of the database. The attributes will be stored in the final .csv
        this function.
        r_types are ['Rrs', 'Ed', 'Ld', 'Lu']
        """
        r_types = RadiometryDB.all_radiometries if r_types is None else listify(r_types)
        self.rdmtries = RadiometryDB._load_mean_radiometries(df=self.data,
                                                             db_anchor=self.path,
                                                             r_types=r_types,
                                                             path_column='Relative_Path',
                                                             attrs=attrs,
                                                             mean_fname=self.config['mean_interpolated']
                                                             )

        for r_type in r_types:
            fn = self.path / (r_type + self.config['summary'])
            print(f'Saving radiometry: {fn}')
            self.rdmtries[r_type].to_csv(fn, index=True)

        print(f'Radiometries {r_types} loaded in dictionary .rdmtries')

    def create_adjusted_reflectance(self, trios_main, qry_str=None):
        """
        Create the adjusted reflectance using Tristan's TRIOS software
        It is necessary to pass the main .py file from TRIOS with full path
        The code can be run for just a subset, using a query_string (qry_str)
        """
        def map_adjusted_reflectance(row):
            """Function to be applied to each row of the data table
            TRIOS from Tristan is executed as shell"""
            print(f"Processing: {row['Relative_Path']}")
            path = Path(self.path/row['Relative_Path'])
            os.system(f"echo '{path.as_posix()}'")
            args = "\"" + path.as_posix() + "\" 150 awr"
            args += f" --lat {row['Lat']} --lon {row['Long']}"
            args += " --odir \"" + path.as_posix() + "\""
            args += ' --format=csv --data_files="Ed_interpolated.csv Ld_interpolated.csv Lu_interpolated.csv"'
            args += " --ofile " + self.config['trios_output']
            args += " --plot --figdir \"" + path.as_posix() + "\""
            args += f" --altitude={row['Altitude']}"

            cmd = f"python \"{trios_main}\" {args}"
            os.system(f"echo {cmd}")
            os.system(f"python \"{trios_main}\" {args}")

            # After the execution of Trios, we open the trios csv and the mean_reflectances.csv and
            # append the results to the mean_reflectances.
            # open the resulting TRIOS csv file
            trios_df = pd.read_csv(path/self.config['trios_output'], header=[0, 1])

            # Grab just the reflectance and the date times
            trios_rrs = trios_df['Rrs'].drop(0)
            trios_dt = trios_df['param'].drop(0).rename(columns={'wl': 'DateTime'})
            trios_dt['DateTime'] = pd.to_datetime(trios_dt['DateTime'])

            # Save the adjusted Rrs (OSOAA) to the interpolated file
            rrs_osoaa = pd.concat([trios_dt, trios_rrs], axis=1)
            rrs_osoaa.to_csv(path / ('Rrs_OSOAA'+self.config['interpolated_name']), sep=';', index=False)

            # and calculate the mean
            trios_mean = trios_rrs.mean()
            trios_mean.name = 'Rrs_OSOAA'

            # open the Mean_Reflectance.csv, drop old Rrs_OSOAA and append the new
            df = pd.read_csv(path/self.config['mean_interpolated'], sep=';', index_col=0)
            df.drop(index='Rrs_OSOAA', errors='ignore', inplace=True)
            df = df.append(trios_mean)
            df['DateTime'] = df.loc['Rrs', 'DateTime']
            df.to_csv(path/self.config['mean_interpolated'], sep=';', index=True)

        # loop through the database and apply the mapping function
        self.loop_database(funcs=map_adjusted_reflectance, qry_str=qry_str)

    def reprocess_database(self, qry_str=None, prefixes=None, reflectance=False, mean_values=False, graphs=False):
        prefixes = prefixes if prefixes is not None else ['Rrs', 'Ed', 'Ld', 'Lu']

        def reprocess_db_callback(row):
            out_path = self.path / row['Relative_Path']

            rdmtry = Radiometry.from_folder(out_path, prefixes=prefixes, load_interpolated=True)

            if reflectance:
                rdmtry.create_reflectance()
                rdmtry.save_radiometries(save_backup=True, subset=False)

            if mean_values:
                rdmtry.save_mean_radiometry(subset=False)

            if graphs:
                rdmtry.save_radiometries_graph(rdmtry.folder, subset=False)

        self.loop_database(funcs=reprocess_db_callback, qry_str=qry_str)

    def update_metadata(self, attrs, qry_str=None, section='Metadata'):
        attrs = listify(attrs)

        def update_metadata_callback(row):
            out_path = self.path / row['Relative_Path']
            fn = out_path / 'metadata.txt'
            config = configparser.ConfigParser()
            config.read(fn)

            config[section].update({column: str(row[column]) for column in attrs})

            with open(fn, 'w') as configfile:
                config.write(configfile)

        self.loop_database(funcs=update_metadata_callback, qry_str=qry_str)

    @staticmethod
    def check_structure(base_dir, prefixes, sep='_', base_name='spectrum_LO', dir_level=0):
        """
        Create a data frame structure based on the sub directories and the TXT/MLB files
        The prefix are the Ed Ld and Lu to search for in the names of the files
        Base name is the any constant the operator has used to save the files
        Separator is the connecting character between the prefix and the base_name (ex. Ld_spectrum LO)
        The file used to grab the dates will be the one with the first prefix
        """

        # first, we will create a filename list
        names_lst = [prefix + sep + base_name for prefix in prefixes]

        complete_dirs = get_complete_dirs(base_dir=base_dir, names_lst=names_lst)

        df = pd.DataFrame(data=complete_dirs, columns=['Dir'])

        df['Group'] = None
        df['Area'] = df['Dir'].map(lambda x: x.parts[-3 + dir_level])
        df['Station'] = df['Dir'].map(lambda x: x.parts[-2 + dir_level])

        # Fill start and end datetime for the measurement
        df['Start_Date'] = df['Dir'].map(partial(BaseRadiometry.get_radiometry_date, base_name=names_lst[0], dt_type='start'))
        df['End_Date'] = df['Dir'].map(partial(BaseRadiometry.get_radiometry_date, base_name=names_lst[0], dt_type='end'))
        df['Duration'] = df['End_Date'] - df['Start_Date']

        # Get the number of measurements for each sensor (just for consistency checks)
        df['Ed_measurements'] = df['Dir'].map(partial(BaseRadiometry.get_number_measurements, rdmtry_type='Ed',
                                                      sep=sep, base_name=base_name))
        df['Ld_measurements'] = df['Dir'].map(partial(BaseRadiometry.get_number_measurements, rdmtry_type='Ld',
                                                      sep=sep, base_name=base_name))
        df['Lu_measurements'] = df['Dir'].map(partial(BaseRadiometry.get_number_measurements, rdmtry_type='Lu',
                                                      sep=sep, base_name=base_name))

        df['Description'] = df['Dir'].map(lambda x: x.stem)

        return df.sort_values(by='Start_Date')

    def append_new_data(self, from_folder, copy_all_files=False, reprocess=False, overwrite=False,
                        base_name='spectrum_LO', sep='_', dir_level=0):
        """Append to the database new data that follows the structure root-Area-Station-Measurement
        The measurement folder will be renamed to match DB standard (yyyymmdd-hhmm)"""

        # Step 1 - Check the new data
        df = RadiometryDB.check_structure(from_folder, prefixes=['Ld', 'Ed', 'Lu'], base_name=base_name, sep=sep,
                                          dir_level=dir_level)

        # Create the destination folder for each new measurement
        df['NewDir'] = self.path.as_posix() + '/' + df['Area'] + '/' + df['Station'] + '/' + df['Start_Date'].astype(
            'str').str.replace('-', '').str.replace(' ', '-').str[:14].str.replace(':', '')

        # Step 2 - Iterate through the rows of the dataframe to copy the old directory into the new directory
        pattern = '*' if copy_all_files else f'*{base_name}*'

        for idx, row in df.iterrows():
            copy_dir(row['Dir'], row['NewDir'], pattern=pattern)

        # Step 3 - If reprocess==True, reprocess radiometry to create Interpolated Reflectance, .PNG, etc
        if reprocess:
            for idx, row in df.iterrows():
                rdmtry = Radiometry.from_folder(row['NewDir'], prefixes=['Ed', 'Ld', 'Lu'])
                rdmtry.create_reflectance(min_wl=320, max_wl=950)
                rdmtry.save_radiometries(save_backup=False, subset=False)
                rdmtry.save_radiometries_graph(rdmtry.folder, subset=False)

                rdmtry.update_metadata()

        # Once everything is reprocessed, put the metadata into the resulting dataframe
        def map_get_attributes(folder):
            rdmtry2 = Radiometry.from_folder(folder)

            return pd.Series([rdmtry2.metadata['Metadata'].get('Metadata', 'ed_device'),
                              rdmtry2.metadata['Metadata'].get('Metadata', 'ld_device'),
                              rdmtry2.metadata['Metadata'].get('Metadata', 'lu_device'),
                              int(rdmtry2.metadata['Metadata'].get('Metadata', 'rrs_measurements'))])

        df['Ed_device'] = df['Ld_device'] = df['Lu_device'] = df['Rrs_measurements'] = None

        df[['Ed_device', 'Ld_device', 'Lu_device', 'Rrs_measurements']] = \
            pd.DataFrame(df['NewDir'].map(map_get_attributes).to_list())

        # step 4 - Create the columns to be updated in the control file
        df['Relative_Path'] = df['NewDir'].map(lambda x: Path(x).relative_to(self.path))
        df['Station_Name'] = df['Area'] + '-' + df['Station']
        df['Measurement'] = df['NewDir'].map(lambda x: Path(x).relative_to(self.path).name)

        # step 5 - Update the stations workbook
        for area, station in df.groupby(by=['Area', 'Station']).count().index:
            # self.create_station(area, station)
            self.insert_station(pd.Series({'Area': area,
                                           'Station': station,
                                           'Station_Name': '-'.join([area, station])}),
                                overwrite=overwrite)

        # step 6 - update the main workbook
        columns = ['Relative_Path', 'Station_Name', 'Measurement', 'Start_Date', 'End_Date', 'Ed_device',
                   'Ld_device', 'Lu_device', 'Rrs_measurements', 'Ed_measurements', 'Ld_measurements',
                   'Lu_measurements', 'Description']

        for _, row in df[columns].iterrows():
            self.insert_measurement(row, overwrite=overwrite)

        return 'OK'

    # Saving Functions
    def commit_workbook(self, workbook='main_workbook'):
        if workbook == 'stations_workbook':
            df = self.stations
        elif workbook == 'summertime_workbook':
            df = self.summertime
        else:
            df = self.data[['Relative_Path', 'Station_Name', 'Measurement', 'SPM', 'Chl-a', 'POC', 'DOC', 'Status',
                            'Start_Date','End_Date', 'gmt_datetime', 'Project', 'Ed_device', 'Ld_device',
                            'Lu_device', 'Rrs_measurements', 'Ed_measurements', 'Ld_measurements',
                            'Lu_measurements', 'Description']]

        append_df_to_excel(self.control_file, df, sheet_name=self.config[workbook], startrow=0,
                           truncate_sheet=True)

    def commit_all(self, workbooks=None):
        """Commit changes to the control file. Workbooks argument specifies which workbook to commit
        Options are: 'main_workbook', 'stations_workbook', 'summertime_workbook'"""

        workbooks = ['main_workbook', 'stations_workbook', 'summertime_workbook'] if workbooks is None \
            else listify(workbooks)

        for workbook in workbooks:
            self.commit_workbook(workbook=workbook)

    # Adjust GMT Times
    @staticmethod
    def in_summer_period(dt, summer_times):
        def in_period(row):
            return (dt > row['StartTime']) and (dt < row['EndTime'])

        return summer_times.apply(in_period, axis=1).any()

    @staticmethod
    def map_gmt_time(row, date_column, gmt_column, use_st_column, summer_times):
        dt = row[date_column]
        is_summer = RadiometryDB.in_summer_period(dt, summer_times=summer_times)

        apply_summertime = (row[use_st_column] == 1) and is_summer

        delta = -row[gmt_column] if not apply_summertime else -(row[gmt_column] + 1)

        return dt + pd.to_timedelta(delta, unit='h')

    def fill_gmt_datetime(self):
        map_func = partial(RadiometryDB.map_gmt_time, date_column='Start_Date', gmt_column='GMT',
                           use_st_column='SummerTime', summer_times=self.summertime)

        self.data['gmt_datetime'] = self.data.apply(map_func, axis=1)

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __repr__(self):
        s = f'HyperspectralTrios Database with {len(self)} records and ' \
            f'{len(self.stations) if self.stations is not None else 0} stations'
        return s
