import pyodbc
from pathlib import Path
import pandas as pd
from io import StringIO
from datetime import timedelta
import numpy as np
from .common import apply_subplot, to_excel_date
from .radiometry import BaseRadiometry
from plotly import subplots
import math


class BaseMDB:
    """
    The BaseMDB class wraps all the basic functions to manipulate the MDB using an ODBC connection.
    All functions are declared as Static Methods and can be called as regular functions.
    """

    # ###########  MDB Functions   ##########
    @staticmethod
    def open_connection(mdb):
        """
        Open a connection to a MDB file.
        :param mdb: path (string or Path object) to the .mdb file
        :return: ODBC connection
        """
        return pyodbc.connect(r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + str(mdb))

    @staticmethod
    def get_table_names(conn):
        """
        Retrieve the tables existent in the MDB
        :param conn: Connection to the MDB file
        :return: Table names
        """
        crsr = conn.cursor()
        return [table_info.table_name for table_info in crsr.tables(tableType='TABLE')]

    @staticmethod
    def exec_query(conn, qry):
        """
        Execute a query on the given (opened) connection
        :param conn: opened ODBC connection
        :param qry: query to be executed
        :return: a list with the results. Each item is a tuple representing the rows
        """
        cur = conn.execute(qry)
        return [row[0] if len(row) == 1 else row for row in cur.fetchall()]

    @staticmethod
    def get_column_values(conn, tbl, column):
        """
        Retrieve the unique values of a column as a list.
        :param conn: ODBC opened connection
        :param tbl: table
        :param column: column name
        :return: list of unique values
        """
        qry = f'SELECT DISTINCT {column} FROM {tbl}'
        return BaseMDB.exec_query(conn, qry)

    # GET MEASUREMENT FUNCTIONS
    @staticmethod
    def create_query_str(table, columns, conditions):
        """
        Create a query string given a table, the columns to be selected and conditions to be satisfied.
        :param table: table name
        :param columns: columns to be selected in the output
        :param conditions: conditions that should be met
        :return:
        """
        # create the base query
        qry = f"SELECT {columns} FROM {table} "

        # add the conditions
        if conditions is not None:
            for i, (column, value) in enumerate(conditions.items()):
                logic = 'WHERE' if i == 0 else 'AND'
                qry += f"{logic} {column} LIKE '{value}' "

        return qry

    @staticmethod
    def get_rows(mdb, select_columns, conditions=None, index_col=None, output_format='list'):
        """
        Given a dictionary of conditions {column: value, column:value, column: value}
        Return selected columns of the rows that meet these specifications.
        Output_format can be 'list' or 'dataframe'
        """

        # if mdb is passed as a string (or Path), open connection and close at the end
        conn = BaseMDB.open_connection(mdb) if isinstance(mdb, (str, Path)) else mdb

        qry = BaseMDB.create_query_str('tblData', select_columns, conditions)

        if output_format == 'list':
            result = BaseMDB.exec_query(conn, qry)
        else:
            result = pd.read_sql(sql=qry, con=conn, index_col=index_col)

        if isinstance(mdb, (str, Path)):
            conn.close()

        return result

    @staticmethod
    def get_date_times(mdb, r_types_descr, logic='and'):
        """
        Retrieve the datetime index list that meets the specified r_types.
        :param mdb: path or opened ODBC connection
        :param r_types_descr: Dictionary of radiometry types to be searched in the MDB
        ex. {'reflectance': {'IDDevice': 'SAM_83AE', 'Comment': 'reflectance'}}
        :param logic: If logic='and', only the datetimes that intersect all the r_types are considered.
        if logic='or' all datetimes tha contains any of the r_types are considered.
        :return: datetime index list that meet any of the given specification
        """

        # if mdb is passed as a string (or Path), open connection and close at the end
        conn = BaseMDB.open_connection(mdb) if isinstance(mdb, (str, Path)) else mdb

        dates_idx = None
        for r_type, conditions in r_types_descr.items():
            rows = BaseMDB.get_rows(conn, 'DateTime', conditions=conditions)

            idx = pd.DatetimeIndex(rows)

            # check for duplicates
            if idx.has_duplicates:
                print(f'Duplicated datetime found for {r_type}. Keeping first occurrences.')

            if logic == 'and':
                dates_idx = idx if dates_idx is None else dates_idx.intersection(idx)
            else:
                dates_idx = idx if dates_idx is None else dates_idx.union(idx)

        if isinstance(mdb, (str, Path)):
            conn.close()

        return dates_idx

    @staticmethod
    def create_measurements_df(measures, additional_header=False):
        measures_df = None
        wls = None

        for idx, row in measures.iterrows():
            data = row['Data']
            iodata = StringIO(data)

            attrs = BaseMDB.parse_attributes(row['Attributes'])

            df = pd.read_csv(iodata, sep=' ', header=None)
            df.drop(index=0, columns=[0, 3, 4], inplace=True)

            df['DateTime'] = row.name
            pivotted = df.pivot(index='DateTime', columns=1, values=2)

            # get the columns that represent the wavelenghts
            wls = pivotted.columns

            # Create new columns
            pivotted['PositionLatitude'] = 0.0
            pivotted['PositionLongitude'] = 0.0
            pivotted['IntegrationTime'] = attrs['IntegrationTime']
            pivotted['Comment'] = row['Comment']
            pivotted['IDData'] = row['IDData']

            # reorder columns
            pivotted = pivotted[
                ['PositionLatitude', 'PositionLongitude', 'IntegrationTime'] + list(wls.values) + ['Comment', 'IDData']]

            measures_df = pivotted if measures_df is None else pd.concat([measures_df, pivotted], axis=0)

        if additional_header:
            c_header = ['DateTime', 'PositionLatitude', 'PositionLongitude', 'IntegrationTime'] + \
                       [f'c{str(i).zfill(3)}' for i in range(1, len(wls) + 1)] + ['Comment', 'IDData']
            c_header = '\t'.join(c_header)
            return measures_df, c_header
        else:
            return measures_df

    # ###########  Attribute Parsing Functions   ##########
    @staticmethod
    def parse_attributes(attrs):
        attrs_dic = {}
        for attr in attrs.splitlines():
            key, value = attr.split('=')
            attrs_dic.update({key.strip(): value.strip()})
        return attrs_dic

    @staticmethod
    def parse_row(row):
        """Expand a measurement row with the attributes in the attributes field and return the row
        as a series with all attributes"""

        # Parse the attributes of this measurement and append to the series
        attrs = BaseMDB.parse_attributes(row['Attributes'])
        series = row.append(pd.Series(attrs))

        return series

    @staticmethod
    def create_header_df(row):
        """Create a header as a DataFrame with the information from the passed row"""

        # First, expand the row, by parsing all the attributes
        series = BaseMDB.parse_row(row)

        # Get just the attributes for the header
        header_attrs = ['IDDevice', 'IDDataType', 'IDDataTypeSub1', 'IDDataTypeSub2', 'IDDataTypeSub3', 'RecordType',
                        'IDMethodType', 'MethodName']
        header_attrs += ['IntegrationTime', 'Unit1', 'Unit2', 'Unit3', 'Unit4', 'IDDataBack', 'IDDataCal',
                         'IDBasisSpec',
                         'PathLength', 'CalFactor']

        header = series.loc[header_attrs]

        # create a DataFrame with the header data
        data = {'Attribute': header.index, 'sep': '=', 'Value': header.values}
        df = pd.DataFrame(data=data).set_index('Attribute')

        return df

    @staticmethod
    def create_radiometry_csv(mdb, description, times=None):
        """Create a csv file in the same format as the .MLB or the .TXT of the MSDA_XE software"""

        # get the measurements in the MDB matching r_type_descr
        measures = BaseMDB.get_rows(mdb, '*', description, output_format='dataframe', index_col='DateTime')

        # if there are specified times, use them to slice the dataframe
        if times is not None:
            measures = measures[measures.index.isin(times)]

            # now check if the passed dates were found in the MDB
            if len(measures) != len(times):
                print(f'Warning: {len(times)} times informed, but {len(measures)} were found')

        # Create the header text, pass the first row for it
        header_txt = BaseMDB.create_header_df(measures.iloc[0]).to_csv(sep='\t', line_terminator='\n', header=False)

        # Create the measurements text
        measures_df, c_header = BaseMDB.create_measurements_df(measures, additional_header=True)

        # Before writing to txt, convert the dates to Excel format
        measures_df.index = measures_df.index.map(to_excel_date)

        measures_txt = measures_df.to_csv(sep='\t', line_terminator='\n')

        for title in ['DateTime', 'PositionLatitude', 'PositionLongitude', 'IntegrationTime', 'Comment', 'IDData']:
            measures_txt = measures_txt.replace(title, 'NaN')

        return header_txt + '\n' + c_header + '\n' + measures_txt

    @staticmethod
    def export_mdb_csv2(mdb_path, r_types_descr, out_dir=None, create_measurement_dir=True, times=None):
        # if an output directory is specified, append it to the current path
        if out_dir is not None:
            out_path = mdb_path.with_name(out_dir)
        else:
            out_path = mdb_path.parent

        # if create_measurement_dir is true, create the datetime dir for the specific measurement
        if create_measurement_dir:
            if times is not None:
                dt = times[0]
            else:
                dt = BaseMDB.get_date_times(mdb_path, r_types_descr)

            out_name = str(dt).replace('-', '').replace(' ', '-').replace(':', '')[:13]
            out_path = out_path / out_name

        print(f'Saving output file to {out_path}')
        out_path.mkdir(parents=True, exist_ok=True)

        # Loop through the r_types
        for r_type, description in r_types_descr.items():
            txt = BaseMDB.create_radiometry_csv(mdb_path,
                                                description=description,
                                                times=times)

            with open(out_path / f'{r_type}_spectrum_LO.txt', "w") as text_file:
                text_file.write(txt)

        return out_path


class TriosMDB:
    """
    The RadiometryMDB class wraps all the functionality needed to access a MSDA MDB file and extract
    the measurements from it.
    """
    data_table = 'tblData'
    default_columns = ['IDDevice', 'IDDataType', 'MethodName', 'IDMethodType', 'Comment', 'CommentSub1', 'CommentSub2']

    def __init__(self, mdb_path):
        """
        Create a TriosMDB object
        :param mdb_path: string or Path to the .mdb file
        """

        # Check the file
        try:
            with BaseMDB.open_connection(mdb_path) as conn:
                self.df = pd.read_sql(f'select * from {self.data_table}', conn, index_col='DateTime')
                self.path = Path(mdb_path)

        except Exception as e:
            print(f'ERROR: It was not possible to open file {mdb_path}')
            print(e)
            self.path = None
            self.df = None

        finally:
            self.times = None
            self.time_window = None
            self.measurements = None
            self.rdmtry_descr = None

    # ##########  MDB ACCESS METHODS  #############
    def exec_query(self, qry, output_format='list'):
        """
        Execute a query on the given (opened) connection
        :param qry: query to be executed
        :param output_format: 'list' or 'pandas'
        :return: a list or a DataFrame with the results. When return as list, each item in the list is a tuple
        representing the rows
        """
        with BaseMDB.open_connection(self.path) as conn:
            if output_format == 'list':
                cur = conn.execute(qry)
                return [row[0] if len(row) == 1 else row for row in cur.fetchall()]
            elif output_format == 'pandas':
                return pd.read_sql(qry, conn)
            else:
                print(f'Output format {output_format} not implemented')

    def summary(self, columns=None):
        """
        Returns a summary of the database with the most important columns and its values. Additionally,
        it can receive the columns to create the summary
        :param columns: If None, it will use the default_columns as output
        :return: Dictionary with the columns and unique values
        """
        columns = self.default_columns if columns is None else columns

        results = dict(FileName=self.path, Records=len(self.df))
        results.update({column: str(list(self.df[column].unique())) for column in columns})
        return results

    def select_radiometries(self, rdmtry_descr, logic='and'):
        """
        Select the radiometries in the MDB that meet the description in the given dictionary
        :param rdmtry_descr: a dictionary of {r_type: {column: value, column:value},
                                              r_type: {column: value, column:value}}
        :param logic: if 'and', select just the DateTimes with occurrences in all radiometries simultaneously
        :return: A datetime list of measurements that meet the criteria
        """
        self.rdmtry_descr = rdmtry_descr

        # get the times of the measurements
        self.times = BaseMDB.get_date_times(self.path, rdmtry_descr, logic)

        if len(self.times) == 0:
            print("No measurements found that meets all the criteria.\nCorrect the description or use 'or'")
            return

        self.measurements = {}
        for rdmtry in rdmtry_descr:
            measures = BaseMDB.get_rows(self.path, '*', rdmtry_descr[rdmtry], output_format='dataframe', index_col='DateTime')
            df = BaseMDB.create_measurements_df(measures, additional_header=False)
            df = df.loc[self.times].replace('-NAN', np.nan)

            numeric_columns = [column for column in df.columns if isinstance(column, (int, float))]

            # convert the values to float
            df[numeric_columns] = df[numeric_columns].astype('float')

            self.measurements[rdmtry] = df

    def filter_times(self, start, end=None, interval=timedelta(minutes=5)):
        """
        Creates a time window that will be used for displaying/exporting the measurements
        :param start: time window start datetime in 'yyyy-mm-dd hh:mm:ss' (str) format
        :param end: time window end datetime in 'yyyy-mm-dd hh:mm:ss' (str) format.
        If None, a interval will be applied to the start time.
        :param interval: A timedelta parameter that's used when end is not passed.
        :return: The resulting datetimes found in the database.
        """

        if not self.check_mdb():
            return

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

        self.time_window = self.times[(self.times >= start) & (self.times <= end)]
        return self.time_window

    def get_radiometry(self, name):
        """
        Return a loaded radiometry, given its name.
        :param name: Name of the radiometry
        :return: DataFrame of the radiometry, filtered by the time_window.
        """

        if not self.check_mdb():
            return

        if name not in self.measurements.keys():
            print(f'Radiometry {name} is not loaded.')
            return

        else:
            time_window = self.times if self.time_window is None else self.time_window
            return self.measurements[name].loc[time_window]

    # ##########  HELPER METHODS  #############
    def check_mdb(self):
        if self.path is None:
            print('TriosMDB object not connected to a MDB. Please call TriosMDB(path) again.')
            return False

        elif self.is_empty:
            print('No measurements to plot. Use select_radiometries(r_types) first.')
            return False

        else:
            return True

    # ##########  PROPERTIES  #############
    @property
    def available_columns(self): return self.df.columns.to_list()

    @property
    def is_empty(self): return self.measurements is None

    # ##########  PLOTTING METHODS  #############
    def plot_radiometry(self, name, min_wl=None, max_wl=None):
        """
        Plot the spectra of a specific radiometry. The color scale correspond to the time of the measurement.
        :param name: Name of the radiometry already loaded in the object.
        :param min_wl: Minimum wave lenght in nanometers
        :param max_wl: Maximum wave lenght in nanometers
        :return: Plotly figure
        """

        if not self.check_mdb():
            return

        # get the numeric columns, that represent the wave lengths.
        numeric_columns = sorted([column for column in self[name] if isinstance(column, (int, float))])

        # define the wave length range
        min_wl = min(numeric_columns) if min_wl is None else min_wl
        max_wl = max(numeric_columns) if max_wl is None else max_wl
        wls = np.array(numeric_columns)
        wls = wls[(wls > min_wl) & (wls < max_wl)]

        fig = BaseRadiometry.plot_reflectances(self[name], wls, color=self[name].index, colorbar=False)

        fig.update_layout(
            showlegend=True,
            title=f"Radiometry: {name}",
            xaxis_title="Wavelength (nm)",
            yaxis_title=f"{name} value",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="black"))

        return fig

    def plot_radiometries(self, cols=2, base_height=400, **kwargs):
        """
        Plot all the radiometries that are loaded.
        :param cols: Number of columns
        :param base_height: Height for each figure row
        :return: Multi-axis figure
        """

        if not self.check_mdb():
            return

        # get the number of rows
        n = len(self.measurements)
        rows = math.ceil(n/cols)

        # get the titles of the graphs
        titles = list(self.measurements.keys())

        # create the main figure
        fig = subplots.make_subplots(rows=rows, cols=cols,
                                     subplot_titles=titles)

        for idx, name in enumerate(self.measurements):
            position = ((idx // cols) + 1, (idx % cols) + 1)
            subplot = self.plot_radiometry(name, **kwargs)

            apply_subplot(fig, subplot, position)
        fig.update_layout(height=base_height * rows)
        return fig

    # ##########  EXPORTING METHODS  #############
    def export_txt(self, names=None, out_dir=None, create_measurement_dir=True):
        """
        Export the selected radiometries to a txt file (Trios/MSDA format
        :param names: Name of the radiometries to be exported. None for all.
        :param out_dir: Target directory. If not specified, the same directory of the MDB is used.
        :param create_measurement_dir: If True, creates a subdirectory in the YYYYMMDD-mmss format.
        :return: None
        """

        if not self.check_mdb():
            return

        # define the names of the radiometries to be exported
        names = list(self.measurements.keys()) if names is None else names
        names_descr = {name: self.rdmtry_descr[name] for name in names}

        # define the time_window
        time_window = self.times if self.time_window is None else self.time_window

        path = BaseMDB.export_mdb_csv2(mdb_path=self.path,
                                       r_types_descr=names_descr,
                                       out_dir=out_dir,
                                       create_measurement_dir=create_measurement_dir,
                                       times=time_window)

        return path

    # ##########  SPECIAL METHODS  #############
    def __getitem__(self, item):
        return self.get_radiometry(item)

    def __repr__(self):
        s = 'TriosMDB\n'
        s += f'MDB file: {str(self.path)} \n'
        s += f'Records: {len(self.df)}'
        return s
