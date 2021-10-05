import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from PIL import ImageColor
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import math

# #######################  BASIC DEFINITIONS  #######################
s2bands = ['443', '490', '560', '665', '705', '740', '783', '842', '865', '940']
s2bands_norm = [f'n{b}' for b in s2bands]

# S3 bands
s3bands = ['400', '412', '442', '490', '510', '560', '620', '665', '674', '681', '709', '754', '761', '764',
           '767', '779', '865', '885', '900', '940']
s3bands_norm = [f'n{b}' for b in s3bands]


def wavelength_range(ini_wl, last_wl, step=1, prefix=''):
    """Creates a range of wavelengths from initial to last, in a defined nanometers step."""
    return [f'{prefix}{wl}' for wl in range(ini_wl, last_wl + 1, step)]


all_wls = wavelength_range(400, 920)
all_wls_norm = [f'n{b}' for b in all_wls]


# #######################  BASIC FUNCTIONS  #######################
def listify(*args):
    """Transform each passed arg into a list, to avoid errors trying to loop through non-iterable instances"""
    result = []
    for arg in args:
        result.append(arg if isinstance(arg, slice) or isinstance(arg, list) else ([] if arg is None else [arg]))
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


# #######################  FILES FUNCTIONS  #######################
def check_file(path, file_name):
        """Check if a file exists in the given path"""
        path = Path(path)

        if path.exists():
            for f in path.iterdir():
                if file_name in f.name:
                    return f

        return False


def get_file_datetime(file: Path, ft='%Y-%m-%d %H:%M:%S') -> str:
    """Get the formatted datetime of a specified file"""

    # if there is a wildcard '*' in the name, search for the first occurence
    if '*' in file.stem:
        files = [f for f in file.parent.glob(file.name)]
        if len(files) > 0:
            file = files[0]
        else:
            return 'Not Found'

    if file.exists():
        dt = datetime.fromtimestamp(file.stat().st_mtime)
        return dt.strftime(ft)
    else:
        return 'Not Found'


def get_file_by_suffix(path, stem, suffixes=None):
    """Get the file that matches the suffix in the order of preference of suffixes"""
    if suffixes is None:
        suffixes = ['.csv', '.txt', '.mlb']

    for suffix in suffixes:
        f = check_file(path, stem + suffix)
        if f:
            return f

    return False


def copy_dir(old_dir, new_dir, pattern='*'):
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    for old_f in old_dir.glob(pattern):
        if not old_f.is_dir():
            new_f = new_dir / old_f.name
            shutil.copy(old_f, new_f)


def match_files_names(files, names_lst):
    """
    Given two lists (files and names) check if all the names appear in the files.
    This function serve to check for the completeness of the directory.
    """
    stems = [f.stem for f in files]

    for name in names_lst:
        if name not in stems:
            return False
    return True


def get_complete_dirs(base_dir, names_lst):
    """
    Go through the subdirectories in base_dir and return those that are completed, that contains all the
    files in names_lst.
    """
    # get all the subdirectories in the base_dir
    dirs = [d for d in base_dir.rglob("*") if d.is_dir()]

    complete_dirs = []
    # look for directories that have the "full structure"
    for d in dirs:
        files = [f for f in d.iterdir() if not f.is_dir()]

        if match_files_names(files, names_lst):
            complete_dirs.append(d)

    return complete_dirs


# #######################  Date Time Conversion Functions   #######################
def from_excel_date(ordinal, epoch=datetime(1900, 1, 1)):
    # Adapted from above, thanks to @Martijn Pieters

    if ordinal > 59:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    in_days = int(ordinal)
    frac = ordinal - in_days
    in_secs = int(round(frac * 86400.0))

    return epoch + timedelta(days=in_days - 1, seconds=in_secs)  # epoch is day 1


def to_excel_date(dt, epoch=datetime(1900, 1, 1)):
    td = dt - epoch
    return round(td.days + 2 + td.seconds / 86400, 6)


# #######################  COLOR FUNCTIONS  #######################
def get_color(loc, colorscale_name):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


# #######################  PLOTTING FUNCTIONS  #######################
def apply_subplot(fig, subplot, position):
    """
    Given a main figure and a subplot, applies the subplot into the figure.
    :param fig: Main figure, previously created by plotly.subplots.make_subplots
    :param subplot: Another figure to be copied into the main figure
    :param position: Position in the main figure, initiating in (row=1, col=1)
    :return: The figure with the subplot applied.
    """
    for t in subplot.data:
        # check if there is a color_bar to adjust its size
        if hasattr(t.marker, 'colorbar') and t.marker.colorbar.len:
            # get the y_axis domain for this specific subplot
            y_domain = fig.get_subplot(position[0], position[1]).yaxis.domain

            # adjust the colorbar accordingly
            t.marker.colorbar.len = y_domain[1] - y_domain[0]
            t.marker.colorbar.y = (y_domain[1] + y_domain[0]) / 2

        fig.add_trace(t, row=position[0], col=position[1])

    for axis, update_ax_func in zip(['xaxis', 'yaxis'], [fig.update_xaxes, fig.update_yaxes]):
        update_dic = {}
        for param in ['title', 'type']:
            update_dic.update({param: subplot.layout[axis][param]})
        update_ax_func(update_dic, row=position[0], col=position[1])

    return fig


def plot_mean_std_traces(df, wls, std_delta=1., opacity=0.2, shaded=True, showlegend=False):

    mean = df[wls].mean()

    if shaded:
        std = df[wls].std()
        upper = mean + std*std_delta
        lower = mean - std*std_delta
    else:
        upper = lower = None

    traces = []

    transparent_color = f'rgba(150, 50, 50, {opacity})'
    if shaded:
        traces.append(go.Scatter(x=wls, y=upper, showlegend=showlegend, mode=None,
                                 line=dict(width=2, color='black', dash='dot'),
                                 name=f'upper ({std_delta} std)'
                                 ))
        traces.append(go.Scatter(x=wls, y=lower, fill='tonexty', showlegend=showlegend, mode=None,
                                 fillcolor=transparent_color,
                                 line=dict(width=2, color='black', dash='dot'),
                                 name=f'lower (-{std_delta} std)'
                                 ))

    traces.append(go.Scatter(x=wls, y=mean, name=f'Mean', line_color='red',
                             showlegend=showlegend))

    return traces


# ------------------- INTERPOLATION ------------------
def convert_columns_titles_types(df, data_type=float):
    """Change the columns titles names to a numerical format to interpolate the values accordingly"""
    for column in df.columns:
        try:
            new_name = data_type(column)

            if data_type is not str:
                frac, _ = math.modf(new_name)
                if frac == 0:
                    new_name = int(new_name)

            df = df.rename(columns={column: new_name})

        except:
            pass

    return df


def sort_numerical_columns_titles(df):
    """Order the numerical columns titles in ascending order. Will consider numerical,
    all the columns whose titles are not strings"""
    str_columns = [column for column in df.columns if isinstance(column, str)]
    num_columns = [column for column in df.columns if not isinstance(column, str)]
    num_columns.sort()
    return df[str_columns + num_columns]


def sort_column_titles(df):
    new_df = convert_columns_titles_types(df, float)
    new_df = sort_numerical_columns_titles(new_df)
    new_df = convert_columns_titles_types(new_df, str)
    return new_df


def create_evenly_spaced_columns(df, step=1, dtype=int, min_col=320, max_col=950):
    num_columns = [column for column in df.columns if not isinstance(column, str)]
    num_columns.sort()

    min_column = num_columns[0] if min_col is None else min_col
    max_column = num_columns[-1] if max_col is None else max_col

    start = math.ceil(min_column)
    end = math.floor(max_column)

    new_columns = np.arange(start, end, step, dtype=dtype)
    # clean the new columns, with existent numerical columns
    new_columns = [c for c in new_columns if c not in num_columns]

    df[new_columns] = pd.DataFrame([[np.nan for _ in new_columns]], index=df.index)
    return list(new_columns)


def create_interpolated_columns(df, step=1, drop_original_columns=True, create_id=True, min_col=None, max_col=None):
    """Create evenly spaced columns (wavelengths), according to a given step and interpolate the values linearly"""
    df = convert_columns_titles_types(df)
    new_columns = create_evenly_spaced_columns(df, step=step, dtype=type(step), min_col=min_col, max_col=max_col)

    # get the numerical and the string columns to treat them separately
    num_columns = [column for column in df.columns if not isinstance(column, str)]
    str_columns = [column for column in df.columns if isinstance(column, str)]
    num_columns.sort()

    # convert all the values to float32 format
    for column in num_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')

    # proceed the interpolation
    df.loc[:, num_columns] = df[num_columns].astype('float').interpolate(method='index',
                                                                         axis=1,
                                                                         limit_area='inside')

    columns = new_columns if drop_original_columns else num_columns
    df = df[str_columns + columns]

    if create_id:
        df = df.reset_index().rename(columns={'index': 'Id'})

    # convert the columns titles back to strings
    convert_columns_titles_types(df, data_type=str)
    return df
