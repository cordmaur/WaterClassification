# Common functions and definitions that serves all subpackages

import numpy as np
import dill
import sys
import io
import plotly.io as pio
from IPython.core.display import display, HTML
from PIL import Image


# basic functions/definitions to be loaded when importing *
# __all__ = ['listify', 's2bands']

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


def one_hot(v):
    """
    Create a one-hot encoded matrix for the vector v. m columns will be created to represent
    each integer in a one-hot encoding "fashion".
    :param v: input vector with integer numbers
    :return: matrix of shape (n x max(v) + 1)
    """
    b = np.zeros((v.size, v.max()+1))
    b[np.arange(v.size),v] = 1
    return b


def calc_area(df, columns=None, col_name="area", norm_band=None):
    """
    Calc the integral of the curve and adds it to a new column.
    Each curve must be in a row in a DataFrame
    :param df: DataFrame where each row contains a curve.
    :param columns: The columns with the values of the curve. If not informed, uses all column which have
                    names as digits (ex. 100, 110, 120, ...)
    :param col_name: The name of the column to be added to the DataFrame
    :param norm_band: the normalization band reflectance will be used to subtract the curve
    :return: The DataFrame with the newly created column
    """

    columns = df.columns[df.columns.str.isdigit()] if columns is None else columns

    values = df.fillna(0)[columns].to_numpy()

    if norm_band is not None:
        norm_vector = df.fillna(0)[norm_band].to_numpy()
        values = values - norm_vector[..., None]

    df[col_name] = np.trapz(values)
    return df


def serialized_sizeof(obj):
    """
    Get the total size of an object, after serialization. This function is used instead of sys.getsizeof because the
    former returns only the size directly attributed, but does not compute the references.
    :param obj: any python object
    :return: size of the object in Mb
    """
    serialized = io.BytesIO()
    save_obj(obj, serialized)
    size = sys.getsizeof(serialized)/1024/1024
    serialized.close()
    return f'{size} Mb'


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


# #######################  INPUT/OUTPUT FUNCTIONS  #######################
def save_obj(obj, file):
    # if name is a string, we assume it is a path, otherwise, it should be an io stream
    if isinstance(file, str):
        with open(str(file), 'wb') as dill_file:
            dill.dump(obj, dill_file)
    else:
        dill.dump(obj, file)


def load_obj(name):
    with open(str(name), 'rb') as f:
        result = dill.load(f)
    return result


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


def superpose_figs(figs, trace_names=None):

    fig = figs.pop(0)

    for sub_fig in figs:
        sub_fig.for_each_trace(fig.add_trace)

    if trace_names:
        for trace, name in zip(fig.data, trace_names):
            trace.name = name
            trace.hovertext = name
            trace.hovertemplate = None
            trace.line = None

    return fig


# #######################  FIG/HTML CONVERTING FUNCTIONS  #######################
def fig_to_html(fig, buttonsToRemove=[], **kwargs):
    """Converts a plotly figure into a HTML graph and displays it. That is necessary to maintain the interactive
    functionality of plotly in the jekyll documentation, on Github."""

    html = io.StringIO()
    buttonsToRemove += ['toggleSpikelines', 'hoverCompareCartesian', 'zoomIn2d',
                        'zoomOut2d', 'autoScale2d', 'hoverClosestCartesian']
    pio.write_html(fig, file=html, auto_open=False,
                   config={'modeBarButtonsToRemove': buttonsToRemove,
                           'displaylogo': False,
                          }
                  )

    display(HTML(data=html.getvalue()))


def showfig(fig, publish_mode='fig', **kwargs):
    if publish_mode == 'html':
        fig_to_html(fig, **kwargs)

    elif publish_mode == 'png':
        png = io.BytesIO()
        pio.write_image(fig, file=png, **kwargs)

        return Image.open(png)
    else:
        return fig
