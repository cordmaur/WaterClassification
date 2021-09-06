import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from PIL import ImageColor

from datetime import datetime, timedelta
import math

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
def get_color(colorscale_name, loc):
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


def plot_reflectances(df, bands, color=None, hover_vars=None, colormap='viridis', log_color=True,
                      colorbar=True, show_legend=False):

    if color is not None:
        cs = df[color] if isinstance(color, str) else pd.Series(color.astype('int'), index=df.index)
        cs = (cs - cs.min())/(cs - cs.min()).max()

        min_color = cs[cs > 0].min()
        max_color = cs.max()

    else:
        min_color = None
        max_color = None
        cs = None

    scatters = []
    for idx in df.index:
        row = df.loc[idx]
        reflectances = row[bands]
        x = reflectances.index
        y = reflectances.values

        # color_value = f'rgb{cmap(norm(color_series.loc[idx]))[:3]}' if color is not None else 'grey'

        hover_text = f'Idx: {idx}'
        for var in listify(hover_vars):
            hover_text += f'{var}: {row[var]}<br>'


        scatters.append(go.Scatter(x=x.astype('float'), y=y,
                                   text=hover_text,
                                   name='', #str(idx),
                                   line=dict(width=0.5, color=f"{get_color('Viridis', cs.loc[idx])}"),
                                   # color=color_value),
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
                                        cmin=min_color,
                                        cmax=max_color,
                                        colorbar=dict(xanchor="left", title='', thickness=30,
                                                      tickvals=[min_color, (min_color + max_color) / 2, max_color],
                                                      ticktext=[min_color, (min_color + max_color) / 2, max_color],
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