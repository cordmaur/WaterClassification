# basic functions/classes that serves the Fitting package
import numpy as np
from math import sqrt, ceil
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

import plotly
import plotly.graph_objects as go
import plotly.express as px

from WaterClassification import common
import pandas as pd

import parser


class OptimizeCriteria:
    Maximize = 1
    Minimize = 2


class Metric:
    """
    The metric class, represents a specific metric that can be used to evaluate the performance of the models.
    """

    def __init__(self, name=None, func=None, optimize_criteria=OptimizeCriteria.Minimize):
        """Initialize a Metric class with a name and a metric function"""
        self.metric_func = func
        self.name = name
        self.optimize_criteria = optimize_criteria

    def __call__(self, y=None, y_hat=None, decimal=None):
        """Calculate the metric given targets (y) and predictions (y_hat)"""

        if self.metric_func:

            # given that RMSLE is not suitable for negative targets, we will clip y < 0
            if self == 'RMSLE':
                y_hat = np.where(y_hat < 0, 0, y_hat)

            if decimal:
                return round(self.metric_func(y, y_hat), decimal)
            else:
                return self.metric_func(y, y_hat)
        else:
            print('Metric function has to be implemented')

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        else:
            return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name}'

    def loss_func(self, params, func, x, y):
        """
        Loss function based on the metric to be used in the optimization package (minimize SCIPY)
        :param params: parameters of the function (as a list)
        :param func: function
        :param x: independent variables
        :param y: targets
        :return: Metric loss
        """

        # evaluate the predictions (y_hat) given the function
        y_hat = func(x, *params)

        # return the metric's loss
        if self.optimize_criteria == OptimizeCriteria.Minimize:
            return self(y, y_hat)
        else:
            return -self(y, y_hat)


class FittingFunction:
    def __init__(self, name, func, dim=1):
        self.name = name
        self.func = func
        self.dim = dim

    def __call__(self, *params):
        return self.func(*params)

    def __repr__(self):
        return f'{self.name}'

    def __str__(self, short=True):
        if short:
            return f'{self.name}'
        else:
            return f'Function {self.name}{str(self.func.__code__.co_varnames)}'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        else:
            return self.__hash__() == other.__hash__()


class Functions:
    linear = FittingFunction(name='Linear',
                             func=lambda x, a, b: a * x + b)

    expo = FittingFunction(name='Exponential',
                           func=lambda x, a, b: a * 10 ** x + b)

    power = FittingFunction(name='Power',
                            func=lambda x, a, b, c: a * x ** b + c)

    nechad = FittingFunction(name='Nechad',
                             func=lambda red, a=610.94, c=0.2324, d=1: a * np.pi * red / (1 - (np.pi * red / c)) + d)

    linear2 = FittingFunction(name='Linear2',
                              func=lambda x, a, b, c: a*x[:, 0] + b*x[:, 1] + c,
                              dim=2)

    power2 = FittingFunction(name='Power2',
                             func=lambda x, a, b, c, d, e: a * x[:, 0] ** b + c*x[:, 1]**d + e,
                             dim=2)

    nechad3 = FittingFunction(name='Nechad2',
                              func=lambda x, a1, c1, a2, c2, d: a1 * x[:, 0] / (1 - (x[:, 0] / c1)) +
                                                                a2 * x[:, 1] / (1 - (x[:, 1] / c2)) + d,
                              dim=2)

    expo2 = FittingFunction(name='Exponential2',
                            func=lambda x, a, b, c: a * 10 ** x[:, 0] + b * 10 ** x[:, 1] + c,
                            dim=2)

    lin_power = FittingFunction(name='Linear_Power',
                                func=lambda x, a, b, c, d: a*x[:, 0] + b*x[:, 1]**c + d,
                                dim=2)

    lin_expo = FittingFunction(name='Linear_Expo',
                               func=lambda x, a, b, c: a*x[:, 0] + b * 10 ** x[:, 1] + c,
                               dim=2)

    lin_nechad = FittingFunction(name='Linear_Nechad',
                                 func=lambda x, a, b, c, d: a*x[:, 0] + b * x[:, 1] / (1 - (x[:, 1] / c)) + d,
                                 dim=2)

    power_expo = FittingFunction(name='Power_Expo',
                                 func=lambda x, a, b, c, d: a * x[:, 0] ** b + c * 10 ** x[:, 1] + d,
                                 dim=2)

    power_nechad = FittingFunction(name='Power_Nechad',
                                   func=lambda x, a, b, c, d, e: a * x[:, 0] ** b +
                                                                 c * x[:, 1] / (1 - (x[:, 1] / d)) + e,
                                   dim=2)

    expo_nechad = FittingFunction(name='Expo_Nechad',
                                  func=lambda x, a, b, c, d: a * 10 ** x[:, 0] + b * x[:, 1] / (1 - (x[:, 1] / c)) + d,
                                  dim=2)

    available_funcs = [linear, expo, power, nechad]

    @staticmethod
    def fitting_func(func):
        if func is None:
            return None
        elif isinstance(func, FittingFunction):
            return func
        else:
            return FittingFunction(name=func.__name__, func=func)


class BaseFit:
    """
    Has the all basic definitions and functionalities that are independent of an instance (@Static)
    """

    # ######################### METRICS ###################################
    r2 = Metric(name='R^2', func=r2_score, optimize_criteria=OptimizeCriteria.Maximize)
    mse = Metric(name='MSE', func=mean_squared_error)
    rmse = Metric(name='RMSE', func=lambda y, y_hat: sqrt(mean_squared_error(y, y_hat)))
    rmsle = Metric(name='RMSLE', func=lambda y, y_hat: sqrt(mean_squared_log_error(y, y_hat)))
    sse = Metric(name='SSE', func=lambda y, y_hat: ((y - y_hat) ** 2).sum())

    available_metrics = [r2, mse, rmse, rmsle, sse]

    summary_params = ['func', 'band'] + available_metrics + ['params', 'qty']

    # #########################  PREDEFINED FITTING FUNCTIONS  #########################
    available_funcs = Functions.available_funcs

    # #########################  FITTING METHODS  #########################
    @staticmethod
    def test_fit(x, y, func=None, params=None, decimal=4, metrics=None):
        """Compute the predictions, calculate the errors and return in a dictionary.
        If there is no func, assume x is already the y_hat"""

        # convert the given function to a fitting function
        func = Functions.fitting_func(func)

        # get the metrics to be tested
        metrics = metrics if metrics is not None else BaseFit.available_metrics

        # calculate the predictions (or assume they have been passed)
        if func is None:
            y_hat = np.array(x)
        else:
            x = np.where(x < 0, 0.1, x)
            y_hat = func(np.array(x), *params)

        # evaluate the metrics
        res = {}
        for metric in metrics:
            res[metric] = metric(y, y_hat, decimal=decimal)

        res.update({'params': params,
                    'y_hat': y_hat})

        return res

    @staticmethod
    def test_combined_fit(fit_list, summary=True):
        """Given a list of fits, it will combine the targets and the predictions and output a combined evaluation"""
        fit_list = common.listify(fit_list)

        targs = np.array([])      # initialize an empty array for the targets
        preds = np.array([])      # initialize an empty array for the predictions
        for fit in fit_list:
            targs = np.concatenate([targs, fit.fit_params['y']])
            preds = np.concatenate([preds, fit.fit_params['y_hat']])

        # now we have two vectors (targs and preds), we can call the test_fit method
        res = BaseFit.test_fit(preds, targs)

        if summary:
            return {i: res[i] for i in BaseFit.available_metrics}
        else:
            return res

    @staticmethod
    def sort_fits(fits, metric=rmsle, reverse=False):
        """
        Sort a list of fits based on a given metric. The order will depend on the metric. Better will come first.
        For errors it will order ascending, and for score it will order descending, unless reverse is True
        :param fits: a list of Fits
        :param metric: a metric to be tested
        :param reverse: If the order should be reversed (worst first)
        :return: a list of sorted fits
        """
        fitted_fits = filter(lambda x: metric in x.fit_params.keys(), fits)
        empty_fits = filter(lambda x: metric not in x.fit_params.keys(), fits)

        reverse = reverse if metric.optimize_criteria == OptimizeCriteria.Minimize else not reverse
        ordered = sorted(fitted_fits, key=lambda x: x.fit_params[metric], reverse=reverse)
        return ordered + list(empty_fits)

    @staticmethod
    def _parse_expr_df(df, expr):
        if expr in df.columns:
            return df[expr]

        code = parser.expr(expr).compile()

        # create the variables in this context
        for var in code.co_names:
            col_name = var if var in df.columns else var[1:]
            exec(var + " = df[col_name]")

        # evaluate the expression
        return eval(code).rename(expr)

    @staticmethod
    def parse_expr_df(df, expr):
        """Gets an arbitrary expression using the columns of the dataframe. The band columns should start with the
        letter b (ex. b865)"""
        # if the expression is already a column, return it
        if expr in df.columns:
            return df[expr]

        # otherwise, if it is not a tuple, parse the expression
        elif isinstance(expr, str):
            return BaseFit._parse_expr_df(df, expr)

        # if it is a tuple, group the results into a numpy array
        else:
            series = [BaseFit._parse_expr_df(df, e) for e in expr]
            return pd.DataFrame(series).T.to_numpy()

    # #########################  PLOTTING METHODS  #########################
    @staticmethod
    def plot_pred_vs_targ(df, y, y_hat, color_group=None, title='', **kwargs):

        # calculate the overall metric (again)
        overall = BaseFit.test_fit(df[y_hat], df[y])

        # force a color mapping so the cluster 0 has the first color and so on...
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24
        color_discrete_map = {key: colors[i] for i, key in enumerate(df[color_group].unique())}

        fig = px.scatter(df, x=y, y=y_hat, color=color_group, color_discrete_map=color_discrete_map,
                         hover_data=[df.index, 'Area', 'Station'], **kwargs)

        # trace the y=x line
        max_value = df[[y, y_hat]].max().max()
        min_value = df[[y, y_hat]].min().min()

        fig.add_trace(go.Scatter(x=[min_value, max_value], y=[min_value, max_value], mode='lines'))

        title = title + f"<br>R^2={overall[BaseFit.r2]} | RMSLE={overall[BaseFit.rmsle]} | RMSE={overall[BaseFit.rmse]}"
        return fig.update_layout(title=title, showlegend=False)


class PlotFit:
    """
    Holds some general plotting functions that are not instance dependant
    """
    @staticmethod
    def plot_mean_reflectances(df, group_by, wls=None, std_delta=1., opacity=0.2, shaded=True, showlegend=True):
        """Plot the mean reflectances of a given dataframe and a group column"""

        # get the wavelengths to plot
        wls = common.all_wls if wls is None else wls

        # get a discrete color list (colors will be controlled manually)
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24

        # get the groups that will be plotted and order them ascending
        groups = df[group_by].unique()
        groups.sort()

        # calculate the mean reflectance for each group
        mean = df.groupby(by=group_by)[wls].mean()

        if shaded:
            std = df.groupby(by=group_by)[wls].std()
            upper = mean + std*std_delta
            lower = mean - std*std_delta
            # upper = df.groupby(by=group_by)[wls].max()
            # lower = df.groupby(by=group_by)[wls].min()
        else:
            upper = lower = None

        fig = go.Figure()

        for c in groups:
            y = mean.loc[c]
            fig.add_trace(go.Scatter(x=wls, y=y, name=f'Cluster {c}', line_color=colors[c],
                                     showlegend=showlegend))

            transparent_color = f"rgba{(*common.hex_to_rgb(colors[c]), opacity)}"
            if shaded:
                y_up = upper.loc[c]
                y_low = lower.loc[c]
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
    def draw_func_trace(func, params, x_interval, pts=100, txt=None):
        """
        Create a plotly trace given a function and proper parameters, x interval and the number of points.
        :param func: the function to be generated the trace
        :param params: the parameters of the function (will be passed in the function call)
        :param x_interval: it can be a tuple of min, max or the x domain
        :param pts: number of points to be used to generate the trace
        :param txt: additional text to be added to the hover_data of the trace
        :return: plotly trace (to be added to any figure)
        """
        xs = np.linspace(np.min(x_interval), np.max(x_interval), pts)
        ys = func(xs, *params)

        func = Functions.fitting_func(func)

        hover = f'Params: {params}<br>{txt}'

        return go.Scatter(x=xs, y=ys, mode='lines', name=repr(func), text=hover)

    @staticmethod
    def plot_fits(fits, cols, base_height, titles=None, **kwargs):
        """Internal function that plot's a list of fits"""

        # get the titles
        titles = [fit.title for fit in fits] if titles is None else titles

        # create the subplots
        rows = ceil(len(fits)/cols)
        fig = plotly.subplots.make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        # loop through the fits to plot them in the main figure
        for idx, fit in enumerate(fits):
            position = ((idx // cols) + 1, (idx % cols) + 1)
            fit.plot_fit(fig=fig, position=position, **kwargs)

        # update the final layout
        update_layout = {'height': base_height * rows}

        # Override with 'new' options from kwargs
        if 'update_layout' in kwargs:
            update_layout.update(kwargs['update_layout'])

        fig.update_layout(update_layout)
        fig.update_coloraxes({'colorscale': 'Plasma'})

        return fig

