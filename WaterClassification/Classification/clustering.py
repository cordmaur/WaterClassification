# This module has the classes to perform the clustering

import numpy as np
import pandas as pd
from sklearn import cluster
import skfuzzy  # for fuzzy c means

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer

from WaterClassification.common import listify, apply_subplot, one_hot, superpose_figs, hex_to_rgb, all_wls, \
    all_wls_norm
from WaterClassification.Fitting import GroupFit, BaseFit
from WaterClassification.Fitting.multiprocessing import PoolManager
from math import ceil

import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots


# Wrapper for the fuzzy c means class
class FuzzyCMeans:
    def __init__(self, n_clusters=2, m=2.):
        self.data = None
        self.labels_ = None
        self.results = None
        self.n_clusters = n_clusters
        self.m = m

    def fit(self, data):
        self.data = np.transpose(data, (1, 0))
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(self.data,
                                                            self.n_clusters,
                                                            self.m,
                                                            error=0.005,
                                                            maxiter=1000,
                                                            init=None)

        self.results = dict(cntr=cntr, u=u, u0=u0, d=d, jm=jm, p=p, fpc=fpc)
        self.labels_ = np.argmax(u, axis=0)


class ClusteringEngine:
    def __init__(self, df, cluster_features, n_clusters, cluster_column='cluster', algo='k-means', norm=None,
                 weights=None):

        self.n_clusters = n_clusters
        self.cluster_column = cluster_column
        self.cluster_features = cluster_features
        self.group_fit = None
        self.fcm = None

        self.df, self.algo = self.clusterize(df=df,
                                             columns=cluster_features,
                                             n_clusters=n_clusters,
                                             cluster_column=cluster_column,
                                             algo=algo,
                                             norm=norm,
                                             weights=weights)

    def fit(self, fit_features, funcs, y='SPM', metric=BaseFit.rmsle, ignore_none=True, thresh=10,
            n_cpus=None, pool=None, metrics=None, optimize_metric=False):

        self.group_fit = GroupFit(self.df, fit_features, funcs, expr_y=y, metric=metric, ignore_none=ignore_none,
                                  thresh=thresh, n_cpus=n_cpus, pool=pool, metrics=metrics,
                                  group_column=self.cluster_column, optimize_metric=optimize_metric)

        # keep the results sorted. That can be changed "a posteriori"
        self.sort_by_variable(variable=y)

    def sort_by_variable(self, variable='SPM', ascending=True):

        # create a series grouped by the cluster and the variable mean value
        series = self.df.groupby(by=self.cluster_column).mean()[variable]

        # create a temporary column to receive it's new values
        self.df['_temp'] = self.df[self.cluster_column]

        # create a variable to hold the final mapping
        mapping = {}
        for i, c in enumerate(series.sort_values(ascending=ascending).index):
            mapping[c] = i

            # adjust the values in the _temp column
            self.df.loc[self.df[self.cluster_column] == c, '_temp'] = i

        # when finished, copy the new adjusted values to the original column and drop the _temp
        self.df[self.cluster_column] = self.df['_temp'].astype('uint8')
        self.df.drop(columns='_temp', inplace=True)

        # adjust the group_fit, if it exists
        if self.group_fit is not None:
            self.group_fit.rename_groups(mapping)

        return mapping

    @staticmethod
    def clusterize(df, columns, inf_columns=None, n_clusters=2, cluster_column='cluster', algo='k-means',
                   norm=None, weights=None):
        """
        Given a DataFrame, do the clustering using the given columns  and save the cluster number
        in a new column (cluster_column).
        :param df: Datafame with the features
        :param columns: The columns that will be used to do the clustering procedure.
        :param inf_columns: The columns to be kept in the resulting dataframe
        :param n_clusters: Number of clusters
        :param cluster_column: The name of the column with the clustering result
        :param algo: The clustering algorithm to be used ('k-means', 'agglomerative', 'FCM')
        :param norm: The normalization method: None, 'Standard', 'MinMax', 'Robust', 'Normalizer', 'Power'
        :param weights: Weights to be applied to each features after normalization. If None, no weights are applied
        :return: DataFrame with a new column indicating the cluster number and the fitted algorithm
        """

        x_train = df[columns].to_numpy()

        if norm:
            if norm == 'Standard':
                scaler = StandardScaler()
            elif norm == 'MinMax':
                scaler = MinMaxScaler()
            elif norm == 'Robust':
                scaler = RobustScaler()
            elif norm == 'Normalizer':
                scaler = Normalizer()
            elif norm == 'Power':
                scaler = PowerTransformer()
            else:
                print(f'Normalization method {norm} not supported')
                return

            x_train = scaler.fit_transform(x_train)

            if weights:
                x_train = x_train * np.array(weights)[None, ...]

        if algo == 'k-means':
            clustering = cluster.KMeans(n_clusters=n_clusters)
        elif algo == 'agglomerative':
            clustering = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        elif algo == 'FCM':
            clustering = FuzzyCMeans(n_clusters=n_clusters, m=2)
        elif algo == 'FCM2':
            clustering = FuzzyCMeans(n_clusters=n_clusters, m=1.3)

        else:
            print(f'Algorithm [{algo}] not implemented')
            return

        clustering.fit(x_train)

        inf_columns = df.columns if inf_columns is None else listify(inf_columns)

        if cluster_column in inf_columns:
            inf_columns = inf_columns.drop(cluster_column)

        cluster_df = pd.concat([df[inf_columns],
                                pd.DataFrame(clustering.labels_.astype('uint8'),
                                             index=df.index,
                                             columns=[cluster_column])],
                               axis=1)

        return cluster_df, clustering

    # ################################  BLENDING ALGO METHODS  #################################
    def calc_all_algos_predictions(self):
        """
        Calculate the predictions (y_hat) using all the models (1 model for each cluster).
        :return: Array of shape (n x n_clusters)
        """
        results = []
        for i in range(self.n_clusters):
            best_fit = self[i].best_fit
            func = best_fit.func
            params = best_fit.fit_params['params']
            expr_x = best_fit.expr_x
            x = BaseFit.parse_expr_df(self.df, expr_x)

            results.append(func(x, *params))

        y_hats = np.array(results).transpose((1, 0))

        return y_hats

    def calc_blended_predictions(self, threshold=0.6, factor=3):
        """
        Calculate the predictions, blending the various models. The weights of the models will be,
        the "similarity" obtained from the Fuzzy-c-means.
        :param threshold: Threshold to consider the measurement to be well contained in one cluster
        :param factor: This factor is to avoid blending discrepant (ratio > factor) models.
        :return: (n,) vector with the new predictions considering the blending
        """

        # First, check if the algo is FuzzyCMeans.
        if not isinstance(self.algo, FuzzyCMeans):
            print(f'The clustering algorithm is not FuzzyCMeans. It is necessary to compute the weights of the models')
            return

        # Check if the clustering is already fitted
        if not self.group_fit:
            print('Clustering not yet fitted. Execute .fit() method before calculating predictions')
            return

        # get the predictions considering all the models
        y_hats = self.calc_all_algos_predictions()

        # get the "weights" for each cluster
        u = self.algo.results['u'].transpose(1, 0)

        # Calculate the predictions as if the measurements were contained in just 1 cluster
        # Will call that hard (clustering) predictions
        hard_pred = np.nansum(y_hats * one_hot(u.argmax(axis=1)), axis=1)

        # if there is just 1 cluster, the blending is the same as the hard prediction
        if self.n_clusters == 1:
            return hard_pred

        # Get the other weights that pass the minimum threshold requirement
        # Other requirement is that y_hat must be positive
        # Last requirement is that the ratio between the prediction and the other predictions
        # must lie in between the 1/factor and factor
        min_threshold = (1 - threshold) / (self.n_clusters - 1)

        valid_ys = y_hats / (y_hats[np.arange(len(y_hats)), u.argmax(axis=1)])[..., None]
        valid_ys = ((1/factor) < valid_ys) & (valid_ys < factor)

        u_valid = ((u > min_threshold) & (y_hats > 0) & valid_ys).astype('int')

        # Get the new u matrix, only where it is valid
        new_u = u * u_valid

        # and normalize it
        new_u = new_u / new_u.sum(axis=1)[..., None]

        # The new_u is ready to be applied to the y_hats. Let's create the blending
        # (soft predictions) based on the new weights for all the matrix
        soft_pred = np.nansum(y_hats * new_u, axis=1)

        # Last thing, we must take care of the threshold
        to_change = u.max(axis=1) < threshold
        blended = np.where(to_change, soft_pred, hard_pred)

        return blended

    def blended_results(self, threshold=0.6, factor=3):
        blended = self.calc_blended_predictions(threshold=threshold, factor=factor)
        if blended is not None:
            return BaseFit.test_fit(blended, self.df[self.group_fit.variable])

    # ################################  PLOTTING METHODS  #################################
    def plot_clustering(self, x, y='SPM', fig=None, position=None, **kwargs):

        # force a color mapping so the cluster 0 has the first color and so on...
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24
        color_discrete_map = {i: colors[i] for i, _ in enumerate(colors)}

        # create a scatter plot with the clustering
        new_fig = px.scatter(self.df, x=x, y=y, color=self.cluster_column,
                             color_discrete_map=color_discrete_map, **kwargs)

        # if existing fig, apply the scatter into the correct position
        fig = new_fig if fig is None else apply_subplot(fig, new_fig, position)

        return fig

    def plot_mean_reflectances(self, wls, std_delta=1., opacity=0.2, shaded=True, showlegend=True):

        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24

        clusters = self.df[self.cluster_column].unique()
        clusters.sort()

        mean = self.df.groupby(by=self.cluster_column)[wls].mean()

        if shaded:
            std = self.df.groupby(by=self.cluster_column)[wls].std()
            upper = mean + std*std_delta
            lower = mean - std*std_delta
        else:
            upper = lower = None

        fig = go.Figure()

        # convert the wls in numeric (if possible)
        # wls = [int(wl) for wl in wls]

        for c in clusters:
            y = mean.loc[c]
            fig.add_trace(go.Scatter(x=wls, y=y, name=f'Cluster {c}', line_color=colors[c],
                                     showlegend=showlegend))

            transparent_color = f"rgba{(*hex_to_rgb(colors[c]), opacity)}"
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

    def plot_box(self):
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Light24

        clusters = sorted(self.df[self.cluster_column].unique())

        fig = go.Figure()
        for c in clusters:
            box = go.Box(y=self.df[self.df[self.cluster_column] == c][self.group_fit.variable],
                         name=f'Cluster {c}', line=dict(color=colors[c]))
            fig.add_trace(box)

        fig.update_yaxes(type='log', title=self.group_fit.variable)
        fig.update_xaxes(title='Cluster')

        return fig

    def plot_summary(self, scatter_x='area', base_height=400):
        fig = subplots.make_subplots(rows=2, cols=2,
                                     subplot_titles=['Scatter',
                                                     'BoxPlot',
                                                     'Mean Reflectances (RAW)',
                                                     'Mean Reflectances (NORM)'])

        self.plot_clustering(x=scatter_x, log_y=True, fig=fig, position=(1,1))
        # fig = apply_subplot(fig, scatter, position=(1, 1))

        for col, wls in enumerate([all_wls, all_wls_norm]):
            mean = self.plot_mean_reflectances(wls)
            fig = apply_subplot(fig, mean, position=(2, col+1))

        box = self.plot_box()
        fig = apply_subplot(fig, box, position=(1, 2))

        fig.update_layout(showlegend=False, height=base_height*2)

        return fig

    def summary(self):
        if self.group_fit:
            print(f'Clustering (n={self.n_clusters}) with {len(self.group_fit.group_fits)} fits')
            print(f'Functions: {self.group_fit.funcs}')
            print(f'Bands: {self.group_fit.bands}')

            return self.group_fit.summary()
        else:
            return {cluster: count
                    for cluster, count in self.df[self.cluster_column].value_counts().iteritems()}

    def list_quantities(self):
        return self.df[self.cluster_column].value_counts().to_list()

    def __repr__(self):
        s = f'ClusteringEngine with {self.n_clusters} clusters'
        if self.group_fit is not None:
            return s + ' - Fitted: ' + repr(self.group_fit)
        else:
            return s

    def __getitem__(self, item):
        return self.group_fit[item]


class MultiClustering:
    def __init__(self, df, cluster_features, range_clusters=(1, 5), cluster_column='cluster', algo='k-means',
                 norm=None, weights=None):

        self.clusters = {}
        for k in range(*range_clusters):
            self.clusters[k] = ClusteringEngine(df,
                                                cluster_features=cluster_features,
                                                n_clusters=k,
                                                cluster_column=cluster_column,
                                                algo=algo,
                                                norm=norm,
                                                weights=weights)

        self.cluster_params = {
            'cluster_features': cluster_features,
            'norm': norm,
            'weights': weights,
            'algo': algo
        }

        # the fitting parameters will be filled when call the fit method
        self.fit_params = None

        # self.clusters_features = cluster_features
        # self.norm = norm
        # self.range_clusters = range_clusters
        # self.cluster_column = cluster_column
        # self.weight = weights

    def fit(self, fit_features, funcs, y='SPM', metric=BaseFit.rmsle, ignore_none=True, thresh=10,
            n_cpus=None, pool=None, metrics=None, optimize_metric=False):

        metrics = BaseFit.available_metrics if metrics is None else metrics

        with PoolManager(n_cpus, pool) as pool_mgr:
            for clustering in self.clusters.values():
                clustering.fit(fit_features=fit_features,
                               funcs=funcs,
                               y=y,
                               metric=metric,
                               ignore_none=ignore_none,
                               thresh=thresh,
                               n_cpus=n_cpus,
                               pool=pool_mgr.pool,
                               metrics=metrics,
                               optimize_metric=optimize_metric)

        self.fit_params = {
            'fit_features': fit_features,
            'funcs': funcs,
            'y': y,
            'metrics': metrics,
            'optimize_metric': optimize_metric
        }

    def summary(self):
        # print(self.cluster_params)
        # print(self.fit_params)
        if self.fit_params:
            params = BaseFit.available_metrics + ['qty']
            results = {key: clustering.group_fit.summary().loc['overall'][params]
                       for key, clustering in self.clusters.items()}

            results = pd.DataFrame(results)
            for key, clustering in self.clusters.items():
                results.loc['Samples', key] = clustering.list_quantities()

            return results

        else:
            return self.clusters

    def blended_summary(self, threshold=0.6, factor=3):
        results = {key: clustering.blended_results(threshold=threshold, factor=factor)
                   for key, clustering in self.clusters.items()}
        return pd.DataFrame(results)

    def save_memory(self, max_n=10):
        """
        Save memory by deleting non-relevant fits and leaving just the best fits.
        :param max_n: maximum number of fits to be stored
        :return: None
        """
        for clustering in self.clusters.values():
            clustering.group_fit.save_memory(max_n=max_n)

    def sort_by_variable(self, variable='SPM', ascending=True):
        for clustering in self.clusters.values():
            clustering.sort_by_variable(variable=variable, ascending=ascending)

    # ################################  PLOTTING METHODS  #################################
    def plot_parameter(self, metric=BaseFit.rmsle, additional_clusters=None):
        series = self.summary().loc[repr(metric)]
        series.name = repr(metric)

        return px.line(series)

    @staticmethod
    def plot_metric_by_cluster(mclusters, names, metric=BaseFit.rmsle):
        figs = [mcluster.plot_parameter(BaseFit.rmsle) for mcluster in mclusters]
        fig = superpose_figs(figs, names)
        fig.update_xaxes(title='No. of Clusters')
        fig.update_yaxes(title=str(metric))
        return fig

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        return self.clusters[item]

    def plot_clustering(self, x, y='SPM', cols=3, marker_size=2, base_height=300, update_layout=None, **kwargs):

        # define the rows
        rows = ceil(len(self)/cols)

        # create the titles
        titles = [f'Cluster {key}' for key in self.clusters]

        # create the grid figure
        fig = subplots.make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        for i, clustering in enumerate(self.clusters.values()):
            position = ((i // cols) + 1, (i % cols) + 1)
            clustering.plot_clustering(x=x, y=y, fig=fig, position=position, **kwargs)

        basic_update = {'showlegend': False, 'height': base_height * rows}
        if update_layout is not None:
            basic_update.update(update_layout)

        fig.update_layout(basic_update)

        fig.update_traces(marker=dict(size=marker_size))
        return fig






