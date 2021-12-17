from WaterClassification.common import *

from RadiometryTrios import RadiometryDB, BaseRadiometry
from WaterClassification.Fitting import BaseFit, Functions, MultiFit
from WaterClassification.Classification.clustering import *
from itertools import product

from WaterClassification.Fitting.multiprocessing import *
from datetime import datetime


def process_combination(combination, df_train, df_test, range_clusters, funcs, fit_bands):
    def parse_combination(combination):
        sampling, ref_type, normalization, algo = combination

        # first, get the input wavelenghts
        if isinstance(sampling, int):
            wls = wavelength_range(380, 940, step=sampling)
            wls_name = f'{sampling}nm'
        else:
            wls = sampling[0]
            wls_name = sampling[1]

        # create the bands, according to the reflectance types
        bands = []
        ref_type_name = ''
        for tp in listify(ref_type):
            ref_type_name = ref_type_name + tp
            if tp == 'raw':
                bands = bands + wls
            elif tp == 'norm':
                bands = bands + [f'n{b}' for b in wls]
            else:
                bands = bands + listify(tp)

        name = '_'.join([algo, wls_name, ref_type_name, str(normalization)])

        return bands, name, normalization, algo

    cluster_bands, name, normalization, algo = parse_combination(combination)

    print(f'#{name} - started at {datetime.now()}')

    mcluster = MultiClustering(df_train,
                               cluster_features=cluster_bands,
                               range_clusters=range_clusters,
                               algo=algo,
                               norm=normalization)

    mcluster.fit(fit_features=fit_bands,
                 funcs=funcs,
                 n_cpus=None,
                 pool=None,
                 optimize_metric=True)

    # once multi clustering is fitted, we need to eval its performance on the test set
    result_df = mcluster.summary(df_test)

    # save the pandas summary
    result_df.to_csv(f'./data/{name}.csv')

    # save the figure
    fig = mcluster.plot_clustering('area', log_y=True)
    pio.write_image(fig, f'./data/{name}.png', width=1200, height=800)

    return fig


def multi_process_clustering(df_train, df_test, range_clusters, samplings, ref_types, normalizations, algos, fit_bands,
                             funcs, calc=False):
    # create the combinations
    combinations = list(product(samplings, ref_types, normalizations, algos))

    df_test = df_train if df_test is None else df_test

    len_combinations = len(combinations)
    print(f'Total of {len_combinations} combinations were created.')

    if not calc:
        return

    pool = mp.ProcessingPool(ncpus=6)
    pool.restart(force=True)
    figs = pool.map(process_combination, combinations, [df_train] * len_combinations, [df_test] * len_combinations,
                    [range_clusters] * len_combinations, [funcs] * len_combinations, [fit_bands] * len_combinations)

    return figs