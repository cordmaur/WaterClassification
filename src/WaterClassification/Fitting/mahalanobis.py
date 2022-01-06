from scipy.spatial import distance
import pandas as pd
import numpy as np


def assign_membership(base_df, group_by, new_df, bands, new_group='assigned_group', dist_func='mahalanobis',
                      groups=None):
    """
    Given a new dataframe - df, fill up the group column with corresponding clusters
    :param groups: groups to be tested for assignment. If None, all groups are tested.
    :param base_df: the dataframe with the original clusters (groups)
    :param group_by: the column which has the clusters (groups) identification
    :param dist_func: Distance algorithm to be used: 'mahalanobis', 'euclidean', 'seuclidean'
    :param new_group: column name to write the groups to
    :param new_df: new dataframe to be assigned membership
    :param bands: bands used for testing membership
    :return: new_df with a group column added.
    """

    # calc the variances for each band in the base_df
    var = base_df.groupby(by=group_by)[bands].var()

    # create the covariance matrix of the bands for each group
    cov = base_df.groupby(by=group_by)[bands].cov()

    # get the mean reflectance for each group
    mean = base_df.groupby(by=group_by)[bands].mean()

    # calculate the distances for each row of df to the clusters
    # first we will create an empty dataframe
    distances_df = pd.DataFrame(index=new_df.index)

    # get the groups to loop through.
    groups = mean.index if groups is None else groups

    for group in mean.index:
        # calculate the distances according to the distance algorithm
        if dist_func == 'mahalanobis':
            # First invert the covariance matrix for the group
            inv_cov = np.linalg.inv(cov.loc[group])

            # Calc Mahalanobis distances
            distances = new_df.apply(lambda row: distance.mahalanobis(row[bands],
                                                                      mean.loc[group],
                                                                      inv_cov),
                                     axis=1)

        elif dist_func == 'euclidean':
            # Calc euclidean distances
            distances = new_df.apply(lambda row: distance.euclidean(row[bands],
                                                                    mean.loc[group]),
                                     axis=1)

        elif dist_func == 'seuclidean':
            # Calc standardized euclidean distances
            distances = new_df.apply(lambda row: distance.seuclidean(row[bands],
                                                                     mean.loc[group],
                                                                     var.loc[group]),
                                     axis=1)

        distances_df[group] = distances

    # get the group, by using argmin
    args = distances_df.to_numpy().argmin(axis=1)
    clusters = list(map(lambda x: distances_df.columns[x], args))
    new_df[new_group] = clusters
