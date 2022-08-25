# CircStat Toolbox
#   Toolbox for circular statistics with Python
#   Adapted from Matlab toolbox
#   Adaptation by Matt TSo 2020 github: tsofoon
# Reference:
#   P. Berens, CircStat: A Matlab Toolbox for Circular Statistics, Journal of Statistical Software,Vol. 31, Issue 10, 2009
#	http://www.jstatsoft.org/v31/i10
#
# Author:
#   Philipp Berens & Marc J. Velasco, 2009

# Descriptive Statistics.
#   circ_mean     - Mean direction of a sample of circular data
#   circ_median   -	Median direction of a sample of circular data
#   circ_r        - Resultant vector length
#   circ_var      - Circular variance
#   circ_std      - Circular standard deviation

#   circ_moment   - Circular p-th moment #   circ_skewness -	Circular skewness
#   circ_kurtosis -	Circular kurtosis
#
# Inferential Statistics.
#  Testing for Circular Uniformity.
#   circ_rtest    - Rayleigh's test for nonuniformity
#   circ_otest    - Hodges-Ajne test (omnibus test) for nonuniformity
#   circ_raotest  - Rao's spacing test for nonuniformity
#   circ_vtest    - V-Test for nonuniformity with known mean direction
#
#  Tests Concerning Mean and Median.
#   circ_confmean - Confidence intervals for mean direction
#   circ_mtest    -	One-sample test for specified mean direction
#   circ_medtest  -	Test for median angle
#   circ_symtest  -	Test for symmetry around median angle
#
#  Paired and Multisample Tests.
#   circ_wwtest   - Two and multi-sample test for equal means; 
#                   one-factor ANOVA
#   circ_hktest   -	Two-factor ANOVA
#   circ_cmtest   - Non-parametric multi-sample test for equal medians
#   circ_ktest    - Test for equal concentration parameter
#   circ_kuipertest - Test for equality of distributions (KS-test)
#
# Measures of Association.
#   circ_corrcc   - Circular-circular correlation coefficient
#   circ_corrcl   -	Circular-linear correlation coefficient
#
# The Von Mises Distribution
#   circ_vmpdf    - Probability density function of the von Mises
#                   distribution
#   circ_vmpar    - Parameter estimation
#   circ_vmrnd    - Random number generation
#
# Others.
#   circ_axial    -	Convert axial data to common scale
#   circ_dist     - Distances around a circle
#   circ_dist2    - Pairwise distances around a circle
#   circ_stats    -	Summary statistics
#   circ_kappa    -	Compute concentration parameter of a VM distribution
#   circ_plot     - Visualization for circular data
#   circ_clust    - Simple clustering
#   circ_rad2ang  - Convert radian to angular values
#   circ_ang2rad  -	Convert angular to radian values
#   circ_samplecdf - Evaluate CDF of a sample
#
# Reference:
#   P. Berens, CircStat: A Matlab Toolbox for Circular Statistics, Journal of Statistical Software,Vol. 31, Issue 10, 2009
#	http://www.jstatsoft.org/v31/i10
#
# Author:
#   Philipp Berens & Marc J. Velasco, 2009

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.stats.circstats as cs


def circ_ang2rad(alpha):
    # alpha = circ_ang2rad(alpha)
    #   converts values in degree to radians
    #
    # Circular Statistics Toolbox for Matlab
    # By Philipp Berens, 2009
    # berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
    alpha = alpha * pi / 180
    return alpha


def circ_axial(alpha, p=1):
    #
    # alpha = circ_axial(alpha, p)
    #   Transforms p-axial data to a common scale.
    #
    #   Input:
    #     alpha	sample of angles in radians
    #     [p		number of modes]
    #
    #   Output:
    #     alpha transformed data
    #
    # PHB 2009
    #
    # References:
    #   Statistical analysis of circular data, N. I. Fisher
    #
    # Circular Statistics Toolbox for Matlab
    # By Philipp Berens, 2009
    # berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
    alpha = (alpha * p) % (2 * np.pi)
    return alpha


def circ_r(alpha, w=None, d=0, axis=0):
    """Computes mean resultant vector length for circular data.

    Args:
        alpha: array
            Sample of angles in radians

    Kargs:
        w: array, optional, [def: None]
            Number of incidences in case of binned angle data

        d: radians, optional, [def: 0]
            Spacing of bin centers for binned data, if supplied
            correction factor is used to correct for bias in
            estimation of r

        axis: int, optional, [def: 0]
            Compute along this dimension

    Return:
        r: mean resultant length

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    #     alpha = np.array(alpha)
    #     if alpha.ndim == 1:
    #         alpha = np.matrix(alpha)
    #         if alpha.shape[0] is not 1:
    #             alpha = alpha

    if w is None:
        w = np.ones(alpha.shape)
    elif (alpha.size is not w.size):
        raise ValueError("Input dimensions do not match")

    # Compute weighted sum of cos and sin of angles:
    r = np.multiply(w, np.exp(1j * alpha)).sum(axis=axis)

    # Obtain length:
    r = np.abs(r) / w.sum(axis=axis)

    # For data with known spacing, apply correction factor to
    # correct for bias in the estimation of r
    if d is not 0:
        c = d / 2 / np.sin(d / 2)
        r = c * r

    return np.array(r)


def reverse_circa_r_test(pval, n):
    """

    :type n: int
    """
    R = 4 * n ** 2 - (((1 + 2 * n) + np.log(pval)) ** 2 - (1 + 4 * n))
    R = np.sqrt(R / 4)
    r = R / n

    return r


def circ_rtest(alpha, w=None, d=0):
    """Computes Rayleigh test for non-uniformity of circular data.
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    Args:
        alpha: array
            Sample of angles in radians

    Kargs:
        w: array, optional, [def: None]
            Number of incidences in case of binned angle data

        d: radians, optional, [def: 0]
            Spacing of bin centers for binned data, if supplied
            correction factor is used to correct for bias in
            estimation of r

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    alpha = np.array(alpha)
    if alpha.ndim == 1:
        alpha = np.matrix(alpha)
    if alpha.shape[1] > alpha.shape[0]:
        alpha = alpha.T

    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        if len(alpha) is not len(w):
            raise ValueError("Input dimensions do not match")
        r = circ_r(alpha, w, d)
        n = w.sum()
    print(r)
    # Compute Rayleigh's
    R = n * r

    # Compute Rayleigh's
    z = (R ** 2) / n

    # Compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    return pval, z

# save figure with given figure name
def save_fig(fig_id, tight_layout=True):
    os.makedirs('plots', exist_ok=True)
    path = os.path.join('plots', fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# return vector length r with given pval and number of observations n
def reverse_circa_r_test(pval, n):
    R = 4 * n ** 2 - (((1 + 2 * n) + np.log(pval)) ** 2 - (1 + 4 * n))
    R = np.sqrt(R / 4)
    r = R / n
    return r

# introduce jitter for stacked points presentation
# default factor 0.1 means points goes from 1,0.9,0.8...
def descending_occurence(df, factor=0.1):
    df['range'] = list(range(len(df)))
    df['ones'] = np.ones_like(df['range']) - factor * df['range']
    del df['range']
    return df

# setup layout for plotting
# schedule: ZT vs CT
def rayleigh_plot_setup(ax, schedule):

    # suppress the radial labels
    plt.setp(ax.get_yticklabels(), visible=False)

    # set the circumference labels
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels(range(24))

    # make the labels go clockwise
    ax.set_theta_direction(-1)

    # place 0 at the top
    ax.set_theta_offset(np.pi / 2.0)
    for i in range(180):
        if schedule == 'ZT':
            day_color = 'yellow'
        elif schedule == 'CT':
            day_color = 'lightgrey'
        else:
            day_color = 'white'

        plt.axvspan(i * 2 * np.pi / 360, (i + 1) * 2 * np.pi / 360, alpha=0.1, color=day_color)
    if schedule == 'ZT' or schedule == 'CT':
        for i in range(180, 360):
            plt.axvspan(i * 2 * np.pi / 360, (i + 1) * 2 * np.pi / 360, alpha=0.1, color='dimgrey')


# plotting rayleigh plots of multiple data groups on the same plot
# df: dataframe
# phase_in_ZT: ZT/CT phase in hours
# group: groups to seperate dataset
# fname: file name to save
# schedule: ZT/CT plotting for background
# stacked: stacked points or not
# factor: scaling factor for stacked points
def rayleigh_plot_same(df, phase_in_ZT, group, fname, schedule='ZT', stacked=False, factor=0.1, points = True):
    df['phase'] = df[phase_in_ZT] * (2 * np.pi / 24)  # convert phase in hours to phase in radian
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # color list for plotting

    # plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    rayleigh_plot_setup(ax, schedule)
    legend_list = []

    # making plot for each group
    for g in range(len(df[group].unique())):
        plotting = df[df[group] == df[group].unique()[g]]
        # introduce jitter if stacked, otherwise all points have radius of 1
        if stacked:
            tmp = pd.DataFrame(columns=plotting.columns)
            for t in plotting['phase'].unique():
                subset = plotting[plotting['phase'] == t]
                subset = descending_occurence(subset, factor)
                tmp = tmp.append(subset)
            plotting = tmp
        else:
            plotting['ones'] = np.ones_like(plotting['phase'])
        # plot data points
        if points:
            points = ax.scatter(plotting['phase'], plotting['ones'], color=colors[g], label=df[group].unique()[g])
        # plot arrow: length = mean vector of all points in the group
        arr_dir = cs.circmean(plotting['phase'])
        arr_len = circ_r(plotting['phase'])
        kw = dict(arrowstyle="->", color=colors[g], linewidth=5)
        ax.annotate("", xy=(arr_dir, arr_len), xytext=(0, 0),
                    arrowprops=kw)
        # plot lateral SD bar. width = +/- SD, length set at half of arrow length
        sd = np.sqrt(cs.circvar(plotting['phase']))
        ax.plot(np.linspace(arr_dir - sd, arr_dir + sd, 100), np.ones(100) * arr_len * 0.5, color='black',
                linestyle='-')
        # show legend
        plt.legend()

    # put the points on the circumference
    plt.ylim(0, 1.02)

    # plt.legend(handles = legend_list, loc = 'upper center',bbox_to_anchor=(1.3, 0.8))
    plt.title(fname, y=1.1)
    plt.grid('off')

    save_fig(fname)
    plt.show()


# plotting rayleigh plots of multiple data groups on separate, individual plots
# df: dataframe
# phase_in_ZT: ZT/CT phase in hours
# group: groups to seperate dataset
# fname: file name to save
# schedule: ZT/CT plotting for background
# stacked: stacked points or not
# factor: scaling factor for stacked points
def rayleigh_plot_separate(df, phase_in_ZT, group, fname, schedule='ZT', stacked=False, factor=0.1):
    df['phase'] = df[phase_in_ZT] * (2 * np.pi / 24)  # convert phase in hours to phase in radian
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # color list for plotting
    # figure setup
    n_plot = df[group].nunique()
    plt.figure(figsize=(8, 4))
    # plot individual group
    for g in range(len(df[group].unique())):
        # plot setup
        ax = plt.subplot(1, n_plot, g + 1, polar=True, clip_on=False)
        rayleigh_plot_setup(ax, schedule)
        legend_list = []
        plotting = df[df[group] == df[group].unique()[g]]
        # introduce jitter if stacked, else all points have radius of 1
        if stacked:
            tmp = pd.DataFrame(columns=plotting.columns)
            for t in plotting['phase'].unique():
                subset = plotting[plotting['phase'] == t]
                subset = descending_occurence(subset, factor)
                tmp = tmp.append(subset)
            plotting = tmp
        else:
            plotting['ones'] = np.ones_like(plotting['phase'])
        # reverse R test ring to show 0.05 significance threshold
        R_test = reverse_circa_r_test(0.05, len(plotting))
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * R_test, color='orange', linestyle='-')
        # plot points
        points = ax.scatter(plotting['phase'], plotting['ones'], color=colors[g], label=df[group].unique()[g])
        # plot arrow, length = mean vector length of all points in the group
        arr_dir = cs.circmean(plotting['phase'])
        arr_len = circ_r(plotting['phase'])
        kw = dict(arrowstyle="->", color=colors[g], linewidth=3)
        ax.annotate("", xy=(arr_dir, arr_len), xytext=(0, 0),
                    arrowprops=kw)
        # plot sd line. width = +/- SD, length set at half of arrow length
        sd = np.sqrt(cs.circvar(plotting['phase']))
        ax.plot(np.linspace(arr_dir - sd, arr_dir + sd, 100), np.ones(100) * arr_len * 0.5, color='black',
                linestyle='-')

        # put the points on the circumference
        plt.ylim(0, 1.05)
        plt.title(fname + '_' + df[group].unique()[g], y=1.1)
        plt.grid('off')

    save_fig(fname + '_sep')
    plt.show()
