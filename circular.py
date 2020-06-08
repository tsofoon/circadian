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

import numpy as np


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
