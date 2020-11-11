import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.stats.circstats as cs
import circular
import numpy as np
import math
from scipy import optimize, signal
import random
from scipy.signal import find_peaks
import util
import streamlit as st


# setup layout for plotting
# schedule: ZT vs CT
def rayleigh_plot_setup(ax, schedule):

    # suppress the radial labels
    plt.setp(ax.get_yticklabels(), visible=False)

    # set the circumference labels
    ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
    ax.set_xticklabels(range(24))

    # make the labels go clockwise
    ax.set_theta_direction(-1)

    # set color
    if schedule == 'ZT' or schedule == 'Human':
        day_color = 'yellow'
    elif schedule == 'CT':
        day_color = 'lightgrey'
    else:
        day_color = 'white'

    # place 6 at the top for
    if schedule == 'Human':
        ax.set_theta_offset(np.pi)
        for i in range(90, 270):
            plt.axvspan(i * 2 * np.pi / 360, (i + 1) * 2 * np.pi / 360, alpha=0.1, color=day_color)
        for i in range(-90, 90):
            plt.axvspan(i * 2 * np.pi / 360, (i + 1) * 2 * np.pi / 360, alpha=0.1, color='dimgrey')
    else:
        ax.set_theta_offset(np.pi/2.0)
        for i in range(0,180):
            plt.axvspan(i*2*np.pi/360, (i+1)*2*np.pi/360, alpha=0.1, color= day_color)
        for i in range(180,360):
            plt.axvspan(i*2*np.pi/360, (i+1)*2*np.pi/360, alpha=0.1, color='dimgrey')


# plotting rayleigh plots of multiple data groups on the same plot
# df: dataframe
# phase_in_hrs:  phase in hours
# group: groups to seperate dataset
# fname: file name to save
# schedule: ZT/CT plotting for background
# stacked: stacked points or not
# factor: scaling factor for stacked points
def rayleigh_plot_same(df, phase_in_hrs, group, fname, schedule='ZT', stacked=False, factor=0.1):
    df['phase'] = df[phase_in_hrs] * (2 * np.pi / 24)  # convert phase in hours to phase in radian
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # color list for plotting

    # plot setup
    fig = plt.figure(figsize=(8, 8))
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
                subset = util.descending_occurence(subset, factor)
                tmp = tmp.append(subset)
            plotting = tmp
        else:
            plotting['ones'] = np.ones_like(plotting['phase'])
        # plot data points
        points = ax.scatter(plotting['phase'], plotting['ones'], color=colors[g], label=df[group].unique()[g])
        # plot arrow: length = mean vector of all points in the group
        arr_dir = cs.circmean(plotting['phase'])
        arr_len = circular.circ_r(plotting['phase'])
        kw = dict(arrowstyle="->", color=colors[g], linewidth=5)
        ax.annotate("", xy=(arr_dir, arr_len), xytext=(0, 0),
                    arrowprops=kw)
        # plot lateral SD bar. width = +/- SD, length set at half of arrow length
        sd = np.sqrt(cs.circvar(plotting['phase']))
        ax.plot(np.linspace(arr_dir - sd, arr_dir + sd, 100), np.ones(100) * arr_len * 0.5, color='black',
                linestyle='-')
        # show legend
        plt.legend()
        print('Group: ', df[group].unique()[g], 'Mean Phase: ', arr_dir*24/np.pi, 'Mean Vecor Length: ', arr_len, 'SD: ', sd)

    # put the points on the circumference
    plt.ylim(0, 1.02)

    # plt.legend(handles = legend_list, loc = 'upper center',bbox_to_anchor=(1.3, 0.8))
    plt.title(fname, y=1.1)
    plt.grid('off')
    st.pyplot(fig)
    util.save_fig(fname)



# plotting rayleigh plots of multiple data groups on separate, individual plots
# df: dataframe
# phase_in_hrs: ZT/CT phase in hours
# group: groups to seperate dataset
# fname: file name to save
# schedule: ZT/CT plotting for background
# stacked: stacked points or not
# factor: scaling factor for stacked points
def rayleigh_plot_separate(df, phase_in_hrs, group, fname, schedule='ZT', stacked=False, factor=0.1):
    df['phase'] = df[phase_in_hrs] * (2 * np.pi / 24)  # convert phase in hours to phase in radian
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k','pink']  # color list for plotting
    # figure setup
    n_plot = df[group].nunique()
    fig = plt.figure(figsize=(8, 4*n_plot//2+1))
    # plot individual group
    for g in range(len(df[group].unique())):
        # plot setup
        if n_plot <= 2:
            ax = plt.subplot(1, n_plot, g + 1, polar=True, clip_on=False)
        else:
            ax = plt.subplot(n_plot/2, 2, g + 1, polar=True, clip_on=False)
        rayleigh_plot_setup(ax, schedule)
        legend_list = []
        plotting = df[df[group] == df[group].unique()[g]]
        # introduce jitter if stacked, else all points have radius of 1
        if stacked:
            tmp = pd.DataFrame(columns=plotting.columns)
            for t in plotting['phase'].unique():
                subset = plotting[plotting['phase'] == t]
                subset = util.descending_occurence(subset, factor)
                tmp = tmp.append(subset)
            plotting = tmp
        else:
            plotting['ones'] = np.ones_like(plotting['phase'])
        # reverse R test ring to show 0.05 significance threshold
        R_test = util.reverse_circa_r_test(0.05, len(plotting))
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * R_test, color='orange', linestyle='-')
        # plot points
        points = ax.scatter(plotting['phase'], plotting['ones'], color=colors[g], label=df[group].unique()[g])
        # plot arrow, length = mean vector length of all points in the group
        arr_dir = cs.circmean(plotting['phase'])
        arr_len = circular.circ_r(plotting['phase'])
        kw = dict(arrowstyle="->", color=colors[g], linewidth=3)
        ax.annotate("", xy=(arr_dir, arr_len), xytext=(0, 0),
                    arrowprops=kw)
        # plot sd line. width = +/- SD, length set at half of arrow length
        sd = np.sqrt(cs.circvar(plotting['phase']))
        ax.plot(np.linspace(arr_dir - sd, arr_dir + sd, 100), np.ones(100) * arr_len * 0.5, color='black',
                linestyle='-')

        # put the points on the circumference
        plt.ylim(0, 1.05)
        plt.title(df[group].unique()[g], y=1.1)
        plt.grid('off')

        print('Group: ', df[group].unique()[g], 'Mean Phase: ', arr_dir*24/np.pi, 'Mean Vecor Length: ', arr_len, 'SD: ', sd)
    st.pyplot(fig)
    util.save_fig(fname + '_sep')



