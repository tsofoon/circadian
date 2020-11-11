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

# Global parameters of dummy body temperature data

mean_temp = 36
days = 7
n = 24 * 60 * days
amp = 0.2
x_data = np.linspace(0, days * 24, num=n)


# get_temp
# Input: base_phase - average phase of the dummy data
#        phase_range - range of phase
# Return: y_data - the dummy data
# dummy data include randomness in both x and y directions.
def get_temp(base_phase=19, phase_range=2):
    np.random.seed(0)
    phase = (random.random() - 0.5) * phase_range + base_phase
    y_data = mean_temp + amp * np.cos(2 * math.pi * (x_data - phase) / 24) + np.random.normal(size=n) / 50
    return y_data


# test_func - return cosine curve to be fitted
def test_func(x, a, b, c):
    return mean_temp + a * np.cos(c + b * x * 2 * math.pi / 24)


# get_smooth - return smoothened data (y_data_smooth) from raw_data (y_data)
def get_smooth(y_data):
    params, params_covariance = optimize.curve_fit(test_func, x_data, y_data)
    y_data_smooth = test_func(x_data, params[0], params[1], params[2])
    return y_data_smooth


# return all peaks found by numpy's find peaks
def get_peaks(y_data_smooth):
    return x_data[find_peaks(y_data_smooth, height=0)[0]]


# from list of peaks from multiple days of recording return the average peak
def get_avg_peaks(peaks):
    avg_peak = (peaks - [24 * i for i in range(len(peaks))]).mean()
    return avg_peak

# plotting function to plot the raw body temperature data, the smoothened data and the first peak found
def get_tmp_plots(y_data, group_name,t = x_data):
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(t, y_data, label='Raw Data',s=0.5, alpha = 0.5)
    y_data_smooth = get_smooth(y_data)
    plt.plot(t , y_data_smooth,
             label='Fitted function',color = 'red')
    peaks = get_peaks(y_data_smooth)
    plt.axvline(peaks[0], label = 'Peak', color = 'green')
    plt.legend(loc='best')
    plt.xticks([24*i for i in range(days)])
    plt.xlabel('Time (h)')
    plt.ylabel('Body Temperature (deg C)')
    plt.title('Example Daily fluctuation of \n Body Temperature: ' + group_name)
    st.pyplot(fig)
    plt.show()


# A master function to generate dummy temperature data and return their peaks
def get_all_data_sim(base_phase, phase_range, group_name, num_samples=20):
    y_raw = ([get_temp(base_phase, phase_range) for _ in range(num_samples)])
    y_smooth = [get_smooth(y_raw[i]) for i in range(len(y_raw))]
    y_peaks = [get_peaks(y_smooth[i]) for i in range(len(y_smooth))]
    y_avg_peaks = [get_avg_peaks(y_peaks[i]) for i in range(len(y_peaks))]

    peaks = pd.DataFrame({'Local Time': y_avg_peaks, 'Exp_Group': [group_name] * num_samples})
    return peaks, y_raw

def get_all_data(raw_data):
    t = raw_data.index
    col_list = list(raw_data.columns.str.split('_'))
    col_dict = {}

    for i in range(len(col_list)):
        if col_list[i][0] not in col_dict:
            col_dict[col_list[i][0]] = col_list[i][0]

    peaks = pd.DataFrame(columns=['Local Time','Exp_Group'])
    for group_name in col_dict:

        y_raw = raw_data[raw_data.columns[raw_data.columns.str.contains(group_name)]]
        get_tmp_plots(y_raw[y_raw.columns[0]], group_name)
        num_samples = len(y_raw.columns)
        y_smooth = [get_smooth(y_raw[i]) for i in y_raw.columns]
        y_peaks = [get_peaks(y_smooth[i]) for i in range(len(y_smooth))]
        y_avg_peaks = [get_avg_peaks(y_peaks[i]) for i in range(len(y_peaks))]

        peak = pd.DataFrame({'Local Time': y_avg_peaks, 'Exp_Group': [group_name] * num_samples})
        peaks = pd.concat([peaks,peak])

    return peaks.reset_index().drop(columns='index')