from __future__ import division, print_function
# coding=utf-8
import streamlit as st
import os
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
import circular as c
import numpy as np
import util, rayleigh_plotting_functions, temperature

### Main program to run streamlit web app to perform circular statistics and plot in polar coordiantes.
### Input: .csv files containing data
### Usage -- In terminal type: streamlit run demo.py




def main():
    os.environ['TZ'] = 'UTC'

    st.markdown("<h1 style='text-align: left; color: lightseagreen;'>Welcome to Circadian!</h1>", unsafe_allow_html=True)
    #st.sidebar.markdown("<h3 style='text-align: left; color: cornflowerblue;'>Best Kicks for your Buck</h3>", unsafe_allow_html=True)

    mode_options = ['Upload phase marker file.','Upload raw traces.','Use dummy body tempearature data.']
    mode = st.sidebar.selectbox("Select Analysis Mode", mode_options[::-1])

    if mode == mode_options[0]:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, index_col= 0)
            st.sidebar.markdown('Loaded file: ' + uploaded_file)
        else:
            df = pd.read_csv('dummy_data.csv', index_col=0)
            st.sidebar.markdown('No file uploaded. Using demo data.')

    elif mode == mode_options[1]:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            raw_data = pd.read_csv(uploaded_file, index_col=0)
            st.sidebar.markdown('Loaded file: ' + uploaded_file)
        else:
            raw_data = pd.read_csv('example_raw_traces.csv', index_col=0)
            st.sidebar.markdown('No file uploaded. Using demo data.')

        df = temperature.get_all_data(raw_data)


    else:
        num_groups = st.slider("Select number of groups", min_value=1, max_value=8, value= 2, step=1)
        traces = {}
        df = pd.DataFrame(columns=['Local Time', 'Exp_Group'])
        for i in range(num_groups):
            default_group_name = 'Group' + str(i+1)
            group_name = st.text_input("Enter group name:", default_group_name)
            if i == 0:
                base_phase_default, phase_range_default = 19.0,2.0
            else:
                base_phase_default, phase_range_default = 13.0, 5.0
            base_phase = st.slider("Select base phase (h) for " + default_group_name + ":", min_value=0.0, max_value=24.0, value= base_phase_default, step=0.2)
            phase_range = st.slider("Select phase range (h) for " + default_group_name + ":", min_value=0.0, max_value=24.0, value= phase_range_default, step=0.2)


            peaks, trace = temperature.get_all_data_sim(base_phase, phase_range, group_name=group_name)
            traces[group_name] = trace[0]
            df = pd.concat([df, peaks])
        for i in traces:
            temperature.get_tmp_plots(traces[i], i)
    show_raw_data = st.checkbox("Show Raw Data")
    if show_raw_data:
        st.write(df)

    phase_col = st.sidebar.selectbox("Select Phase Column", df.columns)
    group_col = st.sidebar.selectbox("Select Group Column", df.columns[::-1])
    schedule = st.sidebar.selectbox("Select Light Schedule", ['Human','ZT','CT'])

    show_sep_plot = st.checkbox("Make plots: plot groups separately", value = True)
    if show_sep_plot:
        rayleigh_plotting_functions.rayleigh_plot_separate(df, phase_col, group_col,schedule=schedule,fname='Test rayleigh plot')

    show_same_plot = st.checkbox("Make plots: plot groups on the same graph")
    if show_same_plot:
        rayleigh_plotting_functions.rayleigh_plot_same(df, phase_col, group_col, schedule= schedule, fname='Test rayleigh plot')

    st.sidebar.title('About the app')
    st.sidebar.markdown(
        """
        Circadian is a web app demo for the circadian package for comparing phases from sinusoidal time series
         data. Either upload your time series or use dummy data, the program will find peaks of each day and average them, determine
         whether peaks from each group are not random, and compare peak phases among groups by plotting on a Rayleigh's plot.\n
        👉 Select your data file and see circadian get to work!
        ### Want to learn more about circadian?
        - Checkout [github](https://github.com/tsofoon/circadian) repo
        - Checkout my [Linkedin](https://www.linkedin.com/in/matttso/) profile\n
        Created by Matt Tso 2020
        """
    )
    # st.sidebar.image(q_product_info['image_link'][0], use_column_width=False)
if __name__ == "__main__":
    main()