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

# save figure with given figure name
def save_fig(fig_id, tight_layout=True):
    os.makedirs('plots', exist_ok = True)
    path = os.path.join('plots',fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# return vector length r with given pval and number of observations n
def reverse_circa_r_test(pval,n):
    R = 4*n**2-(((1+2*n)+np.log(pval))**2-(1+4*n))
    R = np.sqrt(R/4)
    r = R/n
    return r

# introduce jitter for stacked points presentation
# default factor 0.1 means points goes from 1,0.9,0.8...
def descending_occurence(df, factor = 0.1):
    df['range'] = list(range(len(df)))
    df['ones'] = np.ones_like(df['range']) - factor*df['range']
    del df['range']
    return df