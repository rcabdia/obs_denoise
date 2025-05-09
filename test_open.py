import os
import pandas as pd
import numpy as np
from obspy import Stream
#
# # test
output = "/Users/roberto/Documents/data_test/output_test/8J.UP02.193.2022_2.pkl"
# #
df = pd.read_pickle(output)
st = df["streams"]
st.filter(type="bandpass", freqmin=1/80, freqmax=1/20)
st.plot()
# #####



