import os
import pandas as pd
import numpy as np
from obspy import Stream


dir_path = "/Users/roberto/Documents/data_test/output_test/"
files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]

for file in files:
    file_abs = os.path.abspath(os.path.join(dir_path, file))
    print(file)
    try:
        df = pd.read_pickle(file_abs)
        st = df["streams"]
        st.detrend(type="simple")
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=1/80, freqmax=1/20)
        st.detrend(type="simple")
        st.taper(max_percentage=0.05)

        st_time = df["event_info"]["times"][1]
        et_time = df["event_info"]["times"][0]
        t1 = int(5 * 60)
        t2 = int(10 * 60)

        noise_st = st[0].stats.starttime + 300.0
        noise_et = st[0].stats.starttime + 600.0
        print(noise_st, noise_et)
        print(st_time, et_time)
        #st.plot()
        for channel in ["CHZ", "HHZ", "BHZ"]:
            tr_dirty = st.select(channel=channel)[0]
            if len(tr_dirty) > 0:
                tr_clean = st.select(channel="KHZ")[0]
                break


        noise_clean = tr_clean.data[t1:t2]
        noise_dirty = tr_dirty.data[t1:t2]
        tr_clean.trim(starttime=st_time, endtime=et_time)
        tr_dirty.trim(starttime=st_time, endtime=et_time)
        st_noise = Stream([tr_clean, tr_dirty])
        #st_noise.plot()
        RMS_clean = np.mean(np.abs(tr_clean.data)) / np.mean(np.abs(noise_clean))
        RMS_dirty = np.mean(np.abs(tr_dirty.data)) / np.mean(np.abs(noise_dirty))
        print(RMS_dirty, RMS_clean)

    except Exception as e:
        print(e)
        #pass

