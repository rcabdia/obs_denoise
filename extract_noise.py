#UP02 ? CHZ ? P ? 20220712 1931 06.311 GAU 0.00E+00 0.0 0.00 0.0
from obspy import read, UTCDateTime
import numpy as np
mseed_file = "/Users/roberto/Documents/data_test/test_threshold/DO.d07..hh4.D.2011.356"
tr = read(mseed_file)[0]
tr.plot()
data_signal = tr.data
np.savetxt("test_signal.txt", data_signal)
# start = tr.stats.starttime+5*60
# end = tr.stats.endtime
# pick = UTCDateTime("2022-07-12TT19:31:06.311")
#
# tr_noise = tr.copy()
# tr_signal = tr.copy()
#
#
# tr_noise.trim(starttime=start, endtime=pick)
# tr_signal.trim(starttime=pick, endtime=end)
#
# tr_noise.plot()
# tr_signal.plot()
# data_noise = tr_noise.data
# data_signal = tr.data
# np.savetxt("test_nose.txt",data_noise)
# np.savetxt("test_signal.txt",data_signal)