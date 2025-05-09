import math
from copy import deepcopy
from obspy import read, Stream, Trace, UTCDateTime, geodetics
from datetime import timedelta
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from scipy.interpolate import interp1d
from scipy import signal

from plot_tools_new import PlotTools


class ProcessingTools:
    def __init__(self, cut_events, plot=False, path_save=None):
        self.stream = Stream()
        self.stream_original = Stream()
        self.transfers = {}
        self.cut_events = cut_events
        self.plot = plot
        self.path_save = path_save


    def _power_log(self, x):
        n = math.ceil(math.log(x, 2))
        return n

    def _find_nearest(self, a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx], idx
    def split_stream(self, list_files):

        for filename in list_files:
            try:
                st = read(filename)
                self.stream += st
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        self.stream.merge()
        self.standardize_horizontal_components()
        self.stream_original = deepcopy(self.stream)
        # TODO CUT THE EXACT DAY

    def standardize_horizontal_components(self):
        """
        Renames horizontal components of a 4-component stream to standard '1' and '2' convention.
        The function modifies the stream in-place.

        - Z stays Z (vertical)
        - H stays H (hydrophone)
        - N/Y → 1 (first horizontal)
        - E/X → 2 (second horizontal)
        """
        # Mapping of known horizontal directions to standard convention
        horizontal_map = {
            "N": "1", "Y": "1", "1": "1",
            "E": "2", "X": "2", "2": "2"
        }

        for tr in self.stream:
            ch = tr.stats.channel
            if len(ch) < 3:
                continue  # skip malformed names

            # Replace the last character if it matches horizontal directions
            comp = ch[-1].upper()
            if comp in horizontal_map:
                new_channel = ch[:-1] + horizontal_map[comp]
                tr.stats.channel = new_channel

        # Get common time range
        starttimes = [tr.stats.starttime for tr in self.stream]
        endtimes = [tr.stats.endtime for tr in self.stream]

        common_start = max(starttimes)
        common_end = min(endtimes)

        if common_start >= common_end:
            raise ValueError("No overlapping time window among components.")

        self.stream.trim(starttime=common_start, endtime=common_end)

    def concatenate_stream(self, stream: Stream) -> dict:

        """
        Groups and concatenates trace data by physical orientation:
        - Vertical ('Z'), Horizontal ('1' or '2'), Hydrophone ('H')

        Returns a dictionary with keys:
            'vertical', 'horizontal', 'hydrophone'
        and NumPy arrays of concatenated data as values.
        """

        groups = {
            'Z': [],
            'H': [],
            '1': [],
            '2': []

        }
        name = (stream[0].stats.network + "." + stream[0].stats.station + "." + str(stream[0].stats.starttime.julday)+ "." +
                str(stream[0].stats.starttime.year))

        fs = stream[0].stats.sampling_rate
        for tr in stream:
            ch = tr.stats.channel.upper()
            if ch.endswith("Z"):
                groups['Z'].append(tr.data)
            elif ch.endswith("1"):
                groups['1'].append(tr.data)
            elif ch.endswith("2"):
                groups['2'].append(tr.data)
            elif ch.endswith("H"):
                groups['H'].append(tr.data)

        # Concatenate each group
        concatenated = {}
        for key, traces in groups.items():
            if traces:
                concatenated[key] = np.concatenate(traces)
            else:
                concatenated[key] = np.array([])  # empty if no match
        concatenated["fs"] = fs
        concatenated["name"] = name
        return concatenated
    def remove_timespans(self, stream: Stream, spans: list, taper_percentage: float = 0.025) -> Stream:

        """
        Removes multiple time spans from a stream, with optional tapering on retained segments.

        Parameters:
        - stream: ObsPy Stream object
        - spans: List of (starttime, endtime) tuples/lists (as UTCDateTime or convertible)
        - taper_percentage: Fraction of trace length to taper at both ends (default 5%)

        Returns:
        - A new Stream object without the specified time spans, tapered at segment edges.
        """

        # Normalize and merge overlapping spans
        spans = [(UTCDateTime(start), UTCDateTime(end)) for start, end in spans]
        spans.sort()

        merged_spans = []
        for start, end in spans:
            if not merged_spans or start > merged_spans[-1][1]:
                merged_spans.append([start, end])
            else:
                merged_spans[-1][1] = max(merged_spans[-1][1], end)

        output = Stream()

        for tr in stream:
            tr_start = tr.stats.starttime
            tr_end = tr.stats.endtime
            current_spans = [[max(tr_start, s), min(tr_end, e)] for s, e in merged_spans if e > tr_start and s < tr_end]

            if not current_spans:
                # Entire trace is retained
                retained = tr.copy()
                retained.detrend(type="simple")
                retained.detrend(type="constant")
                retained.taper(max_percentage=taper_percentage, type="cosine")
                retained.detrend(type="simple")
                retained.detrend(type="constant")
                output += retained
                continue

            segments = []
            current_start = tr_start
            for start, end in current_spans:
                if current_start < start:
                    segments.append((current_start, start))
                current_start = max(current_start, end)
            if current_start < tr_end:
                segments.append((current_start, tr_end))

            for s, e in segments:
                seg = tr.copy().trim(starttime=s, endtime=e, pad=False)
                seg.detrend(type="simple")
                seg.taper(max_percentage=taper_percentage, type="cosine")
                seg.detrend(type="simple")
                output += seg

        output.merge(method=1, fill_value=0)
        return output
    def split_stream_by_events(self, df, model, inventory, cut=1300):


        if self.cut_events:
            # Assume 3 components from 1 station
            station = self.stream[0].stats.station
            network = self.stream[0].stats.network

            # Extract station coordinates from inventory
            net = inventory.select(network=network, station=station)
            coordinates = net.get_coordinates(self.stream[0].id)
            sta_lat = coordinates['latitude']
            sta_lon = coordinates['longitude']


            # Store event windows
            event_windows = []

            for _, row in df.iterrows():
                origin_time = row['Time']
                ev_lat = row['Latitude']
                ev_lon = row['Longitude']
                ev_depth = row['Depth/km']

                # Compute distance and arrival time
                dist_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)
                dist_deg = geodetics.kilometer2degrees(dist_m / 1000)

                arrivals = model.get_travel_times(source_depth_in_km=ev_depth,
                                                  distance_in_degree=dist_deg,
                                                  phase_list=["P", "p", "Pdiff"])

                #arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg)

                if not arrivals:
                    continue

                first_arrival = UTCDateTime(origin_time + timedelta(seconds=arrivals[0].time))
                end_time = first_arrival + cut
                event_windows.append((first_arrival, end_time))

            # Merge overlapping windows
            event_windows.sort()
            merged_windows = []
            for start, end in event_windows:
                if not merged_windows or start > merged_windows[-1][1]:
                    merged_windows.append([start, end])
                else:
                    merged_windows[-1][1] = max(merged_windows[-1][1], end)

            # Cut the original stream into event and noise streams
            event_stream = Stream()

            for start, end in merged_windows:
                event_stream += self.stream.slice(starttime=start, endtime=end)

            noise_stream = self.remove_timespans(self.stream, [[event_stream[0].stats.starttime, event_stream[0].stats.endtime]])
        else:
            noise_stream = self.stream.copy()

        self.noise_stream_dict = self.concatenate_stream(noise_stream)


    def generate_noise_transfer(self):

        # Apply the cascade filter to save the transfer function, later will be apply to the original stream

        # First Tilt Noise (between horizontal components)
        # Y' = Y - Tyx*X, first we clean Y

        channels = {}
        id_name = self.noise_stream_dict["name"]
        channels["source"] = self.noise_stream_dict["2"]
        channels["response"] = self.noise_stream_dict["1"]
        transfer_info_1_2 = self.transfer_function(channels, self.noise_stream_dict["fs"])
        comp1_c12 = self.remove_noise(channels, transfer_info_1_2, self.noise_stream_dict["fs"])

        # # Z' = Z - Tzx*X, second we clean Z from X (left original X)
        #
        channels["source"] = self.noise_stream_dict["2"]
        channels["response"] = self.noise_stream_dict["Z"]
        transfer_info_Z_2 = self.transfer_function(channels, self.noise_stream_dict["fs"])
        compZ_cZ2 = self.remove_noise(channels, transfer_info_Z_2, self.noise_stream_dict["fs"])
        # # Third Tilt Noise (horizontal - Vertical)
        #
        # # Z'' = Z' - Tz'y'*Y'
        channels["source"] = comp1_c12
        channels["response"] = compZ_cZ2
        channels["info"] = "Z-1-2"
        transfer_info_Z_1 = self.transfer_function(channels, self.noise_stream_dict["fs"])

        if self.plot:
            PlotTools.plot_coherence_transfer(transfer_info_Z_1, channels, id_name,
                                              save_fig=self.plot, path_save=self.path_save)
        compZ_cZ21 = self.remove_noise(channels, transfer_info_Z_1, self.noise_stream_dict["fs"])

        #
        # Third Compliance (Hydrophone - Vertical)
        # Z''' = Z'' - Tz''h*H
        channels["source"] = self.noise_stream_dict["H"]
        channels["response"] = compZ_cZ21
        channels["info"] = "Z-1-2-H"

        transfer_info_Z_H = self.transfer_function(channels, self.noise_stream_dict["fs"])

        if self.plot:
            PlotTools.plot_coherence_transfer(transfer_info_Z_H, channels, id_name,
                                              save_fig=self.plot, path_save=self.path_save)

        compZ_cZ21H = self.remove_noise(channels, transfer_info_Z_H, self.noise_stream_dict["fs"])

        if self.plot:
            PlotTools.plot_compare_spectrums_full(self.noise_stream_dict["fs"], self.noise_stream_dict["Z"], compZ_cZ21, compZ_cZ21H, id_name,
                                                  save_fig=self.plot, path_save=self.path_save)

        self.transfers["transfer_info_1_2"] = transfer_info_1_2
        self.transfers["transfer_info_Z_2"] = transfer_info_Z_2
        self.transfers["transfer_info_Z_1"] = transfer_info_Z_1
        self.transfers["transfer_info_Z_H"] = transfer_info_Z_H

    def transfer_function(self, channels, fs, nfft=15, noverlap=50):

        transfer_info = {}
        # source --> X, Y and H
        # response --> Z

        nfft = nfft * fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap/100))

        s = channels["source"]
        r = channels["response"]

        f, Pss = signal.welch(s, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prr = signal.welch(r, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prs = signal.csd(r, s, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                         detrend='linear', return_onesided=True, scaling='density', axis=-1)

        coherence = Prs/(np.sqrt(Prr*Pss))
        transfer = np.conj(coherence*np.sqrt(Prr/Pss))

        transfer_info["cpsd"] = Prs
        transfer_info["source_power"] = Pss
        transfer_info["response_power"] = Prr
        transfer_info["transfer"] = transfer
        transfer_info["coherence"] = coherence
        transfer_info["frequency"] = f

        return transfer_info

    def remove_noise(self, channels, transfer_info, fs):

        s = Trace()
        r = Trace()

        s.data = channels["source"]
        r.data = channels["response"]

        s.detrend(type="simple")
        s.taper(type="Hamming", max_percentage=0.05)
        r.detrend(type="simple")
        r.taper(type="Hamming", max_percentage=0.05)


        f = transfer_info["frequency"]
        Thr = transfer_info["transfer"]

        Sf = np.fft.rfft(s.data, 2 ** self._power_log(len(s)))
        Rf = np.fft.rfft(r.data, 2 ** self._power_log(len(r)))

        ##Interpolate Thz to Hf
        freq1 = np.fft.rfftfreq(2 ** self._power_log(len(r)), 1/fs)
        set_interp = interp1d(f, Thr, kind='cubic')
        Thrf = set_interp(freq1)

        Rff = (Rf) - (Thrf * Sf)
        value, idx = self._find_nearest(freq1, 0.1)
        Rff[idx:] = Rf[idx:]
        Rnew_data = np.fft.irfft(Rff)

        return Rnew_data[0:len(r)]

    def remove_tilt_compliance_event(self):

        channels = {}

        fs = self.stream[0].stats.sampling_rate

        data_z = self.stream.select(component="Z")[0].data
        data_1 = self.stream.select(component="1")[0].data
        data_2 = self.stream.select(component="2")[0].data
        data_h = self.stream.select(component="H")[0].data

        # Apply the cascade filter to save the transfer function, later will be apply to the original stream

        # First Tilt Noise (between horizontal components)
        # Y' = Y - Tyx*X, first we clean Y

        channels["source"] = data_2
        channels["response"] = data_1

        data_1 = self.remove_noise(channels, self.transfers["transfer_info_1_2"], fs)

        # Z' = Z- Tzx*X, second we clean Z from X (left original X)

        channels["source"] = data_2
        channels["response"] = data_z
        data_z = self.remove_noise(channels, self.transfers["transfer_info_Z_2"], fs)

        # Third Tilt Noise (horizontal - Vertical)

        # Z'' = Z' - Tz'y'*Y'
        channels["source"] = data_1
        channels["response"] = data_z
        data_z = self.remove_noise(channels, self.transfers["transfer_info_Z_1"], fs)

        # Third Compliance (Hydrophone - Vertical)
        # Z''' = Z'' - Tz''h*H
        channels["source"] = data_h
        channels["response"] = data_z
        data_z = self.remove_noise(channels, self.transfers["transfer_info_Z_H"], fs)
        tr_z_clean = self.stream.select(component="Z")[0]
        tr_z_clean.data = data_z
        tr_z_clean.stats.channel = "KHZ"
        tr_z_original = self.stream_original.select(component="Z")[0]
        st_both = Stream([tr_z_clean, tr_z_original])

        return st_both

