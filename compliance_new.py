import gc
from surfquakecore.project.surf_project import SurfProject
from datetime import datetime, timedelta
import math
from obspy import read
from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
from plot_tools_comtilt import PlotComplianceTilt
import copy
import os

class RemoveComplianceTiltFull:
    def __init__(self, root_path, project_path, output_path, plot_path, save=True):

        """

        :param obs_file_path: The file path of pick observations.
        """
        self.root_path = root_path
        self.sp = SurfProject.load_project(path_to_project_file=project_path)
        self.output_path = output_path
        self.plot_path = plot_path
        self.save=save
    def _apply_cascade(self, noise, tr_e, tr_n, tr_z, tr_h):

        channels = {}
        # First Tilt Noise (between horizontal components)
        # Y' = Y - Tyx*X
        channels["source"] = tr_e.copy()
        channels["response"] = tr_n.copy()
        noise.transfer_function(channels)
        #PlotComplianceTilt.plot_coherence_transfer(noise.transfer_info, channels, save_fig=True,
        #                                           path_save=self.plot_path)
        noise.remove_noise(channels)

        # Z' = Z- Tzx*X

        channels["source"] = tr_e.copy()
        channels["response"] = tr_z.copy()
        noise.transfer_function(channels)
        #PlotComplianceTilt.plot_coherence_transfer(noise.transfer_info, channels, save_fig=True,
        #                                           path_save=self.plot_path)
        noise.remove_noise(channels)

        # Second Tilt Noise (horizontal - Vertical)

        # Z'' = Z' - Tz'y'*Y'
        channels["source"] = noise.Nnew.copy()
        channels["response"] = noise.Znew.copy()

        noise.transfer_function(channels)
        #PlotComplianceTilt.plot_coherence_transfer(noise.transfer_info, channels, save_fig=True,
        #                                           path_save=self.plot_path)
        #noise.plot_transfer_function(channels)
        noise.remove_noise(channels)
        #noise.plot_compare_spectrums(channels)

        # Third Compliance (Hydrophone - Vertical)
        # Z''' = Z'' - Tz''h*H

        channels["source"] = tr_h.copy()
        channels["response"] = noise.Znew.copy()
        Ztilt = noise.Znew.copy()
        noise.transfer_function(channels)
        #noise.plot_transfer_function(channels)
        #PlotComplianceTilt.plot_coherence_transfer(noise.transfer_info, channels, save_fig=True,
        #                                           path_save=self.plot_path)
        noise.remove_noise(channels)
        PlotComplianceTilt.plot_compare_spectrums_full(fs=noise.fs, tr_z=tr_z, tr_z_new =noise.Znew,
                                                       channels=channels, Ztilt=Ztilt, path_save=self.plot_path)

        tr_znew = noise.Znew
        tr_znew.stats.channel = 'SCZ'
        if self.save:
            ID=tr_znew.id +"."+ str(tr_znew.stats.starttime.julday) + "." + str(tr_znew.stats.starttime.year)
            name = os.path.join(self.output_path, ID)
            tr_znew.write(name, format="MSEED")

    def run_remove_noise(self, stations, starttime, target_endtime):

        # Convert the strings to datetime objects
        starttime_dt = datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
        target_endtime_dt = datetime.strptime(target_endtime, "%Y-%m-%d %H:%M:%S")

        # Define the interval to add (e.g., 24 hours)
        interval = timedelta(hours=24)

        interval_count = int((target_endtime_dt - starttime_dt) / interval) + 1
        for i in range(interval_count):
            current_time_dt = starttime_dt + i * interval
            current_time_dt_24 = starttime_dt + (i + 1) * interval
            try:
                for station in stations:
                    sp_z = copy.deepcopy(self.sp)
                    sp_z.filter_project_keys(station=station, channel=".+Z")

                    sp_h = copy.deepcopy(self.sp)
                    sp_h.filter_project_keys(station=station, channel=".+DH")

                    sp_n = copy.deepcopy(self.sp)
                    sp_n.filter_project_keys(station=station, channel=".+1")

                    sp_e = copy.deepcopy(self.sp)
                    sp_e.filter_project_keys(station=station, channel=".+2")


                # Calculate the current datetime

                sp_z.filter_project_time(starttime=current_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                       endtime=current_time_dt_24.strftime("%Y-%m-%d %H:%M:%S"))

                sp_h.filter_project_time(starttime=current_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                       endtime=current_time_dt_24.strftime("%Y-%m-%d %H:%M:%S"))

                sp_n.filter_project_time(starttime=current_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                       endtime=current_time_dt_24.strftime("%Y-%m-%d %H:%M:%S"))

                sp_e.filter_project_time(starttime=current_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                       endtime=current_time_dt_24.strftime("%Y-%m-%d %H:%M:%S"))

                for key in sp_z.project:
                    tr_z = read(sp_z.project[key][0][0])[0]

                for key in sp_h.project:
                    tr_h = read(sp_h.project[key][0][0])[0]

                for key in sp_n.project:
                    tr_n = read(sp_n.project[key][0][0])[0]

                for key in sp_e.project:
                    tr_e = read(sp_e.project[key][0][0])[0]

                noise = RemoveComplianceTilt(N=tr_n, E=tr_e, Z=tr_z, H=tr_h)
                self._apply_cascade(noise, tr_e, tr_n, tr_z, tr_h)

                del sp_z
                del sp_e
                del sp_n
                del sp_h
                gc.collect()
                print("Current time Done", current_time_dt)
            except:
                print("ERROR at ", current_time_dt)


class RemoveComplianceTilt:

    def __init__(self, N, E, Z, H):

        """
        N, E, Z, H -> North, East, Vertical and Hydrophon traces

        :param obs_file_path: The file path of pick observations.
        """

        self.N = N
        self.E = E
        self.Z = Z
        self.H = H
        self.Enew = self.E.copy()
        self.Nnew = self.N.copy()
        self.Znew = self.Z.copy()
        self.fs = self.Z.stats.sampling_rate
        self.fn = self.fs/2
        self.transfer_info = {}

    @staticmethod
    def power_log(x):
        n = math.ceil(math.log(x, 2))
        return n

    @staticmethod
    def find_nearest(a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx], idx

    def transfer_function(self, channels, nfft=15, noverlap=50):

        # source --> X, Y and H
        # response --> Z

        nfft = nfft * self.fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap/100))

        s = channels["source"]
        r = channels["response"]
        channels = [s, r]
        maxstart = np.max([tr.stats.starttime for tr in channels])
        minend = np.min([tr.stats.endtime for tr in channels])

        s.trim(maxstart, minend)
        r.trim(maxstart, minend)
        s.detrend(type='simple')
        r.detrend(type='simple')

        f, Pss = signal.welch(s.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prr = signal.welch(r.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prs = signal.csd(r.data, s.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                         detrend='linear', return_onesided=True, scaling='density', axis=-1)

        coherence = Prs/(np.sqrt(Prr*Pss))
        transfer = np.conj(coherence*np.sqrt(Prr/Pss))

        #self.transfer_info["cpsd"] = Prs
        #self.transfer_info["source_power"] = Pss
        #self.transfer_info["response_power"] = Prr
        self.transfer_info["transfer"] = transfer
        self.transfer_info["coherence"] = coherence
        self.transfer_info["frequency"] = f

    def remove_noise(self, channels):

        #print("Calculating new Trace in Frequency Domain")
        s = channels["source"]
        r = channels["response"]
        s.detrend(type="simple")
        s.taper(type="Hamming", max_percentage=0.05)
        r.detrend(type="simple")
        r.taper(type="Hamming",max_percentage=0.05)

        f = self.transfer_info["frequency"]
        Thr = self.transfer_info["transfer"]

        Sf = np.fft.rfft(s.data, 2 ** self.power_log(len(s.data)))
        Rf = np.fft.rfft(r.data, 2 ** self.power_log(len(r.data)))

        ##Interpolate Thz to Hf

        freq1 = np.fft.rfftfreq(2 ** self.power_log(len(r.data)), 1/self.fs)
        set_interp = interp1d(f, Thr, kind='cubic')
        Thrf = set_interp(freq1)
        # set_interp = interp1d(f, np.abs(Thr), kind='cubic')
        #phase_angle = np.angle(Rf)
        #Rff = (np.abs(Rf) - np.abs(Thrf)*np.abs(Hf))* np.exp(1j * phase_angle)
        Rff = (Rf) - (Thrf * Sf)
        value, idx = self.find_nearest(freq1, 0.1)
        Rff[idx:] = Rf[idx:]
        Rnew_data = np.fft.irfft(Rff)

        if r.stats.channel[2] == "Z":
            self.Znew.data = Rnew_data[0:len(r.data)]
        elif r.stats.channel[2] == "N" or r.stats.channel[2] == "1" or r.stats.channel[2] == "Y":
            self.Nnew.data = Rnew_data[0:len(r.data)]
        elif r.stats.channel[2] == "E" or r.stats.channel[2] == "2" or r.stats.channel[2] == "X":
            self.Enew.data = Rnew_data[0:len(r.data)]

if __name__ == '__main__':
    path_to_data = "/Volumes/LaCie 1/UPFLOW_5HZ/toy/data"
    path_to_project = "/Volumes/LaCie 1/UPFLOW_5HZ/toy/surfquake_project_new.pkl"
    output_denoise = "/Volumes/LaCie 1/UPFLOW_5HZ/toy/output_denoise"
    plot_path = "/Volumes/LaCie 1/UPFLOW_5HZ/toy/plots_denoise"

    rm = RemoveComplianceTiltFull(path_to_data, path_to_project, output_path=output_denoise, plot_path=plot_path)
    rm.run_remove_noise(stations=["UP09"], starttime="2021-07-01 00:00:00",
                        target_endtime="2022-09-01 00:00:00")


    #print(sp)