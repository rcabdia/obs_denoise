
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

class PlotTools:

    @staticmethod
    def plot_coherence_transfer(transfer_info, channels, id_name, **kwargs):
        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        label = "Coherence " + channels["info"]
        phase = np.angle(transfer_info["coherence"])
        coherence = np.abs(transfer_info["coherence"])
        f = transfer_info["frequency"]
        title = ('Coherence station ' + id_name + " " + "channels " + channels["info"])
        fig, axs = plt.subplots(2, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        axs[0].semilogx(transfer_info["frequency"], coherence, linewidth=0.75, color='steelblue', label=label)
        axs[0].grid(True, which="both", ls="-", color='grey')
        axs[0].set_xlim(f[1], f[len(f) - 1])
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].set_xlabel('Frequency [Hz]')
        axs[0].set_ylabel('Coherence')
        axs[0].legend()

        axs[1].semilogx(transfer_info["frequency"], phase, linewidth=0.75, color='red', label=label)
        axs[1].set_xlim(f[1], f[len(f) - 1])
        axs[1].grid(True, which="both", ls="-", color='grey')
        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('Phase')

        if save_fig:
            name = id_name + "_" + "coherence"+"_"+channels["info"]+".pdf"
            path = os.path.join(path_save, name)
            plt.savefig(path, dpi=150, format='pdf')
            plt.close()
        else:
            plt.show()


    @staticmethod
    def plot_transfer_function(transfer_info, channels, **kwargs):

        def find_nearest(a, a0):
            "Element in nd array `a` closest to the scalar value `a0`"
            idx = np.abs(a - a0).argmin()
            return a.flat[idx], idx

        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        s = channels["source"]
        r = channels["response"]
        label = "Transfer " + s.id + "-" + r.id
        transfer = np.abs(transfer_info["transfer"])
        f = transfer_info["frequency"]
        value_min, idx_min = find_nearest(f, 0.001)
        value_max, idx_max = find_nearest(f, 0.1)
        transfer = transfer[idx_min: idx_max]
        f = f[idx_min: idx_max]
        fig, axs = plt.subplots(figsize=(10, 10))
        fig.suptitle('Transfer Function', fontsize=16)
        axs.semilogx(f, transfer, linewidth=0.75, color='steelblue', label=label)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude')
        axs.legend()

        if save_fig:
            name = label + "." + str(s.stats.starttime.julday) + "." + str(s.stats.starttime.year) + ".Transfer" + ".png"
            path = os.path.join(path_save, name)
            plt.savefig(path, dpi=150, format='png')
            plt.close()
        else:

            plt.show()

    @staticmethod
    def plot_compare_spectrums(dirty, clean, fs,  nfft=15, noverlap=50):


        nfft = nfft * fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap / 100))
        f, Zpow = signal.welch(dirty, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap,
                               nfft=nfft, detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Zpownew = signal.welch(clean, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                                  detrend='linear', return_onesided=True, scaling='density', axis=-1)
        ##
        fig, axs = plt.subplots(figsize=(8, 8))
        fig.suptitle('Power Spectrum Comparison', fontsize=16)
        axs.semilogx(f[1:], 10 * np.log(Zpow[1:] / 2 * np.pi * f[1:]), linewidth=0.75, color='steelblue')
        axs.semilogx(f[1:], 10 * np.log(Zpownew[1:] / 2 * np.pi * f[1:]), linewidth=0.75, color='green')

        axs.set_xlim(0.001, 0.1)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude Acceleration [$m^2/s^4/Hz$] [dB]')
        plt.show()


    @staticmethod
    def plot_compare_spectrums_full(fs, dirty, dirty_tilt, dirty_compliance, id_name, nfft=15, noverlap=50, **kwargs):

        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        nfft = nfft * fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap / 100))

        f, Zpow = signal.welch(dirty, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap,
                               nfft=nfft, detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Zpowtilt = signal.welch(dirty_tilt, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap,
                                   nfft=nfft, detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Zpownew = signal.welch(dirty_compliance, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                                  detrend='linear', return_onesided=True, scaling='density', axis=-1)
        ##
        fig, axs = plt.subplots(figsize=(8, 8))
        title = ('Power Spectrum Comparison ' + id_name)
        fig.suptitle(title, fontsize=16)
        axs.semilogx(f[1:], 10 * np.log(Zpow[1:] / 2 * np.pi * f[1:]), linewidth=0.75, color='steelblue', label="dirty")
        axs.semilogx(f[1:], 10 * np.log(Zpowtilt[1:] / 2 * np.pi * f[1:]), linewidth=0.75, color='green',
                     label="- Tilt removed")
        axs.semilogx(f[1:], 10 * np.log(Zpownew[1:] / 2 * np.pi * f[1:]), linewidth=0.75, color='red',
                     label="- Compliance removed")
        axs.set_xlim(0.001, 0.1)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude Acceleration dB [(counts/s^2)^2 / Hz] ')
        axs.legend()

        if save_fig:
            name = id_name + "_" + "spectrums" + ".pdf"
            path = os.path.join(path_save, name)
            plt.savefig(path, dpi=150, format='pdf')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_normalized_components(components_dict, sampling_rate=1.0, title="Normalized Components"):
        """
        Plots all components in a single axis, normalized to their max absolute amplitude.

        Parameters:
        - components_dict: dict with keys like 'Z', 'H', '1', '2' and NumPy arrays as values
        - sampling_rate: samples per second (for x-axis time)
        - title: plot title
        """
        plt.figure(figsize=(10, 5))
        colors = {'Z': 'k', 'H': 'c', '1': 'r', '2': 'b'}

        for key, data in components_dict.items():
            if data.size == 0:
                continue
            #norm_data = data / np.max(np.abs(data))
            t = np.arange(len(data)) / sampling_rate
            plt.plot(t, data, label=key, color=colors.get(key, None))

        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_transfer_function(transfer_info, channels, **kwargs):

        def find_nearest(a, a0):
            "Element in nd array `a` closest to the scalar value `a0`"
            idx = np.abs(a - a0).argmin()
            return a.flat[idx], idx

        s = channels["source"]
        r = channels["response"]
        label = "Transfer Function"
        transfer = np.abs(transfer_info["transfer"])
        f = transfer_info["frequency"]
        value_min, idx_min = find_nearest(f, 0.001)
        value_max, idx_max = find_nearest(f, 0.1)
        transfer = transfer[idx_min: idx_max]
        f = f[idx_min: idx_max]
        fig, axs = plt.subplots(figsize=(10, 10))
        fig.suptitle('Transfer Function', fontsize=16)
        axs.semilogx(f, transfer, linewidth=0.75, color='steelblue', label=label)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude')
        axs.legend()
        plt.show()
