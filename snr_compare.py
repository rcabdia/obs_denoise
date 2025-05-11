import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class WaveformQualityEvaluator:
    # TODO WE NEED TO ADD MAGNITUDE
    # TODO WE NEED TO RENAME plot output

    def __init__(self, input_dir, output_pdf, output_txt, noise_window=(300, 600)):
        self.input_dir = input_dir
        self.output_pdf = output_pdf
        self.output_txt = output_txt
        self.noise_window = noise_window
        self.results = []

    def process_all(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.pkl')]
        with PdfPages(self.output_pdf) as pdf:
            for file in files:
                file_path = os.path.join(self.input_dir, file)
                try:
                    self.process_file(file_path, pdf)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
        self.write_summary()

    def process_file(self, file_path, pdf):
        df = pd.read_pickle(file_path)
        st = df["streams"]
        st.detrend(type="simple")
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=1/80, freqmax=1/20)

        st_time = df["event_info"]["times"][1]
        et_time = df["event_info"]["times"][0]
        origin_time = df["event_info"]["origin_time"]
        station_name = st[0].stats.station

        t1 = self.noise_window[0]
        t2 = self.noise_window[1]

        tr_dirty = None
        for ch in ["CHZ", "HHZ", "BHZ"]:
            matches = st.select(channel=ch)
            if len(matches) > 0:
                tr_dirty = matches[0]
                break

        tr_clean = st.select(channel="KHZ")[0] if st.select(channel="KHZ") else None

        if not tr_clean or not tr_dirty:
            raise ValueError("Missing clean or dirty channel")

        noise_clean = tr_clean.data[t1:t2]
        noise_dirty = tr_dirty.data[t1:t2]

        tr_clean_trim = tr_clean.copy().trim(starttime=st_time, endtime=et_time)
        tr_dirty_trim = tr_dirty.copy().trim(starttime=st_time, endtime=et_time)

        RMS_clean = np.mean(np.abs(tr_clean_trim.data)) / np.mean(np.abs(noise_clean))
        RMS_dirty = np.mean(np.abs(tr_dirty_trim.data)) / np.mean(np.abs(noise_dirty))


        self.results.append({
            "station": station_name,
            "origin_time": origin_time,
            "longitude": df["event_info"]["ev_lon"],
            "latitude": df["event_info"]["ev_lat"],
            "depth_km": df["event_info"]["ev_depth"],
            "RMS_dirty": RMS_dirty,
            "RMS_clean": RMS_clean,
            "RMS_gain": RMS_clean - RMS_dirty
        })

        self.plot_comparison(tr_dirty, tr_clean, origin_time, st_time, et_time, os.path.basename(file_path), pdf)

    def plot_comparison(self, tr_dirty, tr_clean, origin_time, st_time, et_time,  title, pdf):
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        times = tr_dirty.times("relative")
        t1 = tr_dirty.stats.starttime-origin_time + self.noise_window[0]
        t2 = tr_dirty.stats.starttime - origin_time + self.noise_window[1]
        st1 = st_time - tr_dirty.stats.starttime
        et2 = et_time - tr_dirty.stats.starttime

        for i, (tr, label) in enumerate(zip([tr_dirty, tr_clean], ["Dirty", "Clean"])):

            if label == "Dirty":

                color="black"
            else:
                color = "steelblue"

            axs[i].plot(times, tr.data, color=color, label=label, linewidth=0.75)
            axs[i].axvspan(t1, t2, color='gray', alpha=0.3, label='Noise Window')
            axs[i].axvspan(st1, et2, color='red', alpha=0.1, label='Signal')
            axs[i].legend()
            axs[i].set_ylabel("Amplitude")

        axs[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Waveform Comparison: {title}")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def write_summary(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_txt, index=False, sep='|')
        print(f"Saved summary to: {self.output_txt}")


if __name__ == "__main__":
    evaluator = WaveformQualityEvaluator(
        input_dir="/Users/roberto/Documents/data_test/output_test/",
        output_pdf="/Users/roberto/Documents/data_test/output_test/SNR/waveform_comparisons.pdf",
        output_txt="/Users/roberto/Documents/data_test/output_test/SNR/waveform_rms_summary.txt",
        noise_window=(350, 850)
    )
    evaluator.process_all()
