import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WaveformQualityEvaluator:

    def __init__(self, input_dir, output_pdf, output_txt, noise_window=(300, 600)):
        self.input_dir = input_dir
        self.output_pdf = output_pdf
        self.output_txt = output_txt
        self.noise_window = noise_window
        self.results = []

    def process_all(self, plot=False):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.pkl')]
        for file in files:
            file_path = os.path.join(self.input_dir, file)
            try:
                self.process_file(file_path, plot=plot)
            except Exception as e:
                print(f"Failed to process {file}: {e}")
        self.write_summary()

    def process_file(self, file_path, plot):
        df = pd.read_pickle(file_path)
        st = df["streams"]
        st.detrend(type="simple")
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=1/80, freqmax=1/40)

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
            "magnitude": df["event_info"]["magnitude"],
            "distance": df["event_info"]["distance"][1],
            "RMS_gain": RMS_clean - RMS_dirty
        })
        if plot:
            self.plot_comparison(tr_dirty, tr_clean, origin_time, st_time, et_time, file_path, df, RMS_clean - RMS_dirty)

    def plot_comparison(self, tr_dirty, tr_clean, origin_time, st_time, et_time, file_path, df, RMS_gain):

        root_name = os.path.basename(file_path)
        name = root_name[0:-4] + "." + "pdf"
        output = os.path.join(self.output_pdf, name)
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Waveform Comparison: {root_name[0:-4]}")
        times = tr_dirty.times("relative")
        t1 = self.noise_window[0]
        t2 = self.noise_window[1]
        st1 = st_time - tr_dirty.stats.starttime
        et2 = et_time - tr_dirty.stats.starttime
        magnitude = df["event_info"]["magnitude"]
        distance = df["event_info"]["distance"][1]
        for i, (tr, label) in enumerate(zip([tr_dirty, tr_clean], ["Dirty", "Clean"])):

            if label == "Dirty":

                color = "black"
            else:
                color = "steelblue"

            axs[i].plot(times, tr.data, color=color, label=label, linewidth=0.75)
            axs[i].axvspan(t1, t2, color='gray', alpha=0.3, label='Noise Window')
            axs[i].axvspan(st1, et2, color='red', alpha=0.1, label='Signal')
            axs[i].legend()
            axs[i].set_ylabel("Amplitude")

        axs[-1].set_xlabel("Time (s)")
        # Create the text content
        info_text = f"Magnitude: {magnitude:.1f}\nDistance: {distance:.1f} km\nSNR: {RMS_gain:.1f}"

        # Add a text box to the figure (top-right corner)
        fig.text(0.12, 0.82, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
        plt.tight_layout()
        #plt.show()
        plt.savefig(output)
        plt.close(fig)

    def write_summary(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_txt, index=False, sep='|')
        print(f"Saved summary to: {self.output_txt}")


if __name__ == "__main__":
    evaluator = WaveformQualityEvaluator(
        input_dir="/Volumes/LaCie/UPFLOW_denoise/new_stuff/output",
        output_pdf="/Volumes/LaCie/UPFLOW_denoise/new_stuff/SNR/waveforms/",
        output_txt="/Volumes/LaCie/UPFLOW_denoise/new_stuff/SNR/waveform_rms_summary.txt",
        noise_window=(350, 850)
    )
    evaluator.process_all(plot=True)
