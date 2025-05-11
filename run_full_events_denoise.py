import copy
import os
from datetime import timedelta
import pandas as pd
from obspy import read_inventory, UTCDateTime, geodetics
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from surfquakecore.project.surf_project import SurfProject
from processing_tools import ProcessingTools
import logging
# from tqdm import tqdm

logging.basicConfig(
    filename='run_denoise.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RunDenoise:

    def __init__(self, path_to_project, path_catalog, inventory_path, stations_file, output_path):

        """
        :param obs_file_path: The file path of pick observations.

        """

        self.stations_file = stations_file
        self.inventory_file = inventory_path
        self.path_catalog = path_catalog
        self.path_to_project = path_to_project
        self.model = TauPyModel('iasp91')
        self.inventory = read_inventory(inventory_path)
        self.output_path = output_path
        if os.path.isdir(self.output_path):
            pass
        else:
            os.makedirs(self.output_path)

    def _load_catalog(self):
        df = pd.read_csv(self.path_catalog, sep='|', skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        return df
    def _load_stations_file(self):
        df = pd.read_csv(self.stations_file, sep='\s+', header=0)
        df['channel_list'] = df['channels'].str.split(',')
        return df

    def _get_event_info(self, row, sta):
        event_info = {}
        event_info["origin_time"] = UTCDateTime(row['Time'])
        event_info["ev_lat"] = row['Latitude']
        event_info["ev_lon"] = row['Longitude']
        event_info["ev_depth"] = row['Depth/km']
        event_info["magnitude"] = row['Magnitude']

        # Extract station coordinates from inventory
        sta_meta = self.inventory.select(station=sta)
        sta_lat = sta_meta[0][0].latitude
        sta_lon = sta_meta[0][0].longitude

        # Compute distance and arrival time
        dist_m, azimuth, back_azimuth = gps2dist_azimuth(event_info["ev_lat"], event_info["ev_lon"], sta_lat, sta_lon)
        dist_deg = geodetics.kilometer2degrees(dist_m / 1000)

        arrivals = self.model.get_travel_times(source_depth_in_km=event_info["ev_depth"], distance_in_degree=dist_deg)

        event_info["first_arrival"] = UTCDateTime(event_info["origin_time"] + timedelta(seconds=arrivals[0].time))
        t_2 = event_info["origin_time"] + (dist_m / 1000) / 2.5
        t_1 = event_info["origin_time"] + (dist_m / 1000) / 5.0
        event_info["times"] = [t_2, t_1]
        event_info["distance"] = [dist_deg, dist_m / 1000]
        event_info["azimuth"] = azimuth
        event_info["backazimuth"] = back_azimuth

        return event_info


    def _save_results(self, event_info, st, ev_num):
        data = {}
        data["event_info"] = event_info
        data["streams"] = st
        net = st[0].stats.network
        station = st[0].stats.station
        julday = st[0].stats.starttime.julday
        year = st[0].stats.starttime.year
        name = net+"."+station+"."+str(julday)+"."+str(year)+"_"+ev_num+"."+"pkl"
        path = os.path.join(self.output_path, name)
        pd.to_pickle(data, path)


    def loop_over_events(self, cut_time = 3600, min_mag=6.5, trim=True, plot=False):
        sp = SurfProject.load_project(self.path_to_project)
        df = self._load_stations_file()
        dt = 3*3600 # seconds 3 hours for searching files
        out_plots = None
        if plot:
            out_plots = os.path.join(self.output_path, "plots")
            if os.path.isdir(out_plots):
                pass
            else:
                os.makedirs(out_plots)

        # for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Station Loop"):

        for index, row in df.iterrows():

             # This loop over specific station
             sp_process = copy.deepcopy(sp)
             logger.info(f"Station Processing: {row['name']}")
             sta = row['name']
             sp_process.filter_project_keys(station=sta)

             # Load the catalog into a DataFrame
             df_catalog = pd.read_csv(self.path_catalog, sep='|', skipinitialspace=True)
             df_catalog.columns = df_catalog.columns.str.strip()
             df_catalog['Time'] = pd.to_datetime(df_catalog['Time'], errors='coerce')

             ev_num = 1

             for _, row in df_catalog.iterrows():
                try:
                    event_info = self._get_event_info(row, sta)
                    sp_process_time = copy.deepcopy(sp_process)
                    origin_time = row['Time']
                    st_time = UTCDateTime(origin_time)
                    et_time = st_time + dt
                    sp_process_time.filter_project_time(starttime=st_time, endtime=et_time)
                    list_files = sp_process_time.data_files

                    if list_files:
                        logger.info(f"Processing event at {st_time} with files: {list_files}")
                        pt = ProcessingTools(cut_events=True, plot=plot, path_save=out_plots)
                        pt.split_stream(list_files)
                        pt.split_stream_by_events(df_catalog, self.model, self.inventory, cut_time)
                        pt.generate_noise_transfer()
                        st_both = pt.remove_tilt_compliance_event()

                        if trim:
                            st_both.trim(starttime=event_info["first_arrival"] - 20 * 60,
                                         endtime=event_info["first_arrival"] + 1.5 * 3600)
                            self._save_results(event_info, st_both, str(ev_num))
                except Exception as e:
                    logger.error(f"Error processing event {ev_num} at station {sta}: {e}", exc_info=True)

                ev_num += 1


if __name__ == '__main__':
    path_to_project = "/Volumes/LaCie/UPFLOW_5HZ/data/all_upflow.pkl"
    path_catalog = "/Users/roberto/Documents/data_test/events.txt"
    inventory_path = "/Users/roberto/Documents/data_test/meta.xml"
    stations_file = "/Users/roberto/Documents/data_test/stations_channels.txt"
    output = "/Users/roberto/Documents/data_test/output_test"
    rd = RunDenoise(path_to_project, path_catalog, inventory_path, stations_file, output)
    rd.loop_over_events(plot=True)