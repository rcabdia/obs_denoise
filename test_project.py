from obspy import UTCDateTime
from surfquakecore.project.surf_project import SurfProject

path_to_project = "/Volumes/LaCie/UPFLOW_5HZ/data/all_upflow.pkl"

sp = SurfProject.load_project(path_to_project)
sta = "UP01"
st_time = UTCDateTime("2022-08-25TT00:00:00")
et_time = st_time+3600*6
sp_process = sp.copy()
sp_process.filter_project_keys(station=sta)
sp_process_time = sp_process.copy()
sp_process_time.filter_project_time(starttime=st_time, endtime=et_time)
print(sp_process_time)
print(sp_process_time.data_files)