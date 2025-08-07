import os
import json
import pickle
import warnings
import datetime
import subprocess
import threading
import time
import shutil

import numpy as np
import obspy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from gamma.utils import association
from pyproj import Proj
from tqdm import tqdm

warnings.filterwarnings("ignore")
matplotlib.use("agg")

class SmartWorkflow:
    """
    SeismicWorkFLow :: Tools Event Detection berbasis metode AI (PhaseNet, Gamma) serta Relokasi Event dengan HypoDD.
    Project :: SMART - IDRIP BMKG @ 2025
    Re-Code :: hakimbmkg 
    
    === buy me coffe ===
    
    ** WAJIB **  ** WAJIB **  ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB **
    ** Untuk menggunakan code ini wajib mereferensi :
    ** 1. Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method." arXiv preprint arXiv:1803.03211 (2018)
    ** 2. Zhu, W., McBrearty, I. W., Mousavi, S. M., Ellsworth, W. L., & Beroza, G. C. (2022). Earthquake phase association using a Bayesian Gaussian Mixture Model. Journal of Geophysical Research: Solid Earth, 127, e2021JB023249. https://doi.org/10.1029/2021JB023249
    ** 3. Waldhauser, F., dan Ellsworth, W.L., 2000, A Double-difference Earthquake Location Algorithm: Method and Application to the Northern Hayward Fault, California,Bulletin of Seismological Society of America, Vol. 90, 1353-1368
    ** 4. Waldhauser, F., 2001, hypoDD-Aprogram to Compute Double-Difference Hypocenter Locations, U.S. Geol. Survey
    ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** ** WAJIB ** 
    """
    def __init__(self, base_dir, region_name, phasenet_model_path, hypodd_exec_path, work_dir_name='work'):
        self.base_dir = base_dir
        self.region_name = region_name
        self.phasenet_model_path = phasenet_model_path
        self.hypodd_exec_path = hypodd_exec_path
        self.work_dir_base = work_dir_name
        self.hypodd_root_path = os.path.join(self.base_dir, "core", "HYPODD")

        self._setup_hypodd_environment() # Memeriksa dan menyiapkan HypoDD
        self._setup_paths_and_directories()
        self._prepare_hypodd_executables()
        
        print(f"Workflow diinisialisasi untuk wilayah: {self.region_name}")
        print(f"Direktori kerja dan output: {os.path.abspath(self.root_dir)}")

    def _setup_hypodd_environment(self):
        """
        Memeriksa, Download, dan mengompilasi HypoDD jika belum ada,
        """
        print("Memeriksa instalasi HypoDD...")
        hypodd_binary = os.path.join(self.hypodd_exec_path, "hypoDD", "hypoDD")
        if os.path.exists(hypodd_binary):
            print("HypoDD sudah terkompilasi.")
            return

        print("HypoDD tidak ditemukan. Memulai proses penyiapan otomatis...")
        
        if not all(shutil.which(cmd) for cmd in ["wget", "tar", "gfortran", "make"]):
            raise EnvironmentError("Ketergantungan tidak terpenuhi. Pastikan 'wget', 'tar', 'gfortran', dan 'make' terpasang.")

        core_dir = os.path.join(self.base_dir, "core")
        os.makedirs(core_dir, exist_ok=True)
        
        hypodd_tar_path = os.path.join(core_dir, "HYPODD_1.3.tar.gz")
        hypodd_url = "http://www.ldeo.columbia.edu/~felixw/HYPODD/HYPODD_1.3.tar.gz"

        try:
            if not os.path.exists(self.hypodd_root_path):
                print(f"Download HypoDD dari {hypodd_url}...")
                subprocess.run(f"wget -O {hypodd_tar_path} {hypodd_url}", shell=True, check=True, cwd=core_dir)
                
                print(f"Mengekstrak {hypodd_tar_path}...")
                subprocess.run(f"tar -xf {hypodd_tar_path}", shell=True, check=True, cwd=core_dir)

            print("Mengompilasi HypoDD...")
            gfortran_path = shutil.which("gfortran")
            
            subprocess.run(f"ln -sf {gfortran_path} f77", shell=True, check=True, cwd=self.hypodd_root_path)
            subprocess.run(f"ln -sf {gfortran_path} g77", shell=True, check=True, cwd=self.hypodd_root_path)
            
            make_path = os.path.join(self.hypodd_root_path, "src")
            subprocess.run("make", shell=True, check=True, cwd=make_path)
            
            print("Kompilasi HypoDD berhasil.")

        except subprocess.CalledProcessError as e:
            print(f"Terjadi error saat menyiapkan HypoDD: {e}")
            raise

    def _setup_paths_and_directories(self):
        """Membuat struktur direktori kerja."""
        self.root_dir = os.path.join(self.work_dir_base, self.region_name)
        self.waveform_dir = os.path.join(self.root_dir, "waveforms")
        self.phasenet_dir = os.path.join(self.root_dir, "phasenet")
        self.hypodd_dir = os.path.join(self.root_dir, "hypodd")
        self.hypoinv_dir = os.path.join(self.root_dir, "hypoinv")

        for path in [self.root_dir, self.waveform_dir, self.phasenet_dir, self.hypodd_dir, self.hypoinv_dir]:
            os.makedirs(path, exist_ok=True)

        self.config_path = os.path.join(self.root_dir, "config.json")
        self.datetimes_path = os.path.join(self.root_dir, "datetimes.json")
        self.index_path = os.path.join(self.root_dir, "index.json")
        self.catalog_path = os.path.join(self.root_dir, "standard_catalog.csv")
        self.stations_json_path = os.path.join(self.root_dir, "stations.json")
        self.stations_pkl_path = os.path.join(self.root_dir, "stations.pkl")
        self.fname_csv_path = os.path.join(self.root_dir, "fname.csv")
        self.phasenet_picks_path = os.path.join(self.phasenet_dir, "picks.csv")
        self.gamma_catalog_path = os.path.join(self.root_dir, "gamma_catalog.csv")
        self.gamma_picks_path = os.path.join(self.root_dir, "gamma_picks.csv")
        self.hypodd_station_path = os.path.join(self.hypodd_dir, "station.dat")
        self.hypodd_phase_path = os.path.join(self.hypodd_dir, "phase.pha")
        
        self.event_plot_path = os.path.join(self.root_dir, "event_locations.png")
        self.station_plot_path = os.path.join(self.root_dir, "station_locations.png")
        
    def _prepare_hypodd_executables(self):
        """Menyalin file eksekusi HypoDD ke direktori kerja."""
        for exe in ["ph2dt", "hypoDD"]:
            source_path = os.path.join(self.hypodd_exec_path, exe, exe)
            if not os.path.exists(source_path):
                 raise FileNotFoundError(f"File eksekusi '{exe}' tidak ditemukan di path yang diharapkan: {source_path}")
            
            target_path = os.path.join(self.hypodd_dir, exe)
            command = f"cp {source_path} {target_path}"
            subprocess.run(command, shell=True, check=True)
            print(f"Menyalin {exe} ke {self.hypodd_dir}")
            
    def set_config(self, center, h_deg, v_deg, starttime_str, endtime_str, networks, channels, client="IRIS", user=None, password=None):
        """Membuat dan menyimpan file konfigurasi utama."""
        print(f"Membuat file konfigurasi dengan FDSN client: {client}")
        degree2km = np.pi * 6371 / 180
        starttime = obspy.UTCDateTime(starttime_str)
        endtime = obspy.UTCDateTime(endtime_str)
        
        config = {
            "region": self.region_name, "center": center,
            "xlim_degree": [center[0] - h_deg / 2, center[0] + h_deg / 2],
            "ylim_degree": [center[1] - v_deg / 2, center[1] + v_deg / 2],
            "degree2km": degree2km,
            "starttime": starttime.datetime.isoformat(timespec="milliseconds"),
            "endtime": endtime.datetime.isoformat(timespec="milliseconds"),
            "networks": networks, "channels": channels, "client": client,
            "user": user, "password": password,
            "phasenet": {}, "gamma": {}, "hypodd": {"MAXEVENT": 1e4}
        }
        
        with open(self.config_path, "w") as fp: json.dump(config, fp, indent=4)
        print(f"Config berhasil disimpan di: {self.config_path}")

        one_hour = datetime.timedelta(hours=1)
        starttimes_list = []
        tmp_start = starttime
        while tmp_start < endtime:
            starttimes_list.append(tmp_start.datetime.isoformat(timespec="milliseconds"))
            tmp_start += one_hour
            
        with open(self.datetimes_path, "w") as fp: json.dump({"starttimes": starttimes_list, "interval": one_hour.total_seconds()}, fp, indent=2)
        print(f"Daftar waktu berhasil disimpan di: {self.datetimes_path}")
        
        with open(self.index_path, "w") as fp: json.dump([list(range(len(starttimes_list)))], fp, indent=2)

    def _get_fdsn_client(self, config):
        """Membuat instance FDSN client dengan atau tanpa kredensial."""
        if config.get("user") and config.get("password"):
            print(f"Menggunakan FDSN client '{config['client']}' dengan user '{config['user']}'.")
            return Client(config["client"], user=config["user"], password=config["password"])
        else:
            print(f"Menggunakan FDSN client '{config['client']}' tanpa autentikasi.")
            return Client(config["client"])
            
    def download_event_catalog(self, plot=False, save_plot=True):
        """Download katalog gempa standar dari FDSN client."""
        print("Download katalog gempa...")
        with open(self.config_path, "r") as fp: config = json.load(fp)
        try:
            client = self._get_fdsn_client(config)
            events = client.get_events(starttime=config["starttime"], endtime=config["endtime"], minlongitude=config["xlim_degree"][0], maxlongitude=config["xlim_degree"][1], minlatitude=config["ylim_degree"][0], maxlatitude=config["ylim_degree"][1])
        except Exception:
            print(f"Gagal menggunakan klien '{config['client']}', mencoba 'IRIS' sebagai fallback...")
            client = Client("iris")
            events = client.get_events(starttime=config["starttime"], endtime=config["endtime"], minlongitude=config["xlim_degree"][0], maxlongitude=config["xlim_degree"][1], minlatitude=config["ylim_degree"][0], maxlatitude=config["ylim_degree"][1])
        print(f"Jumlah gempa yang ditemukan: {len(events)}")
        catalog = {"time": [], "magnitude": [], "longitude": [], "latitude": [], "depth(m)": []}
        for event in events:
            if event.magnitudes:
                catalog["time"].append(event.origins[0].time.datetime); catalog["magnitude"].append(event.magnitudes[0].mag); catalog["longitude"].append(event.origins[0].longitude); catalog["latitude"].append(event.origins[0].latitude); catalog["depth(m)"].append(event.origins[0].depth)
        catalog_df = pd.DataFrame.from_dict(catalog).sort_values(["time"])
        catalog_df.to_csv(self.catalog_path, index=False, float_format="%.3f", date_format="%Y-%m-%dT%H:%M:%S.%f")
        print(f"Katalog berhasil disimpan di: {self.catalog_path}")
        if plot or save_plot:
            plt.figure(figsize=(10, 8)); plt.plot(catalog_df["longitude"], catalog_df["latitude"], "o", markersize=5); plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.axis("scaled"); plt.xlim(config["xlim_degree"]); plt.ylim(config["ylim_degree"]); plt.title(f"Lokasi Gempa ({len(events)} events)")
            if save_plot:
                plt.savefig(self.event_plot_path); print(f"Plot lokasi gempa disimpan di: {self.event_plot_path}")
            if plot: plt.show()
            plt.close()

    def download_station_metadata(self, plot=False, save_plot=True):
        """Download metadata stasiun dari FDSN client."""
        print("Download metadata stasiun...")
        with open(self.config_path, "r") as fp: config = json.load(fp)
        client = self._get_fdsn_client(config)
        stations = client.get_stations(network=",".join(config["networks"]), station="*", starttime=config["starttime"], endtime=config["endtime"], minlongitude=config["xlim_degree"][0], maxlongitude=config["xlim_degree"][1], minlatitude=config["ylim_degree"][0], maxlatitude=config["ylim_degree"][1], channel=config["channels"], level="response")
        print(f"Jumlah stasiun yang ditemukan: {sum(len(net) for net in stations)}")
        station_locs = {}
        for network in stations:
            for station in network:
                for chn in station:
                    sid = f"{network.code}.{station.code}.{chn.location_code}.{chn.code[:-1]}"
                    if sid in station_locs:
                        if chn.code[-1] not in station_locs[sid]["component"]:
                            station_locs[sid]["component"].append(chn.code[-1])
                            station_locs[sid]["response"].append(round(chn.response.instrument_sensitivity.value, 2))
                    else:
                        station_locs[sid] = {
                            "longitude": chn.longitude,
                            "latitude": chn.latitude,
                            "elevation(m)": chn.elevation,
                            "component": [chn.code[-1]],
                            "response": [round(chn.response.instrument_sensitivity.value, 2)],
                            "unit": chn.response.instrument_sensitivity.input_units.lower(),
                        }
        with open(self.stations_json_path, "w") as fp: json.dump(station_locs, fp, indent=2)
        print(f"Metadata stasiun (JSON) berhasil disimpan di: {self.stations_json_path}")
        with open(self.stations_pkl_path, "wb") as fp: pickle.dump(stations, fp)
        print(f"Objek stasiun (PKL) berhasil disimpan di: {self.stations_pkl_path}")
        if (plot or save_plot) and station_locs:
            station_df = pd.DataFrame.from_dict(station_locs, orient="index")
            plt.figure(figsize=(10, 8)); plt.plot(station_df["longitude"], station_df["latitude"], "^", label="Stasiun"); plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.axis("scaled"); plt.xlim(config["xlim_degree"]); plt.ylim(config["ylim_degree"]); plt.legend(); plt.title(f"Lokasi Stasiun ({len(station_df)} stasiun)")
            if save_plot:
                plt.savefig(self.station_plot_path); print(f"Plot lokasi stasiun disimpan di: {self.station_plot_path}")
            if plot: plt.show()
            plt.close()

    def download_waveforms(self, max_threads=4):
        """Download Waveforms dalam segmen per jam."""
        print("Memulai Download Waveforms...")
        with open(self.config_path, "r") as fp: config = json.load(fp)
        with open(self.datetimes_path, "r") as fp: tmp = json.load(fp); starttimes, interval = tmp["starttimes"], tmp["interval"]
        with open(self.stations_pkl_path, "rb") as fp: stations = pickle.load(fp)
        client = self._get_fdsn_client(config); fname_list = []; lock = threading.Lock()
        def _download_task(starttime_str):
            starttime = obspy.UTCDateTime(starttime_str); endtime = starttime + interval; fname = f"{starttime.datetime.strftime('%Y-%m-%dT%H:%M:%S')}.mseed"; fpath = os.path.join(self.waveform_dir, fname)
            if os.path.exists(fpath): print(f"File sudah ada: {fname}"); lock.acquire(); fname_list.append(fname); lock.release(); return
            print(f"Download: {fname}"); stream = obspy.Stream()
            for network in stations:
                for station in network:
                    try: stream += client.get_waveforms(network.code, station.code, "*", config["channels"], starttime, endtime)
                    except Exception as err:
                        if str(err).startswith("No data available for request."): continue
                        print(f"Gagal Download {network.code}.{station.code}: {err}")
            if len(stream) > 0: stream.write(fpath); print(f"Berhasil Download: {fname}"); lock.acquire(); fname_list.append(fname); lock.release()
            else: print(f"Tidak ada data untuk: {fname}")
        threads = []
        for start_str in starttimes:
            t = threading.Thread(target=_download_task, args=(start_str,)); t.start(); threads.append(t)
            if len(threads) >= max_threads:
                for th in threads: th.join()
                threads = []
        for t in threads: t.join()
        with open(self.fname_csv_path, "w") as fp:
            fp.write("fname\n")
            for fname in sorted(fname_list): fp.write(f"{fname}\n")
        print(f"Waveform disimpan di: {self.fname_csv_path}")

    def run_phasenet(self):
        """Menjalankan skrip prediksi PhaseNet."""
        print("Menjalankan PhaseNet untuk phase picking...")
        
        phasenet_script_path = os.path.join(self.base_dir, "core", "PhaseNet", "phasenet", "predict.py")

        command = (f"python {phasenet_script_path} --model={self.phasenet_model_path} "
                   f"--data_dir={self.waveform_dir} --data_list={self.fname_csv_path} "
                   f"--stations={self.stations_json_path} --result_dir={self.phasenet_dir} "
                   f"--format=mseed_array --amplitude")
        print(f"Menjalankan perintah: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"PhaseNet selesai. Hasil disimpan di: {self.phasenet_picks_path}")
        except subprocess.CalledProcessError as e:
            print(f"Terjadi error saat menjalankan PhaseNet: {e}")
            raise

    def run_gamma_association(self):
        """Melakukan asosiasi gempa menggunakan GaMMA."""
        print("Menjalankan GaMMA untuk asosiasi gempa...")
        with open(self.config_path, "r") as fp: config = json.load(fp)
        picks = pd.read_csv(self.phasenet_picks_path, parse_dates=["phase_time"])
        picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_amp": "amp", "phase_type": "type", "phase_score": "prob"}, inplace=True)
        with open(self.stations_json_path, "r") as fp: stations = json.load(fp)
        stations_df = pd.DataFrame.from_dict(stations, orient="index"); stations_df["id"] = stations_df.index
        proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
        stations_df[["x(km)", "y(km)"]] = stations_df.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
        stations_df["z(km)"] = stations_df["elevation(m)"].apply(lambda x: -x / 1e3)
        
        gamma_config = {
            "center": config["center"], "dims": ["x(km)", "y(km)", "z(km)"], 
            "use_dbscan": True, "use_amplitude": True, "method": "BGMM",
            "dbscan_eps": 10, "dbscan_min_samples": 3,
            "vel": {"p": 6.0, "s": 6.0 / 1.73},
            "x(km)": (np.array(config["xlim_degree"]) - config["center"][0]) * config["degree2km"],
            "y(km)": (np.array(config["ylim_degree"]) - config["center"][1]) * config["degree2km"],
            "z(km)": (0, 60),
            "min_picks_per_eq": min(10, len(stations_df) // 2),
            "max_sigma11": 2.0, "max_sigma22": 2.0,
            "oversample_factor": 4
        }
        gamma_config["bfgs_bounds"] = ((gamma_config["x(km)"][0] - 1, gamma_config["x(km)"][1] + 1), (gamma_config["y(km)"][0] - 1, gamma_config["y(km)"][1] + 1), (0, gamma_config["z(km)"][1] + 1), (None, None))
        
        catalogs, assignments = association(picks, stations_df, gamma_config, event_idx0=0, method=gamma_config["method"])
        if not catalogs: print("GaMMA tidak menghasilkan katalog."); return
        catalog_df = pd.DataFrame(catalogs, columns=["time"] + gamma_config["dims"] + ["magnitude", "sigma_time", "sigma_amp", "cov_time_amp", "event_index", "gamma_score"])
        catalog_df[["longitude", "latitude"]] = catalog_df.apply(lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1)
        catalog_df["depth(m)"] = catalog_df["z(km)"].apply(lambda x: x * 1e3); catalog_df.sort_values(by=["time"], inplace=True)
        catalog_df.to_csv(self.gamma_catalog_path, index=False, float_format="%.3f", date_format="%Y-%m-%dT%H:%M:%S.%f")
        print(f"Katalog GaMMA berhasil disimpan di: {self.gamma_catalog_path}")
        assignments_df = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
        picks_df = picks.join(assignments_df.set_index("pick_index")).fillna(-1).astype({"event_index": int}); picks_df.sort_values(by=["timestamp"], inplace=True)
        picks_df.rename(columns={"id": "station_id", "timestamp": "phase_time", "amp": "phase_amp", "type": "phase_type", "prob": "phase_score"}, inplace=True)
        picks_df.to_csv(self.gamma_picks_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
        print(f"Picks GaMMA berhasil disimpan di: {self.gamma_picks_path}")

    def prepare_hypodd_files(self):
        """Mempersiapkan file input untuk HypoDD."""
        print("Mempersiapkan file input untuk HypoDD...")
        self._convert_stations_for_hypodd()
        self._convert_phases_for_hypodd()
        self._create_hypodd_inp() # Membuat file hypoDD.inp
        self._run_ph2dt()

    def _convert_stations_for_hypodd(self):
        """Mengonversi file stasiun ke format HypoDD."""
        stations = pd.read_json(self.stations_json_path, orient="index")
        with open(self.hypodd_station_path, "w") as f:
            for sta, row in stations.iterrows(): _, station_code, _, _ = sta.split("."); f.write(f"{station_code:<8s} {row['latitude']:.3f} {row['longitude']:.3f}\n")
        print(f"File stasiun HypoDD berhasil disimpan di: {self.hypodd_station_path}")

    def _convert_phases_for_hypodd(self):
        """Mengonversi fase ke format HypoDD."""
        picks = pd.read_csv(self.gamma_picks_path); events = pd.read_csv(self.gamma_catalog_path); events.sort_values("time", inplace=True)
        with open(self.hypodd_phase_path, "w") as f:
            picks_by_event = picks.groupby("event_index")
            for _, event in events.iterrows():
                event_time = datetime.datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f")
                line = (f"# {event_time.year:4d} {event_time.month:2d} {event_time.day:2d} {event_time.hour:2d} {event_time.minute:2d} {float(event_time.strftime('%S.%f')):5.2f}  {event['latitude']:7.4f} {event['longitude']:9.4f}   {event['depth(m)']/1e3:5.2f} {event['magnitude']:5.2f} 0.00 0.00 {event['sigma_time']:5.2f} {event['event_index']:9d}\n"); f.write(line)
                if event["event_index"] in picks_by_event.groups:
                    event_picks = picks_by_event.get_group(event["event_index"])
                    for _, pick in event_picks.iterrows():
                        _, station_code, _, _ = pick["station_id"].split("."); pick_time = (datetime.datetime.strptime(pick["phase_time"], "%Y-%m-%dT%H:%M:%S.%f") - event_time).total_seconds(); line = (f"{station_code:<7s}   {pick_time:6.3f}   {pick['phase_score']:5.4f}   {pick['phase_type'].upper()}\n"); f.write(line)
        print(f"File fase HypoDD berhasil disimpan di: {self.hypodd_phase_path}")

    def _create_hypodd_inp(self):
        """Membuat file hypoDD.inp standar berdasarkan referensi."""
        hypodd_inp_content = """
* RELOC.INP:
*--- input file selection
* cross correlation diff times:
*
*catalog P diff times:
dt.ct
*
* event file:
event.sel
*
* station file:
station.dat
*
*--- output file selection
* original locations:
hypoDD.loc
* relocations:
hypoDD.reloc
* station information:
hypoDD.sta
* residual information:
hypoDD.res
* source paramater information:
hypoDD.src
*
*--- data type selection: 
* IDAT:  1= cross corr; 2= catalog; 3= cross & cat 
* IPHA: 1= P; 2= S; 3= P&S
* DIST:max dist [km] between cluster centroid and station 
* IDAT   IPHA   MAXDIST  
    3     3      120
*
*--- event clustering:
* OBSCC:    min # of obs/pair for crosstime data (0= no clustering)
* OBSCT:    min # of obs/pair for network data (0= no clustering)
* OBSCC  OBSCT
    0     8
*
*--- solution control:
* ISTART:  	1 = from single source; 2 = from network sources
* ISOLV:	1 = SVD, 2=lsqr
* NSET:      	number of sets of iteration with specifications following
* ISTART  ISOLV  NSET
    2        2     3
*
*--- data weighting and re-weighting: 
* NITER: 		last iteration to used the following weights
* WTCCP, WTCCS:		weight cross P, S 
* WTCTP, WTCTS:		weight catalog P, S 
* WRCC, WRCT:		residual threshold in sec for cross, catalog data 
* WDCC, WDCT:  		max dist [km] between cross, catalog linked pairs
* DAMP:    		damping (for lsqr only) 
* ---  CROSS DATA ----- ----CATALOG DATA ----
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP
2     -9     -9    -9   -9    1     1    4    20   10
2     -9     -9    -9   -9    1     1    4    20   9
1     -9     -9    -9   -9    1     1    4    20   8
*
*--- 1D model:
* NLAY:		number of model layers  
* RATIO:	vp/vs ratio 
* TOP:		depths of top of layer (km) 
* VEL: 		layer velocities (km/s)
* NLAY  RATIO 
12    1.73
* TOP
5.0 10.0 15.0 25.0 35.0 45.0 60.0 100.0 160.0 210.0 360.0 460.0
* VEL
5.00  6.00  6.75  7.11  7.24  7.37  7.60  7.95  8.17  8.30  8.80  9.52
*
*--- event selection:
* CID: 	cluster to be relocated (0 = all)
* ID:	cuspids of event to be relocated (8 per line)
* CID    
    0      
* ID
    """
        hypodd_inp_path = os.path.join(self.hypodd_dir, "hypoDD.inp")
        with open(hypodd_inp_path, "w") as f:
            f.write(hypodd_inp_content)
        print(f"File hypoDD.inp standar berhasil dibuat di: {hypodd_inp_path}")
        print("-> PENTING: Harap periksa dan sesuaikan parameter di dalam 'hypoDD.inp' sebelum menjalankan 'hypodd_run'.")


    def _run_ph2dt(self):
        """Menjalankan ph2dt."""
        ph2dt_inp_content = "* ph2dt.inp - input control file for program ph2dt\n* Input station file:\nstation.dat\n* Input phase file:\nphase.pha\n*MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS\n   0      120     10     50     8      8     100\n"
        with open(os.path.join(self.hypodd_dir, "ph2dt.inp"), "w") as f: f.write(ph2dt_inp_content)
        command = f"cd {self.hypodd_dir} && ./ph2dt ph2dt.inp"; print(f"Menjalankan perintah: {command}")
        try: subprocess.run(command, shell=True, check=True); print(f"ph2dt berhasil dijalankan. File dt.ct dan event.sel dibuat di: {self.hypodd_dir}")
        except subprocess.CalledProcessError as e: print(f"Terjadi error saat menjalankan ph2dt: {e}"); raise
    
    def run_hypodd(self):
        """Menjalankan relokasi HypoDD."""
        hypodd_inp_path = os.path.join(self.hypodd_dir, "hypoDD.inp"); print("\n--- Relokasi HypoDD ---")
        if not os.path.exists(hypodd_inp_path): print(f"File 'hypoDD.inp' tidak ditemukan di '{self.hypodd_dir}'.\nSilakan buat file tersebut untuk melanjutkan relokasi HypoDD.")
        else:
            command = f"cd {self.hypodd_dir} && ./hypoDD hypoDD.inp"; print(f"Menjalankan perintah: {command}")
            try: subprocess.run(command, shell=True, check=True); print("HypoDD berhasil dijalankan.")
            except subprocess.CalledProcessError as e: print(f"Terjadi error saat menjalankan HypoDD: {e}"); raise

    def plot_relocated_results(self):
        """Membaca file hypoDD.reloc dan membuat plot."""
        print("Membuat plot hasil relokasi...")
        reloc_path = os.path.join(self.hypodd_dir, "hypoDD.reloc")
        if not os.path.exists(reloc_path):
            print(f"Peringatan: File hasil '{reloc_path}' tidak ditemukan. Lewati plotting.")
            return

        cols = ['ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'EX', 'EY', 'EZ', 'YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'MAG', 'NCCP', 'NCCS', 'NCTP', 'NCTS', 'RCC', 'RCT', 'CID']
        
        try:
            eq_df = pd.read_csv(reloc_path, delim_whitespace=True, names=cols)
        except Exception as e:
            print(f"Error saat membaca file hypoDD.reloc: {e}")
            return

        with open(self.stations_json_path) as f:
            station_data = json.load(f)

        stations = []
        for name, props in station_data.items():
            stations.append({
                "name": name.split('.')[1], # Ambil nama stasiun saja
                "latitude": props["latitude"],
                "longitude": props["longitude"],
                "elevation": props["elevation(m)"]
            })
        sta_df = pd.DataFrame(stations)

        plt.figure(figsize=(10, 8))

        sc = plt.scatter(eq_df['LON'], eq_df['LAT'], c=eq_df['DEPTH'], cmap='viridis_r',
                         s=eq_df['MAG']**2 * 10, edgecolor='k', alpha=0.7, label='Gempa Relokasi')
        plt.colorbar(sc, label='Kedalaman (km)')

        plt.scatter(sta_df['longitude'], sta_df['latitude'], c='blue', marker='^',
                    s=80, edgecolor='k', label='Stasiun')

        min_lat = min(eq_df['LAT'].min(), sta_df['latitude'].min()) - 0.1
        max_lat = max(eq_df['LAT'].max(), sta_df['latitude'].max()) + 0.1
        min_lon = min(eq_df['LON'].min(), sta_df['longitude'].min()) - 0.1
        max_lon = max(eq_df['LON'].max(), sta_df['longitude'].max()) + 0.1
        plt.xlim(min_lon, max_lon)
        plt.ylim(min_lat, max_lat)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Peta Sebaran Hasil Relokasi HypoDD")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.event_plot_path)
        print(f"Plot hasil relokasi disimpan di: {self.event_plot_path}")
        plt.close()


    def run_full_workflow(self, config_params, plot=False, save_plots=True):
        """Menjalankan seluruh alur kerja secara berurutan."""
        print("--- Memulai Alur Kerja Pemrosesan Seismik ---")
        print("\n[Langkah 1 dari 7] Mengatur Konfigurasi"); self.set_config(**config_params)
        print("\n[Langkah 2 dari 7] Download Katalog Gempa"); self.download_event_catalog(plot=plot, save_plot=save_plots)
        print("\n[Langkah 3 dari 7] Download Metadata Stasiun"); self.download_station_metadata(plot=plot, save_plot=save_plots)
        print("\n[Langkah 4 dari 7] Download Data Bentuk Gelombang"); self.download_waveforms()
        print("\n[Langkah 5 dari 7] Menjalankan PhaseNet"); self.run_phasenet()
        print("\n[Langkah 6 dari 7] Menjalankan GaMMA"); self.run_gamma_association()
        print("\n[Langkah 7 dari 7] Mempersiapkan File untuk HypoDD"); self.prepare_hypodd_files()
        print("\n--- Alur Kerja Selesai (Hingga Persiapan HypoDD) ---")
        self.run_hypodd()
