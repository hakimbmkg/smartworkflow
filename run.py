import argparse
import os
from main.smartworkflow import SmartWorkflow 

def load_config(config_filepath):
    """Load Config TXT, isi semua Konfigurasi pada file .txt"""
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan di: '{config_filepath}'")
    
    config = {}
    with open(config_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Abaikan baris kosong, baris komentar, atau baris tanpa '='
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def main():
    """
    SeismicWorkFLow :: Tools Event Detection berbasis metode AI (PhaseNet, Gamma) serta Relokasi Event dengan HypoDD.
    Project :: SMART - IDRIP BMKG @ 2025
    Re-Code :: hakimbmkg 
    
    Contoh:
    python run.py full
    python run.py hypodd_run 
    """
    parser = argparse.ArgumentParser(description=
    """
    SeismicWorkFLow :: Tools Event Detection berbasis metode AI (PhaseNet, Gamma) serta Relokasi Event dengan HypoDD.
    Project :: SMART - IDRIP BMKG @ 2025
    Re-Code :: hakimbmkg 
    
    Contoh:
    python run.py full
    python run.py hypodd_run 
    """)
    
    parser.add_argument(
        'step', 
        choices=[
            'config', 'download_events', 'download_stations', 'download_waveforms', 
            'phasenet', 'gamma', 'hypodd_prep', 'hypodd_run', 'plot', 'full'
        ],
        help="Pilih berdasarkan fungsi atau jalankan semua fungsi dengan argumen 'full'."
    )

    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PHASENET_MODEL_PATH = os.path.join(BASE_DIR, "core", "PhaseNet", "model", "190703-214543")
    HYPODD_EXEC_PATH = os.path.join(BASE_DIR, "core", "HYPODD", "src")

    try:
        config_dir = os.path.join(BASE_DIR, 'config')
        config_filepath = os.path.join(config_dir, 'config.txt')
        config = load_config(config_filepath)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Pastikan folder 'config' ada di direktori yang sama dengan run.py,")
        print("dan file 'config.txt' ada di dalamnya.")
        return
        
    # Mengambil nilai dari dictionary config
    REGION_NAME = config.get('REGION_NAME')
    
    # Membuat dictionary parameter untuk diteruskan ke metode
    config_parameters = {
        "center": (float(config.get('CENTER_LON')), float(config.get('CENTER_LAT'))),
        "h_deg": float(config.get('H_DEG')),
        "v_deg": float(config.get('V_DEG')),
        "starttime_str": config.get('STARTTIME_STR'),
        "endtime_str": config.get('ENDTIME_STR'),
        "networks": [net.strip() for net in config.get('NETWORKS').split(',')],
        "channels": config.get('CHANNELS'),
        "client": config.get('FDSN_CLIENT', 'IRIS'),
        "user": config.get('FDSN_USER', None), # User opsional
        "password": config.get('FDSN_PASSWORD', None) # Password opsional
    }
    
    # Validasi path
    if not all([PHASENET_MODEL_PATH, HYPODD_EXEC_PATH, REGION_NAME]):
        print("ERROR: Pastikan path dan nama wilayah sudah benar.")
        return

    # Inisialisasi kelas dengan direktori dasar
    workflow = SmartWorkflow(
        base_dir=BASE_DIR,
        region_name=REGION_NAME,
        phasenet_model_path=PHASENET_MODEL_PATH,
        hypodd_exec_path=HYPODD_EXEC_PATH
    )
    
    # Menjalankan langkah yang dipilih
    if args.step == 'config':
        workflow.set_config(**config_parameters)
    elif args.step == 'download_events':
        workflow.download_event_catalog(save_plot=True)
    elif args.step == 'download_stations':
        workflow.download_station_metadata(save_plot=True)
    elif args.step == 'download_waveforms':
        workflow.download_waveforms()
    elif args.step == 'phasenet':
        workflow.run_phasenet()
    elif args.step == 'gamma':
        workflow.run_gamma_association()
    elif args.step == 'hypodd_prep':
        workflow.prepare_hypodd_files()
    elif args.step == 'hypodd_run':
        workflow.run_hypodd()
    elif args.step == 'plot':
        workflow.plot_relocated_results()
    elif args.step == 'full':
        workflow.run_full_workflow(config_params=config_parameters, save_plots=True)

if __name__ == '__main__':
    main()
