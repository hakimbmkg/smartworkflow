#!/bin/bash

# Mendeteksi OS Mac M1/M2/M3 untuk menggunakan file env_mac.yml.

ENV_NAME="smartworkflow"
PYTHON_VERSION="3.10"

# --- Langkah 1: Cek dan aktifkan Conda ---
echo "--- Memeriksa instalasi Conda... ---"
if ! command -v conda &> /dev/null; then
    echo "Perintah 'conda' tidak ditemukan di PATH. Mencari instalasi..."
    
    # Daftar path umum untuk file aktivasi conda
    CONDA_PATHS=(
        "$HOME/miniconda/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )

    CONDA_ACTIVATION_SCRIPT=""
    for path in "${CONDA_PATHS[@]}"; do
        if [ -f "$path" ]; then
            CONDA_ACTIVATION_SCRIPT="$path"
            break
        fi
    done

    if [ -z "$CONDA_ACTIVATION_SCRIPT" ]; then
        echo "KESALAHAN: Anaconda atau Miniconda tidak ditemukan."
        echo "Silakan instal terlebih dahulu dan jalankan 'conda init'."
        exit 1
    fi
    
    echo "Mencoba mengaktifkan Conda dari: $CONDA_ACTIVATION_SCRIPT"

    source "$CONDA_ACTIVATION_SCRIPT"
    if ! command -v conda &> /dev/null; then
        echo "KESALAHAN: Gagal mengaktifkan Conda secara otomatis."
        exit 1
    fi
fi
echo "Conda ditemukan dan aktif."
echo ""

# --- Langkah 2: Deteksi OS dan instal dependensi ---
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "--- Terdeteksi macOS Apple Silicon (M1/M2/M3). Menggunakan env_mac.yaml. ---"
    if [ ! -f "env_mac.yaml" ]; then
        echo "KESALAHAN: File 'env_mac.yaml' tidak ditemukan untuk instalasi Mac."
        exit 1
    fi
    conda env create -f env_mac.yaml
    if [ $? -ne 0 ]; then
        echo "KESALAHAN: Gagal membuat lingkungan dari env_mac.yaml."
        exit 1
    fi
    echo "Lingkungan '$ENV_NAME' berhasil dibuat dari file YAML."

else
    echo "--- Terdeteksi sistem Linux/Windows/Intel Mac. Melanjutkan instalasi standar. ---"
    # --- Langkah 2a: Buat lingkungan Conda baru ---
    echo "--- Membuat lingkungan baru bernama '$ENV_NAME' dengan Python $PYTHON_VERSION... ---"
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo "KESALAHAN: Gagal membuat lingkungan Conda. Mohon periksa instalasi Conda Anda."
        exit 1
    fi
    echo "Lingkungan '$ENV_NAME' berhasil dibuat."
    echo ""

    # --- Langkah 2b: Instal dependensi utama dengan Conda ---
    echo "--- Menginstal dependensi utama dari conda-forge (termasuk TensorFlow)... ---"
    conda install -n $ENV_NAME --channel conda-forge numpy obspy pandas matplotlib pyproj tqdm tensorflow scikit-learn scipy h5py -y
    if [ $? -ne 0 ]; then
        echo "KESALAHAN: Gagal menginstal dependensi utama dengan Conda."
        exit 1
    fi
    echo "Dependensi utama berhasil diinstal."
fi
echo ""

# --- Langkah 3: Kloning Repositori Eksternal ---
CORE_DIR="core"
PHASENET_DIR="$CORE_DIR/PhaseNet"
GAMMA_DIR="$CORE_DIR/GaMMA"

# Kloning PhaseNet
echo "--- Mengkloning repository PhaseNet... ---"
if [ -d "$PHASENET_DIR" ]; then
    echo "Direktori '$PHASENET_DIR' sudah ada, proses kloning dilewati."
else
    mkdir -p $CORE_DIR
    git clone https://github.com/AI4EPS/PhaseNet.git $PHASENET_DIR
    if [ $? -ne 0 ]; then
        echo "KESALAHAN: Gagal mengkloning PhaseNet."
        exit 1
    fi
    echo "PhaseNet berhasil dikloning."
fi

# Kloning GaMMA
echo "--- Mengkloning repository GaMMA... ---"
if [ -d "$GAMMA_DIR" ]; then
    echo "Direktori '$GAMMA_DIR' sudah ada, proses kloning dilewati."
else
    mkdir -p $CORE_DIR
    git clone https://github.com/AI4EPS/GaMMA.git $GAMMA_DIR
    if [ $? -ne 0 ]; then
        echo "KESALAHAN: Gagal mengkloning GaMMA."
        exit 1
    fi
    echo "GaMMA berhasil dikloning."
fi
echo ""

# --- Langkah 4: Instal pustaka dari requirements.txt (jika ada) ---
if [ -f "requirements.txt" ]; then
    echo "--- Menginstal pustaka dari requirements.txt dengan pip... ---"
    conda run -n $ENV_NAME python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "KESALAHAN: Gagal menginstal dari requirements.txt."
        exit 1
    fi
    echo "Instalasi dari requirements.txt berhasil."
    echo ""
fi

# --- Langkah 5: Instal PhaseNet dan GaMMA secara lokal (tanpa dependensi) ---
echo "--- Menginstal PhaseNet dan GaMMA dari folder lokal... ---"
conda run -n $ENV_NAME python -m pip install --no-deps -e $PHASENET_DIR
conda run -n $ENV_NAME python -m pip install --no-deps -e $GAMMA_DIR
if [ $? -ne 0 ]; then
    echo "KESALAHAN: Gagal menginstal PhaseNet atau GaMMA secara lokal."
    exit 1
fi
echo "PhaseNet dan GaMMA berhasil diinstal."
echo ""

# --- Langkah 6: Verifikasi instalasi ---
echo "--- Menjalankan verifikasi dengan 'python run.py --help'... ---"
conda run -n $ENV_NAME python run.py --help
if [ $? -ne 0 ]; then
    echo "PERINGATAN: Verifikasi gagal. Pastikan 'run.py' berada di direktori yang sama."
else
    echo "Verifikasi berhasil!"
fi
echo ""

# --- Selesai ---
echo "====================================================================="
echo " Penyiapan Selesai!"
echo "====================================================================="
echo "Untuk mengaktifkan lingkungan kerja Anda, jalankan perintah berikut:"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "Setelah itu, Anda bisa menjalankan skrip seperti biasa, contohnya:"
echo "  python run.py config"
echo "====================================================================="

