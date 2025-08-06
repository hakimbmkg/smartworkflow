#!/bin/bash

# Script untuk menyiapkan lingkungan kerja smartworkflow secara otomatis.
# Pastikan Anda memiliki Anaconda atau Miniconda terinstal.

ENV_NAME="smartworkflow"
PYTHON_VERSION="3.10"

# --- Langkah 1: Cek apakah Conda terinstal ---
echo "--- Memeriksa instalasi Conda... ---"
if ! command -v conda &> /dev/null
then
    echo "KESALAHAN: Anaconda atau Miniconda tidak ditemukan."
    echo "Silakan instal terlebih dahulu sebelum menjalankan skrip ini."
    exit 1
fi
echo "Conda ditemukan."
echo ""

# --- Langkah 2: Buat lingkungan Conda baru ---
echo "--- Membuat lingkungan baru bernama '$ENV_NAME'... ---"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y
if [ $? -ne 0 ]; then
    echo "KESALAHAN: Gagal membuat lingkungan Conda. Mohon periksa instalasi Conda Anda."
    exit 1
fi
echo "Lingkungan '$ENV_NAME' berhasil dibuat."
echo ""

# --- Langkah 3: Instal dependensi dari requirements.txt ---
echo "--- Menginstal pustaka yang dibutuhkan dari requirements.txt... ---"
if [ ! -f "requirements.txt" ]; then
    echo "KESALAHAN: File 'requirements.txt' tidak ditemukan."
    exit 1
fi
# Menggunakan 'python -m pip' untuk memastikan pip yang benar digunakan di dalam environment
conda run -n $ENV_NAME python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "KESALAHAN: Gagal menginstal dependensi. Mohon periksa file requirements.txt dan koneksi internet Anda."
    exit 1
fi
echo "Semua dependensi berhasil diinstal."
echo ""

# --- Langkah 4: Verifikasi instalasi ---
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

