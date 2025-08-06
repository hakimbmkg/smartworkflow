# SmartWorkFlow

**Tools Event Detection berbasis metode AI (PhaseNet, Gamma) serta Relokasi Event dengan HypoDD.**  
Bagian dari proyek **SMART - IDRIP BMKG @ 2024**  
Re-Code by: `@hakimbmkg`

---

## 

SeismicWorkFlow merupakan alat bantu untuk mendeteksi kejadian gempa bumi secara otomatis menggunakan pendekatan kecerdasan buatan dan statistika, yang meliputi:

- **PhaseNet** untuk picking waktu tiba gelombang seismik
- **Gamma** untuk asosiasi fase gempa
- **HypoDD** untuk relokasi event 

## Contoh
```
chmod +x install.sh
```
```
./install.sh
```
```
conda activate smartworkflow
```   
```
python run.py --help
```
```
python run.py config
```
```
python run.py download_waveforms
```
```
Set Konfigurasi di Folder config/config.txt
=================================================
Konfigurasi 
=================================================

--- FDSN CLIENT ---
FDSN_CLIENT = LOCAL

--- KREDENSIAL FDSN (OPSIONAL) ---
#Kosongkan jika tidak memerlukan autentikasi.
FDSN_USER = admin
FDSN_PASSWORD = admin

--- PARAMETER WILAYAH ---
REGION_NAME = BToru

--- PARAMETER GEOGRAFIS DAN WAKTU ---
CENTER_LON = 99.10
CENTER_LAT = 1.60
H_DEG = 1.5
V_DEG = 1.5
STARTTIME_STR = 2025-08-03T00:00:00
ENDTIME_STR = 2025-08-03T23:59:59

--- PARAMETER JARINGAN DAN CHANNELS ---
Pisahkan dengan koma, tanpa spasi
NETWORKS = IA,AM,BT
CHANNELS = SHZ,EHZ
```
---

## Referensi 

**WAJIB mencantumkan referensi berikut dalam penggunaan atau publikasi yang memanfaatkan kode ini:**

1. Zhu, Weiqiang, and Gregory C. Beroza.  
   *PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method.*  
   arXiv preprint [arXiv:1803.03211](https://arxiv.org/abs/1803.03211) (2018)

2. Zhu, W., McBrearty, I. W., Mousavi, S. M., Ellsworth, W. L., & Beroza, G. C. (2022).  
   *Earthquake phase association using a Bayesian Gaussian Mixture Model.*  
   *Journal of Geophysical Research: Solid Earth*, 127, e2021JB023249.  
   [https://doi.org/10.1029/2021JB023249](https://doi.org/10.1029/2021JB023249)

3. Waldhauser, F., & Ellsworth, W. L. (2000).  
   *A Double-difference Earthquake Location Algorithm: Method and Application to the Northern Hayward Fault, California.*  
   *Bulletin of the Seismological Society of America*, **90**, 1353–1368.

4. Waldhauser, F. (2001).  
   *HypoDD - A Program to Compute Double-Difference Hypocenter Locations.*  
   U.S. Geological Survey Open-File Report 01–113.

---

## Lisensi

Project ini bersifat open-source, namun **penggunaan akademik dan publikasi WAJIB mencantumkan referensi yang disebutkan di atas.**

---



