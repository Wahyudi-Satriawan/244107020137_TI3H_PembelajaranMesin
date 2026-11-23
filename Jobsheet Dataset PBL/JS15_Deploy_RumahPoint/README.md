# Paket Deployment Produksi RumahPoint — Hugging Face Spaces

Direktori ini berisi paket layanan inferensi web dan API siap rilis (*production-ready deployment package*) untuk model klasifikasi kelas harga properti multi-kota (**RumahPoint**) berbasis **Deep Neural Network (Keras 3)**, **Scikit-Learn Pipeline**, dan **Flask WSGI** di dalam container **Docker**.

---

## 1. Struktur Direktori

```text
JS15_Deploy_RumahPoint/
├── artifacts/                  # Folder artefak biner hasil training JS15.ipynb di Colab
│   ├── model_rumahpoint.keras  # Model Deep Learning terlatih (Keras 3 native zip format)
│   ├── preprocessor.joblib     # Scikit-Learn ColumnTransformer (imputer + scaler + OHE)
│   └── metadata.json           # Top-15 lokasi, label target, versi library, threshold kuantil
├── app.py                      # Aplikasi Flask (Web UI + REST API /health & /predict)
├── requirements.txt            # Dependensi Python container (Flask, Keras, TensorFlow CPU)
├── Dockerfile                  # Konfigurasi container non-root user port 7860 untuk Hugging Face
└── README.md                   # Dokumentasi operasional deployment ini
```

> **Catatan:** File biner model di dalam folder `artifacts/` tidak disertakan di Git repository (dikecualikan via `.gitignore` demi menjaga ukuran repositori). Artefak tersebut dihasilkan secara otomatis saat Anda mengeksekusi notebook `Jobsheet Dataset PBL/JS15.ipynb` di Google Colab.

---

## 2. Cara Ekspor Artefak dari Google Colab

1. Buka dan jalankan seluruh sel pada notebook `Jobsheet Dataset PBL/JS15.ipynb` di Google Colab.
2. Di Bagian 3 dan 4 notebook, sel kode akan mengekspor 3 artefak ke folder `artifacts/` dan mengemasnya ke dalam file `rumahpoint_hf_deploy.zip`.
3. Unduh file `rumahpoint_hf_deploy.zip` ke komputer lokal Anda.
4. Ekstrak isi folder `artifacts/` dari file zip tersebut ke dalam direktori `JS15_Deploy_RumahPoint/artifacts/`:
   - `artifacts/model_rumahpoint.keras`
   - `artifacts/preprocessor.joblib`
   - `artifacts/metadata.json`

---

## 3. Menjalankan Aplikasi Secara Lokal (Tanpa Docker)

Pastikan lingkungan Python Anda (disarankan virtualenv atau conda) telah menginstal dependensi:

```bash
# Pindah ke direktori deployment
cd "Jobsheet Dataset PBL/JS15_Deploy_RumahPoint"

# Install dependensi
pip install -r requirements.txt

# Jalankan server Flask lokal
python app.py
```

Aplikasi akan berjalan dan dapat diakses melalui browser pada:
👉 `http://localhost:7860`

---

## 4. Menjalankan Menggunakan Docker (Simulasi Lokal)

Untuk memastikan container identik dengan lingkungan cloud Hugging Face Spaces:

```bash
# 1. Build image Docker
docker build -t rumahpoint-deploy:latest .

# 2. Jalankan container pada port 7860
docker run -p 7860:7860 rumahpoint-deploy:latest
```

Buka browser ke `http://localhost:7860` untuk menguji antarmuka web.

---

## 5. Langkah Deployment ke Hugging Face Spaces

1. **Buat Space Baru di Hugging Face:**
   - Masuk ke akun [Hugging Face](https://huggingface.co/).
   - Klik foto profil di sudut kanan atas > pilih **New Space**.
   - **Space Name:** `rumahpoint-classifier` (atau nama lain yang Anda kehendaki).
   - **License:** Apache 2.0 atau MIT.
   - **Space SDK:** Pilih **Docker** (Blank / Kosong).
   - **Space Hardware:** CPU Basic (Gratis).
   - **Visibility:** Public.
   - Klik **Create Space**.

2. **Dapatkan User Access Token:**
   - Buka menu **Settings > Access Tokens** di profil Hugging Face Anda.
   - Buat token baru dengan izin (*role*) **Write**. Simpan token tersebut untuk autentikasi git.

3. **Clone Repository Space & Unggah File:**
   ```bash
   # Clone repositori Space Hugging Face Anda
   git clone https://huggingface.co/spaces/USERNAME_ANDA/rumahpoint-classifier
   cd rumahpoint-classifier

   # Salin seluruh isi folder JS15_Deploy_RumahPoint/ (termasuk folder artifacts/) ke dalam repositori ini
   cp -r /path/to/JS15_Deploy_RumahPoint/* .

   # Commit dan Push ke Hugging Face
   git add .
   git commit -m "Deploy RumahPoint Classifier ke HF Spaces"
   git push origin main
   ```
   *Saat diminta password git, masukkan Access Token bertipe Write yang telah Anda buat.*

4. **Verifikasi Status:**
   - Buka tab **App** di halaman Space Hugging Face Anda.
   - Status akan berubah dari **Building** menjadi **Running** dalam kurun waktu 2–4 menit.

---

## 6. Dokumentasi Endpoint REST API

### 1. Health Check
- **URL:** `GET /health`
- **Response:**
  ```json
  {
    "status": "ok",
    "service": "RumahPoint Price Tier Prediction Service",
    "artifacts": {
      "model_loaded": true,
      "preprocessor_loaded": true,
      "metadata_loaded": true
    }
  }
  ```

### 2. Prediksi Kelas Harga
- **URL:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Payload Input:**
  ```json
  {
    "luas_m2": 120,
    "luas_bangunan": 90,
    "jumlah_kamar": 3,
    "jumlah_kamar_mandi": 2,
    "lokasi_bersih": "Kota Jakarta Selatan",
    "kondisi": "Bekas"
  }
  ```
- **Contoh Response Sukses:**
  ```json
  {
    "status": "success",
    "prediksi": {
      "kelas_harga": "Sedang",
      "kategori_indeks": 1,
      "confidence": 84.52,
      "probabilitas": {
        "Murah": 0.0512,
        "Sedang": 0.8452,
        "Mahal": 0.1036
      }
    },
    "ambang_batas_referensi": {
      "Murah": "< Rp 483,82 Juta",
      "Sedang": "Rp 483,82 Juta - Rp 1,235 Miliar",
      "Mahal": "> Rp 1,235 Miliar"
    }
  }
  ```

### 3. Pengujian via cURL
```bash
curl -X POST https://USERNAME-rumahpoint-classifier.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{
       "luas_m2": 150,
       "luas_bangunan": 120,
       "jumlah_kamar": 4,
       "jumlah_kamar_mandi": 3,
       "lokasi_bersih": "Kota Bandung",
       "kondisi": "Baru"
     }'
```
