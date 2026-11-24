# Repositori Pembelajaran Mesin (Machine Learning)
### Portofolio Praktikum & Penerapan Proyek PBL RumahPoint (Dataset Properti Multi-Kota Indonesia)
**Program Studi:** D-IV Teknik Informatika — Jurusan Teknologi Informasi — Politeknik Negeri Malang  
**Tahun Akademik:** 2025/2026 (Semester 5)

---

## 1. Identitas Mahasiswa & Informasi Akademik
- **Nama Mahasiswa:** Wahyudi Satriawan Hamid
- **NIM:** 244107020137
- **Kelas:** TI 3H
- **Program Studi:** D-IV Teknik Informatika
- **Mata Kuliah:** Pembelajaran Mesin (Machine Learning)
- **Repositori GitHub:** [`https://github.com/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin`](https://github.com/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin)

---

## 2. Struktur Repositori Dua Versi
Repositori ini mengadopsi arsitektur terpisah dua versi untuk memastikan kepatuhan akademik terhadap silabus praktikum standar Polinema sekaligus membuktikan kompetensi adaptasi pemodelan industri pada studi kasus nyata **PBL RumahPoint**:

```text
├── Jobsheet Sesuai Modul/              # VERSI 1: Arsip Praktikum Standar (Dataset Bawaan Modul)
│   ├── JS02.ipynb                      # Titanic, Dummy Blobs, Lena Image (OpenCV)
│   ├── JS03.ipynb                      # Titanic, Lena Image (Pillow), Dummy Feature Selection
│   ├── JS04.ipynb                      # Iris, Mall Customers, Dummy Blobs K-Means
│   ├── JS05.ipynb                      # Blobs, Moons, Circles (Hierarki & DBSCAN)
│   ├── JS06.ipynb                      # Salary Dataset & Boston Housing (Regresi Linier & SVR)
│   ├── JS09.ipynb                      # Iris & SMS Spam (k-NN & Naive Bayes NLP)
│   ├── JS11.ipynb                      # Dummy Non-Linier, LFW Faces, & Day/Night HSV (SVM)
│   └── TG1_...ipynb                    # Tugas 1 Vektor Similaritas / ANN
│
├── Jobsheet Dataset PBL/               # VERSI 2: Implementasi Proyek PBL RumahPoint (Dataset Riil)
│   ├── JS02.ipynb                      # Eksplorasi Data Mentah, Outlier Filtering, & Pembersihan
│   ├── JS03.ipynb                      # Rekayasa Fitur Domain, VIF Scratch, & ANOVA F-Score
│   ├── JS04.ipynb                      # K-Means Clustering Spasial-Fisik, Elbow, & Silhouette
│   ├── JS05.ipynb                      # DBSCAN Spasial, Hierarchical Linkage, & Native HDBSCAN
│   ├── JS06_Regresi.ipynb              # Multiple Linear, Ridge OLS, & SVR RBF (Metrik Rupiah)
│   ├── JS06_ANN.ipynb                  # Approximate Nearest Neighbors (FAISS & HNSWLIB)
│   ├── JS09.ipynb                      # k-NN Optimal Tuning K & Multi-Varian Naive Bayes Tabular
│   ├── JS11.ipynb                      # Support Vector Machine (Linear, RBF, Poly) & GridSearchCV
│   ├── JS13.ipynb                      # Artificial Neural Network (MLP 3-Class Softmax & Regresi)
│   ├── JS14.ipynb                      # Convolutional Neural Network (Cat vs Dog & VGG-mini CIFAR)
│   ├── JS15.ipynb                      # Pipeline MLOps & Ekspor Artefak Produksi
│   └── JS15_Deploy_RumahPoint/         # Web Service Flask API & Dockerfile Hugging Face Spaces
│
├── Kuis 1/                             # Evaluasi Tengah Semester: Komparasi 9 Metode Clustering
│   ├── dataset_properti.csv            # Subset Data Evaluasi Kuis
│   ├── KMeansEuclidean.ipynb           # K-Means dengan Jarak Euclidean
│   ├── KMeansManhattan.ipynb           # K-Means dengan Jarak Manhattan
│   ├── KMeansCosine.ipynb              # K-Means dengan Jarak Cosine
│   ├── HierarchicalEuclidean.ipynb     # Agglomerative dengan Jarak Euclidean
│   ├── HierarchicalManhattan.ipynb     # Agglomerative dengan Jarak Manhattan
│   ├── HierarchicalCosine.ipynb        # Agglomerative dengan Jarak Cosine
│   ├── DBScanEuclidean.ipynb           # DBSCAN dengan Jarak Euclidean
│   ├── DBScanManhattan.ipynb           # DBSCAN dengan Jarak Manhattan
│   ├── DBScanCosine.ipynb              # DBSCAN dengan Jarak Cosine
│   ├── Perbandingan_KMeans_DBScan.ipynb # Sintesis & Analisis Komparatif 9 Kombinasi
│   └── hasil_perbandingan_9_metode.csv # Rekapitulasi Metrik Evaluasi Kuis 1
│
└── Dataset/                            # Manajemen Dataset Proyek PBL
    ├── properti_multi_kota.csv         # Data Mentah Scraper (64.650 baris, 8 kolom)
    └── properti_multi_kota_clean.csv   # Data Bersih Terverifikasi (56.170 baris, 12 kolom)
```

---

## 3. Tabel Indeks Lengkap Modul Pembelajaran Mesin (`JS01` s.d. `JS15`)

Tabel di bawah ini merangkum seluruh modul pembelajaran mesin beserta tautan langsung ke notebook Versi 1 (bawaan modul) dan Versi 2 (adaptasi PBL), badge *Open in Colab*, serta justifikasi adaptasinya:

| Modul | Topik Utama | Versi 1 (Dataset Bawaan) | Versi 2 (Dataset PBL) | Colab (Versi PBL) | Catatan Adaptasi & Metodologi |
|:---|:---|:---:|:---:|:---:|:---|
| **JS01** | Pengantar & Konsep Dasar Machine Learning | [TG1.ipynb](Jobsheet%20Sesuai%20Modul/TG1_244107020137_WAHYUDI_SATRIAWAN_H.ipynb)<br>([GitBook Polinema](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js01-pengenalan-pembelajaran-mesin)) | *(Teori / Konseptual)* | — | Modul fondasi teoretis (paradigma Supervised, Unsupervised, & Reinforcement Learning). Cukup 1 versi dokumen tugas tanpa notebook dataset. |
| **JS02** | Pra-Pengolahan Data (*Data Preprocessing*) | [JS02.ipynb](Jobsheet%20Sesuai%20Modul/JS02.ipynb) | [JS02.ipynb](Jobsheet%20Dataset%20PBL/JS02.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS02.ipynb) | Pembersihan 64.650 baris mentah: deduplikasi 7.156 baris, pembuangan 1.324 outlier ekstrem, kalkulasi umur bangunan (tahun acuan 2026), penyimpanan data bersih 56.170 baris. Pengecualian sah citra `lena.jpg`. |
| **JS03** | Rekayasa Fitur & Seleksi Fitur (*Feature Engineering*) | [JS03.ipynb](Jobsheet%20Sesuai%20Modul/JS03.ipynb) | [JS03.ipynb](Jobsheet%20Dataset%20PBL/JS03.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS03.ipynb) | 6 Fitur domain properti (`is_komersil`, `rasio_bangunan_tanah`, `luas_per_kamar`, dll.), VIF OLS scratch (eksklusi `total_ruangan`), ANOVA F-Score, dan Mutual Information. Target 3 kelas seimbang berbasis kuantil. |
| **JS04** | Klasterisasi I: K-Means & Evaluasi Klaster | [JS04.ipynb](Jobsheet%20Sesuai%20Modul/JS04.ipynb) | [JS04.ipynb](Jobsheet%20Dataset%20PBL/JS04.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS04.ipynb) | Segmentasi pasar berbasis 5 fitur jarak fisik + `log1p(harga)`, mitigasi OOM Silhouette via `sample_size=5000`, Elbow WCSS, Davies-Bouldin Index, dan profiling sebaran geografis klaster tanpa sparse OHE lokasi. |
| **JS05** | Klasterisasi II: DBSCAN, Hierarki, & HDBSCAN | [JS05.ipynb](Jobsheet%20Sesuai%20Modul/JS05.ipynb) | [JS05.ipynb](Jobsheet%20Dataset%20PBL/JS05.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS05.ipynb) | Penentuan `eps` via k-Distance knee graph, subsampel A (3.000) untuk dendrogram hierarki, subsampel B (10.000) untuk DBSCAN & native `sklearn.cluster.HDBSCAN`, komparasi adil pada subsampel A. |
| **JS06 (Regresi)** | Regresi Linier, Regularisasi Ridge, & SVR RBF | [JS06.ipynb](Jobsheet%20Sesuai%20Modul/JS06.ipynb) | [JS06_Regresi.ipynb](Jobsheet%20Dataset%20PBL/JS06_Regresi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS06_Regresi.ipynb) | Transformasi target $\log(1+y)$ untuk skewness harga, pelaporan metrik bisnis Rupiah via $\exp(y)-1$, stabilisasi multikolinearitas dengan Ridge L2, subsampel 5.000 baris untuk SVR RBF $\mathcal{O}(N^2)$. |
| **JS06 (ANN)** | *Approximate Nearest Neighbors* (Vektor Properti) | [TG1_ANN.ipynb](Jobsheet%20Sesuai%20Modul/TG1_244107020137_WAHYUDI_SATRIAWAN_H.ipynb) | [JS06_ANN.ipynb](Jobsheet%20Dataset%20PBL/JS06_ANN.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS06_ANN.ipynb) | Korpus indeks 55.670 properti dan 500 kueri uji terisolasi. Penyelarasan memori `float32` C++, indeks kemiripan properti via FAISS (`IndexFlatL2`, `IndexIVFFlat`) dan HNSWLIB (graf HNSW). Akselerasi 10x–50x. |
| **JS09** | Klasifikasi I: k-NN & Multi-Varian Naive Bayes | [JS09.ipynb](Jobsheet%20Sesuai%20Modul/JS09.ipynb) | [JS09.ipynb](Jobsheet%20Dataset%20PBL/JS09.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS09.ipynb) | Target 3 kelas seimbang `kelas_harga`. Tuning nilai $K$ via ANOVA F-Score pada 10.000 subsample dilanjutkan refit 44.936 train. Pengecualian sah SMS Spam NLP; komparasi GaussianNB, Yeo-Johnson, & MultinomialNB. |
| **JS11** | Klasifikasi II: Support Vector Machine (SVM) | [JS11.ipynb](Jobsheet%20Sesuai%20Modul/JS11.ipynb) | [JS11.ipynb](Jobsheet%20Dataset%20PBL/JS11.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS11.ipynb) | Pengecualian sah citra LFW Faces & Day/Night. Strategi 2-skala: `LinearSVC` $\mathcal{O}(N)$ pada 44.936 train penuh vs Kernel `SVC` + GridSearchCV pada 8.000 subsample. Proyeksi 2D PCA margin visual. Evaluasi bersama pada 11.234 test. |
| **JS13** | Jaringan Saraf Tiruan (*Artificial Neural Network*) | [GitBook JS13](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js13-artificial-neural-network-ann-dan-evaluasi-classifier) | [JS13.ipynb](Jobsheet%20Dataset%20PBL/JS13.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS13.ipynb) | Forward/backprop XOR 2-layer scratch NumPy. Penggantian Iris & Boston Housing ke Deep MLP Softmax 3-kelas & MLP Regresi `log1p(harga)` pada 56.170 baris dengan EarlyStopping. Benchmark klasifikasi MNIST 784D. |
| **JS14** | *Convolutional Neural Network* (CNN Citra) | [GitBook JS14](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js14-convolutional-neural-network) | [JS14.ipynb](Jobsheet%20Dataset%20PBL/JS14.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS14.ipynb) | **Pengecualian Sah Data Tabular:** Data properti tabular tidak memiliki korelasi spasial piksel 2D. Mempertahankan domain citra: Cat vs Dog Keras 3 pipeline augmentasi, arsitektur VGG-mini CIFAR-10 (> 80% akurasi), dan MNIST. |
| **JS15** | *Pipeline Machine Learning & Cloud Deployment* | [GitBook JS15](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js15-pipeline-machine-learning-and-deployment) | [JS15.ipynb](Jobsheet%20Dataset%20PBL/JS15.ipynb)<br>+ [Folder Deploy](Jobsheet%20Dataset%20PBL/JS15_Deploy_RumahPoint/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wahyudi-Satriawan/244107020137_TI3H_PembelajaranMesin/blob/main/Jobsheet%20Dataset%20PBL/JS15.ipynb) | Pipeline MLOps end-to-end: ekspor 3 artefak (`preprocessor.joblib`, `model_rumahpoint.keras`, `metadata.json`), anti training-serving skew, web service Flask API produksi, container Dockerfile di Hugging Face Spaces. |