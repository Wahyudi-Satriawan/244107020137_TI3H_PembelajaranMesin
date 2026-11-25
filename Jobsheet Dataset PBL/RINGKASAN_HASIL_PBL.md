# Ringkasan Hasil Praktikum PBL RumahPoint (`JS02`–`JS15`)

## 1. Identitas & Ringkasan Dataset PBL (RumahPoint)

### Identitas Akademik
- **Mahasiswa:** Wahyudi Satriawan Hamid
- **NIM:** 244107020137
- **Kelas:** TI-3H
- **Program Studi:** D-IV Teknik Informatika — Jurusan Teknologi Informasi — Politeknik Negeri Malang
- **Mata Kuliah:** Pembelajaran Mesin (Semester 5, Tahun Akademik 2025/2026)
- **Studi Kasus:** Proyek PBL RumahPoint (Prediksi Harga, Segmentasi Pasar, dan Pencarian Kemiripan Properti Multi-Kota Indonesia)

### Profil Dataset
- **Data Mentah (`properti_multi_kota.csv`):** 64.650 baris dan 8 kolom (`sumber`, `lokasi`, `luas_m2`, `luas_bangunan`, `jumlah_kamar`, `jumlah_kamar_mandi`, `usia_bangunan`, `harga`) yang mencakup 595 entri `lokasi` dan 513 `sumber`.
- **Alur Pembersihan Data (`JS02`):**
  1. Deduplikasi baris identik menghapus 7.156 baris (11,07%), menyisakan 57.494 baris.
  2. Penyaringan batas kewajaran domain fisik dan finansial (`10 <= luas_m2 <= 2.000`, `luas_bangunan` kosong atau `10..2.000`, `Rp 50 Juta <= harga <= Rp 100 Miliar`, `jumlah_kamar` dan `jumlah_kamar_mandi` kosong atau `<= 20`) menghapus 1.324 baris (2,05%), menghasilkan **56.170 baris** valid (86,88% dari data mentah).
- **Data Bersih (`properti_multi_kota_clean.csv`):** 56.170 baris dan 12 kolom (8 kolom awal ditambah `kondisi`, `lokasi_bersih`, `kelas_harga`, dan `kelas_harga_biner`). Nilai hilang pada kolom numerik tetap dibiarkan (`NaN`) agar proses imputasi dilakukan setelah pemisahan data latih-uji di dalam `Pipeline`.
- **Target Pemodelan:**
  - **Regresi (`harga`):** rentang Rp 50.000.000 hingga Rp 100.000.000.000 (median Rp 750.000.000). Karena distribusi harga sangat miring ke kanan (*right-skewed*), pelatihan regresi menggunakan transformasi logaritmik `np.log1p(harga)` dan evaluasi skala mata uang dilakukan setelah transformasi balik `np.expm1()`.
  - **Klasifikasi (`kelas_harga`):** 3 kelas seimbang berbasis tertil kuantil, yaitu `Murah` (< Rp 483.819.314; 18.723 baris / 33,33%), `Sedang` (Rp 483.819.314–Rp 1.235.000.000; 18.727 baris / 33,34%), dan `Mahal` (> Rp 1.235.000.000; 18.720 baris / 33,33%).
- **Keterbatasan Data:**
  1. **`usia_bangunan` >92% kosong:** pada data mentah hilang 92,59% (59.861 baris) dan setelah normalisasi tahun acuan 2026 pada data bersih tetap hilang 92,29% (51.838 baris). Informasi yang tersedia diturunkan menjadi kolom kategorikal `kondisi` (`Tidak diketahui`: 51.838, `Bekas`: 3.415, `Baru`: 917).
  2. **588 variasi ejaan `lokasi_bersih`:** penulisan wilayah masih tidak seragam (misalnya `Bekasi` terpisah dari `Kabupaten Bekasi`, `Tangerang` dari `Kabupaten Tangerang`, atau ketidakkonsistenan prefiks `Kota` saat inferensi). Pemangkasan ke Top-15 lokasi hanya mencakup 39,41% populasi, sedangkan ~60,59% sisanya masuk ke kategori `Lainnya`.
  3. **4,01% properti komersial (`jumlah_kamar == 0`):** sebanyak 2.252 baris pada data bersih tercatat memiliki 0 kamar tidur dengan median harga Rp 1.337.325.129,50 (1,82 kali lipat median hunian biasa sebesar Rp 735.000.000,00). Baris ini merepresentasikan aset komersial (ruko, kantor, atau lahan) dan direpresentasikan melalui fitur biner `is_komersil`.

---

## 2. Rekapitulasi Hasil per Jobsheet (`JS02` s.d. `JS15`)

| Jobsheet | Fokus Metode | Dataset / Sampel | Metrik Utama (Output Aktual) | Temuan Singkat |
|:---|:---|:---|:---|:---|
| `JS02` | EDA, deteksi anomali, pembersihan data, demo pra-proses | `properti_multi_kota.csv` (64.650 baris -> 56.170 baris bersih) | Duplikat: 7.156 (11,07%); outlier: 1.324 (2,05%); korelasi `log1p_harga`: `luas_bangunan` ($r=0{,}6359$), `jumlah_kamar_mandi` ($r=0{,}6138$), `luas_m2` ($r=0{,}4932$) | Pembersihan menghasilkan 3 kelas kuantil seimbang dan memvalidasi retensi 2.252 properti komersial (`jumlah_kamar=0`). |
| `JS03` | Rekayasa fitur, VIF, ANOVA/MI/RF importance, 5-Fold CV | `properti_multi_kota.csv` -> 56.170 bersih (44.936 train / 11.234 test) | VIF `total_ruangan` = 874,66 (dibuang). 5-Fold CV Skenario A -> B -> C: LogReg Acc 0,6555 -> 0,7115 -> 0,6891; RF (`depth=10`) Acc 0,6982 -> 0,7644 -> 0,7580 | Rekayasa fitur rasio dan sinyal lokasi menaikkan akurasi LogReg +5,60% dan Random Forest +6,62% pada Skenario B. |
| `JS04` | Klasterisasi K-Means & DBSCAN + PCA 2D | `properti_multi_kota_clean.csv` (56.170 penuh; 10.000 subsampel untuk DBSCAN & komparasi) | PCA 2D variansi: 80,88%; K-Means penuh $K=2$: Sil 0,5661 (N=5k), DBI 0,9994, ARI 0,1072; $K=3$: Sil 0,4359, ARI 0,3154; DBSCAN 10k ($\varepsilon=0{,}5652, ms=30$): 13 klaster, noise 16,93%, Sil 0,2675, DBI 1,6131, ARI 0,2096 | K-Means cepat mempartisi seluruh data tetapi terpengaruh outlier; DBSCAN mengisolasi properti mewah/komersial (median Rp 2,73 M) sebagai noise. |
| `JS05` | Klasterisasi Hierarki (Agglomerative) & HDBSCAN | `properti_multi_kota_clean.csv` (Sub-A: 3.000; Sub-B: 10.000) | Sub-B HDBSCAN (`mcs=350, ms=50`): 4 klaster, noise 24,69%, Sil 0,3016, DBI 1,3679; Sub-A Agglo (`average, euclidean, K=3`): Sil 0,7954, DBI 0,4894, ARI 0,0000; Sub-A Ward ($K=4$): Sil 0,4215, DBI 1,0636, ARI 0,2607; Sub-A HDBSCAN (`mcs=50, ms=15`): 8 klaster, noise 25,30%, Sil 0,2816, ARI 0,2190 | HDBSCAN memisahkan 4 segmen padat multi-kepadatan dan 24,69% outlier tanpa parameter $\varepsilon$; linkage `average` Euclidean terkena *chaining* (ARI 0,0000). |
| `JS06_Regresi` | Simple Linear, Multiple OLS, Ridge, & SVR RBF | `properti_multi_kota_clean.csv` (44.936 train / 11.234 test; SVR pada 4.000 train / 1.000 test) | Test penuh: SLR $R^2_{\log}=0{,}4039$; OLS/Ridge $R^2_{\log}=0{,}5640$, MAE Rp 2,46 M, MAPE 92,61%–92,62%. Subsampel (4k/1k): SVR RBF ($C=1{,}0$) $R^2_{\log}=0{,}7138$, $R^2_{\text{Rp}}=0{,}2764$, MAE Rp 882,54 jt, MAPE 47,98% | Penambahan fitur multivariat dan lokasi menaikkan $R^2_{\log}$ dibanding SLR; SVR RBF unggul karena mampu memodelkan hubungan non-linear harga properti. |
| `JS06_ANN` | Approximate Nearest Neighbors (Exact NN, Annoy, FAISS, HNSW) | Sintetis (hingga 100k x 5D & 20k x 32D) + `properti_multi_kota_clean.csv` (55.670 indeks / 500 kueri, 5D `float32`) | PBL 55.670 vektor (Top-10): Exact 0,2589 ms (3.862 QPS, Recall 100%); Annoy 0,1003 ms (9.968 QPS, Recall 99,24%); FAISS IVFFlat 0,1578 ms (6.336 QPS, Recall 99,58%); HNSW 0,0178 ms (56.290 QPS, Recall 98,02%). Mobile PyDroid3: `[Belum ada output eksekusi]` | HNSW menghasilkan throughput tertinggi (~14,6x Exact NN) dengan Recall@10 98,02% serta kecocokan Top-5 100% pada uji kueri sampel properti Depok. |
| `JS09` | Klasifikasi k-NN (ANOVA + tuning $K$) & Multi-Varian Naive Bayes | `properti_multi_kota_clean.csv` (44.936 train / 11.234 test; CV pada 10.000 subsampel) | k-NN (`K=25, distance`, 27 fitur): Acc 75,63%, Macro F1 0,7541, inferensi 8,748 s; GNB+Yeo-Johnson: Acc 61,32%, Macro F1 0,5709, inferensi 0,061 s; GNB baseline: Acc 58,01%; MultinomialNB: Acc 50,85% | k-NN jauh lebih akurat (+14,31 poin persentase atas NB terbaik), namun waktu inferensinya ~143x lebih lambat dibanding GNB+YJ. |
| `JS11` | Support Vector Machine (Linear, RBF, Poly) & GridSearchCV | `properti_multi_kota_clean.csv` (LinearSVC: 44.936; Kernel SVC: 8.000 subsampel; test: 11.234) | Baseline: LinearSVC 66,94% (F1 0,6480), SVC Linear 67,85%, SVC Poly 68,49%, SVC RBF 69,73%. Tuned RBF (`C=10,0, gamma=0,1`): Acc 71,17%, Macro F1 0,7094 | Kernel RBF mengungguli kernel linear pada batas keputusan non-linear, tetapi ukuran data latih dibatasi 8.000 baris karena kompleksitas $\mathcal{O}(N^2)$. |
| `JS13` | ANN Scratch (XOR), MLP Klasifikasi & Regresi PBL, MLP MNIST | XOR (4 titik); PBL (44.936 train / 11.234 test); MNIST (60k train / 10k test) | XOR 3-ReLU MSE 0,000353; PBL Klasifikasi Model 4 (`128-64-32 + Dropout 0,2`): Acc 73,93%, Macro F1 0,7374, ROC-AUC 0,8957; PBL Regresi (`64-32 ReLU, lr=0,01`): $R^2_{\log}=0{,}7056$, MAE Rp 802,45 jt, MAPE 49,76%; MNIST MLP Terbaik: 97,88% | MLP mampu menskalakan pemodelan non-linear pada 44.936 data penuh baik untuk klasifikasi maupun regresi log-harga properti. |
| `JS14` | CNN Klasifikasi Citra (Cat vs Dog, CIFAR-10, MNIST) | Cat vs Dog (2k/1k); CIFAR-10 (50k/10k, split 45k/5k); MNIST (60k/10k, split 54k/6k) | Cat vs Dog: Val Acc 74,20%; CIFAR-10 Baseline: 68,88% -> VGG-mini Modifikasi: 85,48% (Macro F1 0,8525); MNIST CNN: 99,31% (Macro F1 0,9930) | Penggantian arsitektur tunggal ke VGG-mini + BN + Dropout + Augmentasi + ReduceLROnPlateau menaikkan akurasi CIFAR-10 sebesar +16,60%. |
| `JS15` | Pipeline MLOps Keras 3, Serialisasi Artefak, & Verifikasi Inferensi | `properti_multi_kota_clean.csv` (39.319 train / 8.425 val / 8.426 test) | Test Acc 74,23%, Test Loss 0,5818, Macro F1 0,7390, Macro ROC-AUC 0,8957; 3 artefak diekspor (total ~99,6 KB); Flask/Docker smoke test: `[Belum ada output eksekusi]` | Pipeline `ColumnTransformer` dan model `.keras` terverifikasi konsisten pada 3 skenario uji input mentah dari disk. |

### Catatan Khusus `JS14` dan `JS15`
- **`JS14` (Domain Citra):** Seluruh eksperimen pada `JS14` menggunakan dataset citra (`cats_and_dogs_filtered`, CIFAR-10, dan MNIST) karena lapisan konvolusi 2D memerlukan struktur grid spasial antar piksel yang tidak dimiliki oleh data tabular properti RumahPoint. Hasil utama memperlihatkan peningkatan akurasi CIFAR-10 dari 68,88% menjadi 85,48% serta peningkatan akurasi MNIST menjadi 99,31%.
- **`JS15` (Artefak Pipeline & Deployment):** Notebook mengeksekusi pelatihan model `64-32 + BatchNormalization + ReLU + Dropout(0,2)` (4.099 parameter) dengan `EarlyStopping` (bobot terbaik pada epoch ke-57), mengekspor `model_rumahpoint.keras` (94.032 bytes), `preprocessor.joblib` (4.138 bytes), dan `metadata.json` (1.398 bytes), serta menjalankan uji inferensi 3 skenario mentah (`Murah` 87,72%, `Mahal` 49,89%, dan properti komersial `Mahal` 83,08%). Untuk implementasi dan *smoke test* web service Flask serta kontainer Docker, berkas maupun output eksekusinya belum tersedia di repositori (`[Belum ada output eksekusi]`).

---

## 3. Evaluasi Komparatif Metode: Performa Terbaik vs Kurang Optimal

### A. Klasifikasi `kelas_harga` (`JS03`, `JS09`, `JS11`, `JS13`, `JS15`)

| Sumber | Model / Konfigurasi | Skala Evaluasi | Akurasi | Macro F1 | Macro ROC-AUC (OvR) | Waktu Latih / Inferensi |
|:---|:---|:---|---:|---:|---:|:---|
| `JS03` | Random Forest (`depth=10`, Skenario B) | 5-Fold CV (44.936 train) | 76,44% | 0,7631 | — | — |
| `JS03` | Random Forest (`depth=10`, Skenario C) | 5-Fold CV (44.936 train) | 75,80% | 0,7569 | — | — |
| `JS03` | Logistic Regression (Skenario B) | 5-Fold CV (44.936 train) | 71,15% | 0,7103 | — | — |
| `JS09` | **k-NN Optimal (`K=25, distance`, 27 fitur)** | Test 20% (11.234) | **75,63%** | **0,7541** | 0,3117* | Latih 0,239 s / Pred 8,748 s |
| `JS09` | GaussianNB + Yeo-Johnson | Test 20% (11.234) | 61,32% | 0,5709 | 0,3924* | Latih 0,531 s / Pred 0,061 s |
| `JS09` | GaussianNB Baseline (`StandardScaler`) | Test 20% (11.234) | 58,01% | 0,5539 | 0,4164* | Latih 0,174 s / Pred 0,055 s |
| `JS09` | MultinomialNB (`MinMaxScaler`) | Test 20% (11.234) | 50,85% | 0,4851 | 0,3576* | Latih 0,204 s / Pred 0,064 s |
| `JS11` | SVM RBF Tuned (`C=10,0, gamma=0,1`, latih 8k) | Test 20% (11.234) | 71,17% | 0,7094 | 0,2877* | GridSearch 77,48 s (36 fit) |
| `JS11` | SVC RBF Baseline (`C=1,0`, latih 8k) | Test 20% (11.234) | 69,73% | 0,6929 | — | Latih 2,670 s / Pred 6,028 s |
| `JS11` | SVC Poly `deg=3` (latih 8k) | Test 20% (11.234) | 68,49% | 0,6849 | — | Latih 4,013 s / Pred 2,494 s |
| `JS11` | SVC Linear (latih 8k) | Test 20% (11.234) | 67,85% | 0,6749 | — | Latih 3,296 s / Pred 2,099 s |
| `JS11` | LinearSVC (latih penuh 44.936) | Test 20% (11.234) | 66,94% | 0,6480 | — | Latih 1,248 s / Pred 0,034 s |
| `JS13` | MLP Deep + Dropout (`128-64-32` ReLU) | Test 20% (11.234) | 73,93% | 0,7374 | **0,8957** | Latih 36,64 s (50 epoch) |
| `JS13` | MLP Baseline (`16-8` ReLU) | Test 20% (11.234) | 73,55% | — | — | Latih 22,63 s (50 epoch) |
| `JS13` | MLP Sigmoid (`64-32` Sigmoid) | Test 20% (11.234) | 66,09% | — | — | Latih 5,31 s (10 epoch) |
| `JS15` | **MLP Produksi (`64-32` + BN + Dropout 0,2)** | Test 15% (8.426) | **74,23%** | **0,7390** | **0,8957** | 60 epoch (best epoch 57) |

*\*Catatan metrik ROC-AUC pada `JS09` dan `JS11`: nilai aktual tercetak di bawah 0,50 (0,2877–0,4164) disebabkan ketidakcocokan urutan kolom antara `label_binarize(..., classes=['Murah', 'Sedang', 'Mahal'])` dengan urutan alfabetis `predict_proba` (`['Mahal', 'Murah', 'Sedang']`). Pada `JS13` dan `JS15`, label dipetakan ke integer `0/1/2` sehingga Macro ROC-AUC terbaca akurat sebesar 0,8957.*

- **Metode Terbaik:** Pada pengujian *hold-out* 20% yang identik, **`k-NN (K=25, distance)` (`JS09`)** meraih akurasi dan Macro F1 tertinggi (**75,63% / 0,7541**), sedangkan secara 5-Fold CV di `JS03`, **`Random Forest` (`depth=10`, Skenario B)** mencatat rata-rata **76,44% / 0,7631**. Untuk kebutuhan sistem produksi, **`MLP Keras 3 + BatchNorm + Dropout` (`JS15`)** menawarkan kompromi terbaik karena mendekati akurasi teratas (**74,23%**, Macro F1 **0,7390**, ROC-AUC **0,8957**) dengan ukuran model hanya 94 KB dan inferensi cepat tanpa menyimpan 44.936 baris latih seperti k-NN (yang memerlukan 8,748 detik untuk 11.234 sampel uji).
- **Metode Kurang Optimal:** Seluruh varian **`Naive Bayes` (`JS09`)** dan model linear sederhana (**`LinearSVC`** 66,94%, **`MLP Sigmoid`** 66,09%) tertinggal signifikan. `MultinomialNB` hanya mencapai **50,85%** (F1 0,4851) karena asumsi frekuensi diskrit tidak sesuai untuk fitur kontinu terstandarisasi $[0,1]$. `GaussianNB` baseline tertahan di **58,01%** karena fitur luas fisik sangat *right-skewed* dan asumsi independensi kondisional antar fitur terlanggar oleh korelasi erat antara `luas_bangunan`, `luas_m2`, `jumlah_kamar`, `jumlah_kamar_mandi`, dan rasio turunannya; transformasi Yeo-Johnson memperbaiki akurasi menjadi **61,32%**, tetapi kelas `Sedang` tetap runtuh dengan F1 hanya 0,2955. Pada seluruh model terbaik sekalipun, kelas `Sedang` selalu menjadi kelas tersulit (F1 0,6144–0,6439) karena tumpang tindih batas kuantil antar-kota dengan fitur saat ini.

---

### B. Regresi Prediksi `harga` (`JS06_Regresi`, `JS13`)

| Sumber | Model | Skala Data (Train / Test) | $R^2$ (Log) | $R^2$ (Rp) | MAE (Rp) | RMSE (Rp) | MAPE (%) |
|:---|:---|:---|---:|---:|---:|---:|---:|
| `JS06_Regresi` | Simple Linear (`luas_bangunan`) | Penuh (44.936 / 11.234) | 0,4039 | -478,0536 | Rp 3,39 M | Rp 77,43 M | 152,57% |
| `JS06_Regresi` | Multiple Linear (OLS) | Penuh (44.936 / 11.234) | 0,5640 | -287,3326 | Rp 2,46 M | Rp 60,07 M | 92,61% |
| `JS06_Regresi` | Ridge Regression (`alpha=1,0`) | Penuh (44.936 / 11.234) | 0,5640 | -287,3733 | Rp 2,46 M | Rp 60,08 M | 92,62% |
| `JS13` | **ANN MLP (`64-32 ReLU, lr=0,01`)** | Penuh (44.936 / 11.234) | **0,7056** | **-0,0067** | **Rp 802,45 Jt** | **Rp 3,55 M** | **49,76%** |
| `JS06_Regresi` | Simple Linear (`luas_bangunan`) | Subsampel (4.000 / 1.000) | 0,4406 | -0,3494 | Rp 1,32 M | Rp 5,35 M | 77,22% |
| `JS06_Regresi` | Multiple Linear (OLS) | Subsampel (4.000 / 1.000) | 0,6138 | -0,7367 | Rp 1,27 M | Rp 6,07 M | 58,48% |
| `JS06_Regresi` | Ridge Regression (`alpha=1,0`) | Subsampel (4.000 / 1.000) | 0,6138 | -0,7389 | Rp 1,27 M | Rp 6,07 M | 58,49% |
| `JS06_Regresi` | **SVR RBF (`C=1,0, eps=0,1`)** | Subsampel (4.000 / 1.000) | **0,7138** | **0,2764** | **Rp 882,54 Jt** | **Rp 3,92 M** | **47,98%** |
| `JS06_Regresi` | SVR RBF (`C=10,0, eps=0,1`) | Subsampel (4.000 / 1.000) | 0,7054 | 0,2607 | Rp 921,89 Jt | Rp 3,96 M | 48,09% |

*(Pada eksperimen variasi learning rate `JS13`, `Adam lr=0,0001` mencapai Test MSE Log 0,3409 dan `Adam lr=0,01` terbaik dengan Test MSE Log 0,3136; `Adam lr=0,001` tercatat mengalami penghentian dini pada epoch ke-10 dengan Test MSE Log 11,4069 akibat penggunaan ulang objek callback `EarlyStopping` di dalam loop.)*

- **Metode Terbaik vs Terlemah:** Model non-linear (**`ANN MLP`** pada data penuh dengan $R^2_{\log}=0{,}7056$, MAE Rp 802,45 juta, MAPE 49,76%; serta **`SVR RBF`** pada subsampel dengan $R^2_{\log}=0{,}7138$, $R^2_{\text{Rp}}=0{,}2764$, MAPE 47,98%) mengungguli seluruh model linear secara telak. `Simple Linear Regression` menjadi metode terlemah ($R^2_{\log}=0{,}4039$, MAPE 152,57%) karena mengabaikan lokasi, jumlah kamar, dan tipe komersial. Penambahan fitur pada `Multiple OLS` / `Ridge` menaikkan $R^2_{\log}$ menjadi 0,5640, tetapi karena harga properti tidak bertumbuh secara linear terhadap luas pada ruang log, ekstrapolasi garis lurus pada properti berukuran besar mengalami lonjakan galat eksponensial setelah diinversi via `expm1()`, menyebabkan $R^2_{\text{Rp}}$ bernilai sangat negatif (-478,05 dan -287,37) dengan RMSE di atas Rp 60 miliar. `SVR RBF` dan `ANN MLP` mampu meredam deviasi ekstrem tersebut, dengan keunggulan praktis pada `ANN MLP` yang dapat dilatih pada seluruh 44.936 baris tanpa hambatan kompleksitas kuadratik $\mathcal{O}(N^2)$ milik SVR.

---

### C. Klasterisasi Segmentasi Properti (`JS04`, `JS05`)

| Sumber | Algoritma & Konfigurasi | Ukuran Sampel | Klaster | Noise (%) | Silhouette | DBI | ARI vs `kelas_harga` |
|:---|:---|:---|---:|---:|---:|---:|---:|
| `JS04` | K-Means ($K=2$) | 56.170 (Sil: 5.000) | 2 | 0,00% | 0,5661 | 0,9994 | 0,1072 |
| `JS04` | K-Means ($K=3$) | 56.170 (Sil: 5.000) | 3 | 0,00% | 0,4359 | 1,0663 | 0,3154 |
| `JS04` | K-Means ($K=2$) | 10.000 (identik DBSCAN) | 2 | 0,00% | 0,5574 | 0,9953 | 0,1158 |
| `JS04` | DBSCAN ($\varepsilon=0{,}5652, ms=30$) | 10.000 | 13 | 16,93% | 0,2675 | 1,6131 | 0,2096 |
| `JS05` | Agglomerative (`average, euclidean, K=3`) | 3.000 (Sub-A) | 3 | 0,00% | 0,7954 | 0,4894 | 0,0000 |
| `JS05` | Agglomerative (`ward, euclidean, K=4`) | 3.000 (Sub-A) | 4 | 0,00% | 0,4215 | 1,0636 | 0,2607 |
| `JS05` | Agglomerative (`average, cosine, K=3`) | 3.000 (Sub-A) | 3 | 0,00% | 0,2336 | 1,2380 | 0,3736 |
| `JS05` | DBSCAN ($\varepsilon=1{,}028, ms=10$) | 3.000 (Sub-A) | 1 | 6,73% | NaN | NaN | 0,0124 |
| `JS05` | HDBSCAN (`mcs=50, ms=15`) | 3.000 (Sub-A) | 8 | 25,30% | 0,2816 | 1,4840 | 0,2190 |
| `JS05` | **HDBSCAN (`mcs=350, ms=50`)** | 10.000 (Sub-B) | **4** | **24,69%** | 0,3016 | 1,3679 | — |

- **Analisis Segmentasi:**
  - **K-Means** unggul dari sisi skalabilitas (0,41–0,61 detik pada 56.170 baris) dan memberikan partisi makro yang rapi pada $K=3$ (ARI 0,3154) atau $K=2$ (Silhouette 0,5661), namun karena memaksa 100% titik masuk ke klaster, properti ekstrem menggeser posisi centroid.
  - **Agglomerative `average + euclidean` ($K=3$)** mencatat Silhouette (0,7954) dan DBI (0,4894) tertinggi secara numerik, tetapi **ARI = 0,0000** membuktikan partisi tersebut gagal memisahkan segmen pasar karena efek *chaining* hanya mengisolasi segelintir outlier ekstrem sementara hampir seluruh populasi menyatu ke dalam satu klaster besar. Sebaliknya, `ward + euclidean` ($K=4$, ARI 0,2607) dan `average + cosine` ($K=3$, ARI 0,3736) membentuk segmen yang jauh lebih seimbang, walaupun tetap terbatas kompleksitas $\mathcal{O}(N^2)$ memori dan $\mathcal{O}(N^3)$ waktu.
  - **DBSCAN** kurang cocok untuk data properti multi-kota karena menggunakan satu radius kepadatan global $\varepsilon$: pada Sub-A ($\varepsilon=1{,}028$), pasar metropolitan dan pinggiran melebur menjadi 1 klaster tunggal (ARI 0,0124), sedangkan pada 10.000 sampel ($\varepsilon=0{,}5652$) menghasilkan 13 klaster terfragmentasi.
  - **HDBSCAN (`JS05` Sub-B)** memberikan segmentasi berbasis kepadatan yang paling informatif: tanpa memerlukan $\varepsilon$, algoritma ini mengekstrak 4 klaster inti pasar yang terbedakan jelas (Klaster 1 ekonomis median Rp 396,5 jt; Klaster 2 menengah-bawah Rp 800 jt; Klaster 3 menengah Rp 1,00 M; Klaster 0 premium Rp 1,78 M) sekaligus mengisolasi 24,69% properti mewah/komersial (median Rp 2,40 M, luas tanah 221 m², 11,9% komersial) ke label noise `-1`.

---

### D. Pencarian Kemiripan Properti / ANN Retrieval (`JS06_ANN`)

Hasil benchmark pada dataset PBL RumahPoint (55.670 vektor korpus 5D `float32`, 500 kueri uji, Top-10):

| Metode | Waktu Build (s) | Total Query 500x (s) | Latensi (ms/kueri) | Throughput (QPS) | Recall@10 (%) |
|:---|---:|---:|---:|---:|---:|
| Exact NN (Brute-Force) | 0,0010 | 0,1295 | 0,2589 | 3.862,1 | 100,00% |
| Spotify Annoy (20 Trees) | 2,9282 | 0,0502 | 0,1003 | 9.967,7 | 99,24% |
| Meta FAISS (`IndexIVFFlat`, `nlist=100, nprobe=15`) | 0,1681 | 0,0789 | 0,1578 | 6.336,2 | 99,58% |
| **HNSW (`hnswlib`, `M=16, ef_c=200, ef=50`)** | 5,1402 | **0,0089** | **0,0178** | **56.290,3** | 98,02% |

- **Trade-Off Kecepatan–Akurasi Terbaik:** **`HNSW (hnswlib)`** unggul jauh pada kecepatan kueri online dengan latensi **0,0178 ms/kueri** (**56.290,3 QPS**, ~14,6x lebih cepat dari Exact NN) dan tetap mempertahankan Recall@10 **98,02%** (serta 5/5 identik dengan Exact NN pada uji kueri sampel properti Depok). Pada benchmark dimensi lebih tinggi (`20.000` vektor 32D, Praktikum 5), HNSW juga konsisten mendominasi kurva Pareto Recall–QPS (Recall@10 97,74% pada 12.170 QPS; 99,80% pada 6.163 QPS) dibanding FAISS-IVF (Recall@10 94,82% pada 7.506 QPS) dan Annoy (Recall@10 84,64% pada 2.839 QPS). Kelemahan HNSW hanya terletak pada *build time* offline yang paling lama (5,1402 s), sehingga apabila sistem menuntut pembangunan ulang indeks secara sangat sering dengan Recall@10 > 99,5%, **FAISS `IndexIVFFlat`** (build 0,1681 s, Recall@10 99,58%) menjadi alternatif terbaik.

---

### E. Perbandingan Deep Learning Citra (`JS13` MLP vs `JS14` CNN)

1. **Peningkatan CIFAR-10 (`JS14`):**
   - **Baseline Awal (3x Single `Conv2D`, 122.570 parameter, 10 epoch, tanpa regularisasi/augmentasi):** Test Accuracy **68,88%**, Test Loss **0,9756** (mengalami *overfitting* dengan akurasi latih 76,70% vs validasi 68,88%).
   - **VGG-mini Modifikasi (3 blok Double `Conv2D` `32->64->128`, `BatchNormalization`, `Dropout` progresif `0,2->0,5`, `Data Augmentation`, `ReduceLROnPlateau` + `EarlyStopping`, 552.874 parameter pada split 45k/5k):** Test Accuracy meningkat menjadi **85,48%** (**+16,60 poin persentase**), Test Loss turun ke **0,4375**, dan Macro F1 mencapai **0,8525**. *(Sebagai catatan tambahan pada tugas klasifikasi biner Cat vs Dog di `JS14`, CNN mencapai akurasi validasi 74,20% pada 1.000 citra).*
2. **Perbandingan MNIST (`JS13` MLP vs `JS14` CNN):**
   - Pada `JS13`, perataan citra 28x28 menjadi vektor 1D 784 dimensi dengan arsitektur MLP menghasilkan Test Accuracy **97,70%** (`256-128 ReLU`), **97,62%** (`256-128 Sigmoid`), dan tertinggi **97,88%** (`256-128-64 ReLU`).
   - Pada `JS14`, CNN 2 blok konvolusi (`32 -> 64` filter + `BatchNormalization` + `Dropout(0,3)`, 422.026 parameter) menembus Test Accuracy **99,31%** (Test Loss **0,0229**, Macro F1 **0,9930**).
   - Selisih +1,43 poin persentase menunjukkan keunggulan *inductive bias* konvolusi 2D dalam mempertahankan ketetanggaan piksel, berbagi bobot filter lokal (*weight sharing*), serta menjaga invarian terhadap pergeseran posisi digit dibanding arsitektur *fully connected* MLP.

---

## 4. Sintesis & Rekomendasi untuk Sistem RumahPoint

Untuk kebutuhan produksi pada sistem **RumahPoint**, arsitektur model sebaiknya dipilah per fungsi layanan berdasarkan keseimbangan akurasi, latensi, dan skalabilitas:
1. **Klasifikasi Kelas Harga (`Murah` / `Sedang` / `Mahal`):** gunakan pipeline `ColumnTransformer` + **`MLP Keras 3 (64-32 + BatchNorm + Dropout 0,2)`** (`JS15`) karena memberikan akurasi kompetitif (74,23%, Macro ROC-AUC 0,8957) dengan jejak artefak sangat ringan (~99,6 KB) dan inferensi seragam anti *training-serving skew*; jika akurasi tabular maksimal diutamakan dibanding ukuran artefak, model berbasis pohon seperti **`Random Forest`** (76,44% CV di `JS03`) atau *gradient boosting* layak dievaluasi pada pipeline serupa.
2. **Estimasi Harga Kontinu (`harga`):** gunakan model non-linear dengan transformasi target `log1p(harga)` seperti **`ANN MLP`** (`JS13`, $R^2_{\log}=0{,}7056$, MAE Rp 802,45 jt, MAPE 49,76%) yang mampu dilatih pada seluruh 44.936 data latih, alih-alih regresi linear atau `SVR RBF` yang dibatasi kompleksitas $\mathcal{O}(N^2)$.
3. **Pencarian Properti Serupa:** gunakan indeks **`HNSW (hnswlib)`** (`56.290,3 QPS`, Recall@10 `98,02%`) untuk rekomendasi real-time berlatensi sub-milidetik, atau **`FAISS IndexIVFFlat`** jika katalog listing sangat sering dibangun ulang.

Peningkatan kualitas prediksi yang paling berdampak ke depan tidak lagi berasal dari penyesuaian arsitektur model semata, melainkan dari perbaikan data input: (1) normalisasi 588 variasi ejaan `lokasi_bersih` ke tingkat kota/kabupaten dan kecamatan baku serta pengayaan koordinat lintang/bujur agar 60,59% data di luar Top-15 tidak kehilangan sinyal lokasi saat masuk kategori `Lainnya` (termasuk memperbaiki inkonsistensi prefiks `Kota`/`Kab.` pada payload API); (2) penambahan fitur pembeda nilai properti seperti legalitas (SHM/HGB), jumlah lantai, daya listrik, serta pemisahan tipe aset (`Rumah`, `Ruko`, `Tanah`) secara eksplisit; serta (3) perbaikan teknis minor pada kode evaluasi dan kelengkapan modul Flask/Docker sebelum rilis layanan.
