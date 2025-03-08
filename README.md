# Laporan Proyek Machine Learning - Istia Budi

## Domain Proyek

Kualitas udara merupakan salah satu faktor krusial yang mempengaruhi kesehatan masyarakat dan lingkungan. Polusi udara, terutama dari PM2.5, PM10, SO2, NO2, CO, dan O3, telah dikaitkan dengan berbagai penyakit pernapasan dan kardiovaskular. Oleh karena itu, kemampuan untuk memprediksi kualitas udara dapat membantu pemerintah dan masyarakat dalam mengambil langkah-langkah preventif.

Dataset Air Quality PRSA Huairou berisi data historis tentang berbagai parameter polusi udara dan kondisi meteorologi dari stasiun pemantauan udara Huairou di Beijing, China. Dengan menggunakan teknik Predictive Analytics, kita dapat membangun model yang mampu memprediksi kualitas udara berdasarkan parameter lingkungan seperti suhu (TEMP), tekanan udara (PRES), kelembaban (DEWP), curah hujan (RAIN), kecepatan angin (WSPM), dan arah angin (wd).

## Business Understanding

Proyek ini bertujuan untuk menganalisis tren polusi udara, mengidentifikasi faktor-faktor yang berkontribusi, serta membangun model prediksi yang dapat memperkirakan tingkat polutan utama di masa depan.

### Problem Statements

- Bagaimana kondisi materi partikulat PM(2.5) yang dimonitoring pada stasiun Huairou.
- Menganalisis dan membandingkan beberapa algoritma Machine Learning untuk menentukan model dengan performa terbaik dalam prediksi polusi udara.
- Apakah ada pola anomali atau pencilan dalam data kualitas udara yang dapat mengindikasikan kejadian polusi ekstrem?

### Goals

- Menganalisis tren polusi udara berdasarkan data historis.
- Membangun model prediksi kualitas udara menggunakan machine learning.
- Mendeteksi kejadian polusi ekstrem menggunakan teknik anomali deteksi.

### Solution statements

- Melakukan Exploratory Data Analysis - EDA
- Membuat model machine learning dan memilih yang terbaik diantara ketiga algoritma yaitu:
  - Support Vector Machine
  - K-Nearest Neighbor
  - Random Forest
  - Neural Network
- Metrik evaluasi yang digunakan yaitu:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari situs Github, dengan total 35064 baris data dan 18 kolom fitur yang mencakup informasi terkait polusi udara. Fitur-fitur dalam dataset terbagi menjadi dua jenis: fitur kategorikal (non-numerik) dan numerik, yang akan digunakan untuk menemukan pola dalam data. Kolom PM2.5 berfungsi sebagai fitur target untuk prediksi polusi udara. Fitur-fitur pada Dataset Berikut adalah deskripsi seluruh fitur yang terdapat dalam dataset:

Dataset bisa diunduh pada link ini: [Air Quality Datasets](https://github.com/marceloreis/HTI/tree/master/PRSA_Data_20130301-20170228).

### Variabel-variabel pada Air Quality Dataset adalah sebagai berikut:

| Nama Variabel | Deskripsi                                                             |
| ------------- | --------------------------------------------------------------------- |
| No            | Nomor urut data (index dari dataset)                                  |
| year          | Tahun pengambilan data                                                |
| month         | Bulan pengambilan data (1 = Januari, 2 = Februari, dst.)              |
| day           | Hari dalam bulan pengambilan data                                     |
| hour          | Jam dalam hari pengambilan data (format 24 jam)                       |
| PM2.5         | Konsentrasi PM2.5 dalam µg/m³ (partikel halus dengan diameter ≤2.5µm) |
| PM10          | Konsentrasi PM10 dalam µg/m³ (partikel kasar dengan diameter ≤10µm)   |
| SO2           | Konsentrasi Sulfur Dioksida (SO₂) dalam µg/m³                         |
| NO2           | Konsentrasi Nitrogen Dioksida (NO₂) dalam µg/m³                       |
| CO            | Konsentrasi Karbon Monoksida (CO) dalam mg/m³                         |
| O3            | Konsentrasi Ozon (O₃) dalam µg/m³                                     |
| TEMP          | Temperatur udara dalam derajat Celcius (°C)                           |
| PRES          | Tekanan atmosfer dalam hPa (hectopascal)                              |
| DEWP          | Dew Point (Titik Embun) dalam °C (suhu di mana udara mulai mengembun) |
| RAIN          | Curah hujan dalam mm                                                  |
| wd            | Arah angin (dalam bentuk kategori, misalnya: N, NW, SE, dll.)         |
| WSPM          | Kecepatan angin dalam m/s                                             |
| station       | Nama stasiun pemantauan (dalam dataset ini: "Huairou")                |

### EDA

#### EDA-Univariate

![Numerical Univariate](https://github.com/user-attachments/assets/df72771b-70c4-49b1-8af6-d7d351a89a50)


Dilihat dari histogram variabel 'PM2.5', yang merupakan target fitur (label),
Data distribusi menunjukkan Miring ke Kanan, Pola ini menunjukkan bahwa konsentrasi partikel halus di udara umumnya dalam batas aman, tetapi terdapat periode di mana jumlahnya meningkat tajam. Partikel halus ini sangat berbahaya karena dapat masuk ke dalam sistem pernapasan manusia dan sering kali meningkat akibat pembakaran bahan bakar dan aktivitas industri.

![Wind Direction univariate](https://github.com/user-attachments/assets/bca5c63a-ca5d-44bc-8f0f-2c48b552be4e)


Terdapat 16 kategori yang mempresentasikan mata angin pada fitur wd, dari data tersebut bisa disimpulkan bahwa datanya hampir merata dengan yang tertinggi pada hari pada arah barat laut yaitu 14.4% dan arah selatan yang paling kecil datanya dengan presentase 3.8 persen.

#### EDA-Multivariate

![numerikal multivariat](https://github.com/user-attachments/assets/29322975-8717-43f6-a20f-2e7d479370dc)


Berdasarkan scatter plot diatas CO, NO2, SO2, PM10 terlihat berhubungan dengan PM2.5 secara positif.

![kategori multivariat](https://github.com/user-attachments/assets/b81d1a04-6154-40fa-898f-805020445374)


- Beberapa arah angin seperti SSE (South-Southeast) dan WNW (West-Northwest) memiliki PM2.5 tertinggi, yang berarti polusi lebih banyak ketika angin bertiup dari arah ini.
- Jika angin datang dari arah dengan PM2.5 tinggi, kemungkinan daerah tersebut memiliki sumber pencemaran signifikan (misalnya industri, pembakaran sampah, atau aktivitas kendaraan berat). Jika angin datang dari arah dengan PM2.5 rendah, kemungkinan angin membawa udara lebih bersih, atau ada faktor geografis yang membantu menyebarkan polutan.

![pca](https://github.com/user-attachments/assets/223b59fa-5ac2-4791-9099-9ff9673c0c21)


Berdasarkan scatter plot diatas, PM2.5, SO2, NO2, CO, TEMP, PRES, DEWP, WSPM akan digunakan sebagai fitur untuk membuat model.

![korelasi matrix](https://github.com/user-attachments/assets/26c89c16-0bd2-4c60-a5d2-e0fa4bd98856)


Korelasi antar variabel menunjukkan bahwa PM2.5 memiliki hubungan kuat dengan polutan lain seperti CO, NO2 dan SO2, yang umumnya berasal dari kendaraan dan industri. Ozon (O3) menunjukkan pola kebalikan dengan NO2 dan CO, karena terbentuk melalui reaksi fotokimia di udara. Dengan memahami hubungan ini, strategi pengendalian polusi dapat difokuskan pada pengurangan emisi dan pemanfaatan faktor lingkungan untuk meningkatkan kualitas udara.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Pada tahap ini, dilakukan proses pemodelan menggunakan beberapa algoritma machine learning untuk memprediksi tingkat polusi udara berdasarkan variabel lingkungan yang tersedia dalam dataset. Algoritma yang digunakan meliputi Decision Tree (DT), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest (RF). Model yang memiliki performa terbaik akan dipilih berdasarkan evaluasi metrik seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R-squared (R²).

**Algoritma yang Digunakan**

Decision Tree (DT)

Algoritma Decision Tree bekerja dengan membangun pohon keputusan berdasarkan fitur yang paling berpengaruh terhadap target. Algoritma ini cocok untuk dataset yang memiliki pola kompleks dan tidak linier.

Kelebihan:

- Mudah diinterpretasikan dan divisualisasikan.
- Tidak memerlukan normalisasi data.

Kekurangan:

- Rentan terhadap overfitting, terutama jika pohon terlalu dalam.
- Sensitif terhadap perubahan kecil dalam data.

K-Nearest Neighbors (KNN)

KNN bekerja dengan membandingkan sampel baru dengan k-tetangga terdekatnya dalam ruang fitur. Model ini sangat bergantung pada metrik jarak.

Kelebihan:

- Sederhana dan mudah diimplementasikan.
- Tidak memerlukan pelatihan eksplisit, karena semua data digunakan dalam prediksi.

Kekurangan:

- Kurang efisien untuk dataset besar karena proses perhitungan jarak yang mahal.
- Sensitif terhadap fitur yang memiliki skala berbeda (memerlukan normalisasi).

Support Vector Machine (SVM)

SVM digunakan untuk membangun hyperplane yang memisahkan data dengan margin terbesar. Dalam regresi, SVM mencari fungsi yang tetap berada dalam batas toleransi kesalahan tertentu.

Kelebihan:

- Bekerja dengan baik dalam dataset berdimensi tinggi.
- Dapat menangani hubungan non-linear dengan menggunakan kernel.

Kekurangan:

- Kurang efisien pada dataset besar.
- Sulit untuk menyesuaikan parameter kernel yang optimal.

Random Forest (RF)

Random Forest merupakan metode ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting.

Kelebihan:

- Lebih stabil dan tidak mudah overfitting dibandingkan Decision Tree.
- Dapat menangani missing values dan data outlier dengan baik.

Kekurangan:

- Sulit untuk diinterpretasikan karena kompleksitas modelnya.
- Memerlukan lebih banyak sumber daya komputasi dibandingkan Decision Tree.

Untuk meningkatkan performa model, dilakukan optimasi hyperparameter menggunakan GridSearchCV pada model terbaik yaitu Decision Tree.

```python
param_grid_dt = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gs_dt = GridSearchCV(DecisionTreeRegressor(), param_grid_dt, cv=5, scoring='r2')
gs_dt.fit(X_train, y_train)
best_dt = gs_dt.best_estimator_
```

**Isolation Forest**

Isolation Forest adalah algoritma unsupervised learning yang digunakan untuk mendeteksi anomali dalam data dengan cara mengisolasi titik data menggunakan decision tree.

Kelebihan:

- Efektif dalam mendeteksi outlier dalam dataset besar.
- Cepat dalam pelatihan karena menggunakan subset dari fitur.

Kekurangan:

- Kurang efektif untuk prediksi data reguler karena lebih fokus pada anomali.
- Sensitif terhadap jumlah estimators yang digunakan.

![hasil isolation forest](https://github.com/user-attachments/assets/1028f3bb-f0b7-41ff-8e54-4e6363a385ec)

Sebagian besar anomali berada pada nilai PM2.5 yang tinggi (>200). Ini bisa menunjukkan kejadian polusi ekstrem atau kesalahan sensor. Puncak polusi yang sangat tinggi (>500) juga terdeteksi sebagai outlier, yang mungkin terjadi akibat kondisi lingkungan tertentu seperti kebakaran hutan atau pencemaran industri.

## Evaluation

- **Mean Absolute Error (MAE):**  
  MAE = (1/n) Σ |yᵢ - ŷᵢ|  
  Mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi.

- **Mean Squared Error (MSE):**  
  MSE = (1/n) Σ (yᵢ - ŷᵢ)²  
  Mengukur rata-rata kesalahan kuadrat, memberikan penalti lebih besar pada kesalahan besar.

- **R-squared (R²):**  
  R² = 1 - (SS_res / SS_tot)  
  Mengukur seberapa baik model menjelaskan variabilitas dalam data, di mana:
  - SS_res = Σ (yᵢ - ŷᵢ)² (jumlah kuadrat residual).  
  - SS_tot = Σ (yᵢ - ȳ)² (jumlah kuadrat total dari nilai aktual ke rata-rata).

| Model | MAE     | MSE       | R2-Score |
| ----- | ------- | --------- | -------- |
| DT    | 0.0206  | 0.1891    | 0.9999   |
| KNN   | 7.8499  | 163.0623  | 0.9683   |
| SVM   | 49.5530 | 5536.7050 | -0.0731  |
| RF    | 0.0252  | 0.5907    | 0.9998   |

Dari tabel di atas, Decision Tree (DT) memiliki nilai R² tertinggi (0.9999) dan MSE terkecil (0.1891), sehingga dipilih sebagai model terbaik untuk prediksi tingkat polusi udara.

Kesimpulan

- Decision Tree dipilih sebagai model terbaik karena memberikan prediksi yang sangat akurat dengan error yang kecil.
- Isolation Forest digunakan untuk mendeteksi outlier dalam dataset.
- Hyperparameter tuning meningkatkan performa model dengan optimasi parameter yang relevan.
