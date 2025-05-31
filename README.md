# Laporan Proyek Machine Learning - Nofendra Tahta Dirgantara

## Domain Proyek

### Latar Belakang
Pasar mobil bekas mengalami pertumbuhan yang signifikan seiring dengan meningkatnya permintaan akan kendaraan pribadi yang terjangkau. Menentukan harga jual yang tepat untuk mobil bekas menjadi tantangan bagi penjual dan pembeli, karena melibatkan berbagai faktor seperti merek, model, tahun produksi, jarak tempuh, dan kondisi kendaraan. Kesalahan dalam penentuan harga dapat menyebabkan kerugian finansial atau kesulitan dalam menjual kendaraan.
Dengan kemajuan teknologi, khususnya dalam bidang kecerdasan buatan dan machine learning, memungkinkan untuk membangun model prediktif yang dapat memperkirakan harga jual mobil bekas secara lebih akurat. Model ini dapat membantu pelaku industri otomotif, dealer, dan konsumen dalam membuat keputusan yang lebih informasional dan efisien.
Model dikembangkan menggunakan pendekatan supervised learning dengan algoritma Linear Regression sebagai baseline. Evaluasi performa dilakukan menggunakan metrik regresi seperti MSE, RMSE, dan R².

## Business Understanding

### Problem Statement

Berdasarkan latar belakang di atas, permasalahan yang ingin diselesaikan dalam proyek ini adalah:

1. Bagaimana cara memprediksi harga mobil bekas berdasarkan atribut-atribut kendaraan yang tersedia?
2. Apa saja fitur yang paling berpengaruh terhadap harga jual mobil?
3. Seberapa baik performa model regresi yang dibangun dalam memprediksi harga mobil bekas?

### Goals

Tujuan dari proyek ini adalah:

- Membangun model regresi yang dapat memprediksi harga jual mobil bekas berdasarkan fitur-fitur seperti merek, model, tahun produksi, dan jarak tempuh.
- Memprediksi harga mobil dengan mengeksplorasi fitur-fitur yang paling berkontribusi dalam penentuan harga jual.
- Mengevaluasi dan membandingkan performa beberapa algoritma regresi untuk menentukan model terbaik dalam memprediksi harga mobil bekas.

## Solution Statements

Untuk mencapai tujuan tersebut, langkah-langkah yang akan dilakukan meliputi:

- Melakukan analisis eksploratif pada data untuk memahami distribusi dan hubungan antar fitur serta pembersihan data yang diperlukan.
-  Melakukan pra-pemrosesan data, termasuk penanganan nilai hilang, outlier, encoding variabel kategorikal, dan normalisasi fitur numerik.
- Membangun dan melatih beberapa model regresi, seperti K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost dan mengevaluasinya dengan metrik MAE, MSE, RMSE, dan R².
- Melakukan tuning hyperparameter menggunakan GridSearchCV untuk meningkatkan performa model.
-  Mengevaluasi model menggunakan metrik seperti Root Mean Squared Error (RMSE) dan R² Score

## Data Understanding

### URL/tautan sumber data
Dataset untuk prediksi risiko kredit ini dapat diunduh dari Kaggle: [Link Dataset](https://www.kaggle.com/datasets/makslypko/cars-dataset/data)

#### **Dataset Awal**
Dataset yang digunakan dalam proyek ini berasal dari Kaggle dan berisi informasi tentang mobil bekas, termasuk fitur-fitur seperti:

- Dataset terdiri dari **8128 baris dan 5 kolom**.
- Tiga kolom bertipe **object (kategori)**: `brand`, `fuel`, dan `owner`.
- Dua kolom bertipe **numerik** (`int64`): `km_driven` dan `selling_price`.
- Tidak ditemukan **missing values**, yang merupakan hal positif karena tidak perlu melakukan imputasi data atau penghapusan baris/kolom.

#### **Variabel-variabel pada Credit Risk Dataset Awal:**
* `brand`: Merk mobil
* `selling_price`: Harga jual (target)
* `km_driven`: Jarak tempuh (dalam kilometer)
* `fuel`: Jenis bahan bakar
* `owner`: Jumlah kepemilikan mobil sebelumnya

Beberapa langkah EDA yang dilakukan:

1. Menampilkan ringkasan statistik deskriptif serta mengamati untuk kolom numerik (`selling_price`).
  -**Visualisasi Jumlah Mobil berdasarkan brand**
   ![ss12](https://github.com/user-attachments/assets/a58385a3-eb34-4ebb-a4c5-9d460dcf3ece)
  - **visualisasi Rata-rata harga jual berdasarkan merek mobil**
  ![ss4](https://github.com/user-attachments/assets/ce9f613e-2093-4f0b-9a22-c57d316f4ead)
  menganalisis **rata-rata harga jual mobil bekas** berdasarkan jumlah pemilik sebelumnya (_owner_), setelah dilakukan winsorization (penanganan outlier).
  - **visualisasi Rata-rata harga jual berdasarkan jenis bahan bakar**
 ![ss5](https://github.com/user-attachments/assets/9e90e095-d776-4c15-85e8-7937f63e2a91)
   Bar plot ini memvisualisasikan rata-rata harga jual ('selling_price') untuk mobil berdasarkan jenis bahan bakarnya ('fuel').
  - **Visualisasi Rata-rata harga jual mobil bekas berdasarkan owner**
  ![ss6](https://github.com/user-attachments/assets/d50c8e20-ac72-44cd-a7ff-fc3e993b3570)
  Plot ini menunjukkan rata-rata harga jual ('selling_price') mobil berdasarkan kategori 'owner' (kepemilikan sebelumnya).
 
2. Menganalisis korelasi antar fitur numerik.
 - **Pair Plot of Target Features**
  ![ss7](https://github.com/user-attachments/assets/0a697296-34c6-4c28-8900-2bae81e087c8)
  PairPlot berikut adalah beberapa insight yang bisa didapatkan mengenai hubungan antara km_driven (jarak tempuh) dan selling_price (harga jual):   
 - **Correlation Matrix**
 ![ss8](https://github.com/user-attachments/assets/717d2471-f6b1-433f-b22d-a80e8b66b9db)
 Heatmap ini menampilkan matriks korelasi antara dua variabel: 'selling_price' dan 'km_driven'. Penggunaan .abs() pada kode menunjukkan bahwa nilai yang ditampilkan adalah nilai absolut dari koefisien korelasi. Artinya, kita melihat kekuatan hubungan tanpa mempertimbangkan arahnya (positif atau negatif).
Berikut beberapa tahapan yang telah dilakukan:

## Data Preparation

#### Data awal sebelum Handling Outlier
Langkah-langkah preprocessing yang dilakukan:

1. **Pembersihan Data**:
   * Melakukan duplikat cek dan menemukan 1678 data duplikat
   *  Melakukan drop duplikat
   * Dataset tidak memiliki missing value sehingga tidak diperlukan teknik imputasi atau penghapusan data.

2. **Deteksi dan Penanganan Outlier**:

   * Outlier terdeteksi pada kolom `km_driven` dan `selling_price`.
   * Digunakan metode **Winsorization** berbasis Interquartile Range (IQR), yang mengganti outlier ekstrem dengan nilai ambang batas bawah/atas.
   * Tujuannya untuk mengurangi pengaruh ekstrem tanpa membuang data.
   ![ss1](https://github.com/user-attachments/assets/2bfbad08-5a87-4210-bc08-5cc5c358a238)
### Distribusi Outlier
#### `km_driven` (Jarak Tempuh)
- **Jumlah outlier**: 166
- **Rentang nilai outlier**: 0 - 2.0 juta km
- **Interpretasi**:
  - Mobil dengan jarak tempuh > 1.5 juta km termasuk sangat ekstrim
  - Kemungkinan penyebab:
    - Mobil tua yang masih digunakan (contoh: taksi/truk)
    - Kesalahan input data (misal: 150,000 km → 1,500,000 km)

#### `selling_price` (Harga Jual)
- **Jumlah outlier**: 167  
- **Rentang nilai outlier**: ₹0 - ₹10 juta
- **Interpretasi**:
  - Harga ≈₹0 mungkin:
    - Giveaway/iklan promo
    - Kesalahan input
  - Harga >₹8 juta biasanya:
    - Mobil mewah bekas (BMW, Mercedes)
    - Mobil klasik langka
---
#### Data sesudah Handling Duplicated & Missing Value

Setelah teridentifikasi duplikat data, kita melakukan **drop_duplicated** sebagai teknik penanganannya dan data setelah dilakukannya drop duplicated kini menjadi 6450

#### Data sesudah Handling outlier

![ss2](https://github.com/user-attachments/assets/7803b120-3f2c-40cc-95c1-191200255db7)
## Penanganan Outlier dengan Winsorization

Setelah outlier teridentifikasi, kita melakukan **Winsorization** sebagai teknik penanganannya. Berbeda dengan metode trimming (menghapus data), Winsorization mengganti outlier dengan nilai ambang batas tertentu.

Metode yang digunakan berbasis **Interquartile Range (IQR)**:
- **Q1 (Kuartil 1)**: Nilai pada persentil ke-25
- **Q3 (Kuartil 3)**: Nilai pada persentil ke-75
- **IQR = Q3 - Q1**

Batas bawah dan atas ditentukan dengan rumus:
- Lower Bound = Q1 - 1.5 * IQR
- Upper Bound = Q3 + 1.5 * IQR

Setiap nilai yang lebih rendah dari batas bawah akan diubah menjadi nilai batas bawah, begitu juga dengan yang lebih tinggi dari batas atas.

### Tujuan:
- Mengurangi pengaruh outlier ekstrem tanpa mengorbankan jumlah data.
- Menjaga integritas data agar tetap representatif.
- Meningkatkan performa dan stabilitas model regresi di tahap selanjutnya.

Metode ini sangat cocok jika data memiliki outlier tapi kita tidak ingin kehilangan informasi sebanyak saat menggunakan metode penghapusan (drop).

3. **Feature Engineering**:
   - Memisahkan target agar tidak ikut encode

4. **Standarisasi Fitur Numerik**:
   * Fitur numerik seperti `km_driven` dinormalisasi menggunakan **StandardScaler** dari scikit-learn.
   * Ini dilakukan agar model yang sensitif terhadap skala fitur (seperti KNN) dapat bekerja secara optimal, karena setiap fitur memiliki skala distribusi yang berbeda.
   * Proses standarisasi dilakukan **setelah** splitting data untuk menghindari kebocoran data (data leakage).

5. **Encoding**:
   - Menggunakan `pd.get_dummies`. untuk fitur kategorikal: `fuel`, `owner`, dan `brand`.

6. **Splitting Data**:
   * Dataset dibagi menjadi **training set dan test set** dengan rasio 90:10 menggunakan `train_test_split`.
   * Tujuan splitting adalah untuk memastikan evaluasi model dilakukan pada data yang belum pernah dilihat sebelumnya.

### Urutan dan Alasan Pemilihan Teknik:

| No | Teknik                       | Alasan                                                               |
| -- | ---------------------------- | -------------------------------------------------------------------- |
| 1  | Pembersihan Data             | Tidak ada nilai kosong, jadi tidak perlu imputasi.                   |
| 2  | Deteksi & Penanganan Outlier | Menghindari distorsi akibat data ekstrem.                            |
| 3  | One-Hot Encoding             | Model regresi tidak bisa menangani data kategorikal langsung.        |
| 4  | Feature Engineering          | Memisahkan target agar tidak ikut termodifikasi saat preprocessing.  |
| 5  | StandardScaler               | Menyamaratakan skala fitur numerik, penting untuk model seperti KNN. |
| 6  | Splitting                    | Menghindari kebocoran data, memastikan evaluasi model yang adil.     |

---

## Modelling

Model yang digunakan adalah:

Beberapa algoritma regresi digunakan untuk membangun model prediktif:

1. **K-Nearest Neighbors (KNN)**:
K-Nearest Neighbors (KNN) merupakan salah satu algoritma supervised learning yang digunakan baik untuk klasifikasi maupun regresi. Pada kasus regresi, KNN memprediksi nilai target dengan menghitung rata-rata dari nilai-nilai target K tetangga terdekat yang paling mirip (berdasarkan jarak) dengan sampel yang akan diprediksi.

Pada proyek ini, model KNN digunakan dengan menggunakan fungsi `KNeighborsRegressor` dari library scikit-learn. Parameter utama yang digunakan adalah sebagai berikut:

* `n_neighbors = 10`: Parameter ini menentukan jumlah tetangga terdekat yang digunakan untuk menghitung nilai prediksi. Pemilihan nilai ini mempertimbangkan keseimbangan antara overfitting (jika nilai terlalu kecil) dan underfitting (jika nilai terlalu besar).
* `metric = 'minkowski'`: Merupakan metode pengukuran jarak antar titik data. Dalam kasus ini digunakan jarak Euclidean karena nilai default `p = 2`.

Algoritma ini sangat bergantung pada **skala fitur**, sehingga seluruh fitur numerik distandarisasi menggunakan `StandardScaler` sebelum proses pelatihan. KNN memiliki keunggulan dari sisi kesederhanaan dan interpretasi hasil, namun dapat menjadi lambat dalam prediksi ketika jumlah data besar karena menghitung jarak terhadap seluruh titik latih.

---

2. **Random Forest**:
Random Forest adalah algoritma ensemble yang dibangun berdasarkan konsep **bagging** (Bootstrap Aggregating). Model ini terdiri dari sekumpulan decision tree yang dilatih secara independen dari sampel acak data, dan hasil prediksinya dirata-ratakan untuk regresi. Dengan pendekatan ini, Random Forest dapat **mengurangi variansi model**, sehingga lebih tahan terhadap overfitting dibandingkan dengan single decision tree.

Model ini dibangun menggunakan `RandomForestRegressor` dari scikit-learn dengan parameter awal sebagai berikut:

* `n_estimators = 50`: Jumlah pohon yang akan dibangun dalam model. Semakin banyak pohon, hasil rata-rata lebih stabil, namun memerlukan waktu komputasi lebih besar.
* `max_depth = 16`: Kedalaman maksimum setiap pohon. Batasan ini digunakan untuk menghindari overfitting dengan membatasi kompleksitas pohon.

Keunggulan Random Forest meliputi kemampuannya untuk:

* Menangani **fitur-fitur non-linear** dan interaksi antar fitur.
* Memberikan **importance score** terhadap masing-masing fitur, yang berguna dalam analisis lebih lanjut.

Namun, performa model sangat bergantung pada pemilihan parameter seperti kedalaman pohon dan jumlah estimators, sehingga proses tuning sangat diperlukan untuk mendapatkan performa optimal.

---

3. **AdaBoost**:

  AdaBoost (Adaptive Boosting) adalah algoritma boosting yang membangun sekumpulan **weak learner** secara berurutan, di mana setiap model selanjutnya difokuskan untuk memperbaiki kesalahan dari model sebelumnya. Model akhir merupakan kombinasi dari seluruh weak learner dengan bobot berdasarkan performa masing-masing.

Model ini dibangun menggunakan `AdaBoostRegressor` dari scikit-learn dengan parameter sebagai berikut:

* `learning_rate = 0.05`: Parameter ini mengontrol kontribusi masing-masing weak learner terhadap model akhir. Nilai yang lebih kecil membuat pelatihan lebih lambat tetapi bisa meningkatkan generalisasi model.
* `n_estimators = 50`: Jumlah total weak learners (biasanya berupa decision tree sederhana) yang akan dilatih secara bertahap.

Kelebihan utama AdaBoost terletak pada kemampuannya meningkatkan akurasi model sederhana (misalnya decision stump), namun algoritma ini juga **sensitif terhadap outlier dan noise**, karena upaya terus-menerus memperbaiki kesalahan bisa menyebabkan overfitting pada sampel tidak representatif.

---

4. **Hyperparameter Tuning**:

   Untuk meningkatkan performa model, dilakukan proses **hyperparameter tuning** menggunakan metode `GridSearchCV`. Metode ini menguji seluruh kombinasi dari parameter yang ditentukan dengan proses cross-validation untuk memilih parameter terbaik berdasarkan metrik performa rata-rata.

Parameter yang dituning meliputi:

* **Random Forest**:

  * `n_estimators`: Banyaknya pohon, nilai diuji antara 50 hingga 150.
  * `max_depth`: Kedalaman maksimum pohon, diuji antara 8 hingga 32.
  * `min_samples_split`: Jumlah minimum sampel untuk membagi node.
  * `min_samples_leaf`: Jumlah minimum sampel di setiap daun pohon.

* **AdaBoost**:

  * `n_estimators`: Jumlah weak learners.
  * `learning_rate`: Laju pembelajaran yang menentukan seberapa besar kontribusi masing-masing learner terhadap prediksi akhir.

GridSearchCV menggunakan 5-fold cross-validation dalam setiap kombinasi parameter untuk menghindari overfitting terhadap data pelatihan, dan hasil terbaik digunakan untuk evaluasi pada data uji.

---

Model dilatih pada data training, dan digunakan untuk memprediksi harga pada data test. Proses pelatihan berlangsung dengan efisien dan cepat karena linear regression bersifat sederhana dan memiliki kompleksitas rendah.

## Evaluation

Evaluasi dilakukan menggunakan metrik regresi berikut:
* **Root Mean Squared Error (RMSE)**: Mengukur rata-rata kesalahan prediksi model. Semakin rendah nilai RMSE, semakin baik performa model.
* **R² Score**: Menunjukkan proporsi variansi dalam data target yang dapat dijelaskan oleh model. Nilai R² mendekati 1 menunjukkan model yang baik.

### Hasil Evaluasi:
| Model                     | MSE (Train)      | MSE (Test)     |
| :------------------------ | :--------------- | :------------- |
| KNN                       | 35,702,044.93    | 45,486,426.17  |
| Random Forest             | 27,268,530.11    | 47,109,176.79  |
| Boosting (Original)       | 57,408,545.46    | 56,083,611.26  |

![ss9](https://github.com/user-attachments/assets/2e991d43-5743-4e12-b52b-2f8fed2aa85c)

*Visualisasi MSE*

| Model               | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
| :------------------ | :----------- | :---------- | :--------- | :-------- |
| Random Forest Tuned | 188,352.78   | 215,080.72  | 0.60       | 0.49      |
| Boosting Tuned      | 189,749.05   | 207,779.42  | 0.59       | 0.52      |

![ss10](https://github.com/user-attachments/assets/16e5947a-c2da-406e-9627-5627c7fee222)

*Visualisasi RMSE & R2*

Hasil menunjukkan bahwa model cukup baik dalam menjelaskan variasi harga mobil bekas. Namun, masih terdapat kemungkinan untuk meningkatkan performa dengan algoritma lain seperti Ridge Regression, Lasso Regression, XGBoost, serta melakukan hyperparameter tuning.

## Conclusion and Future Work

Model linear regression berhasil membangun prediksi harga mobil bekas dengan akurasi yang cukup tinggi (R² = 0.67). Fitur-fitur seperti `km_driven`, `brand`, dan `km_driven` menjadi faktor dominan dalam penentuan harga.
Model ini memberikan dasar yang baik untuk pengembangan sistem pricing otomatis dalam platform jual-beli mobil bekas.

> Model yang dikembangkan telah mencapai standar Minimum Viable Product (MVP) yang ditetapkan, sehingga dapat segera digabungkan dengan infrastruktur penilaian kredit yang sudah berjalan.
