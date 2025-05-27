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
3. Model machine learning apa yang paling efektif dalam memprediksi harga mobil bekas dengan akurasi tinggi?
4. Seberapa baik performa model regresi yang dibangun dalam memprediksi harga mobil bekas?

### Goals

Tujuan dari proyek ini adalah:

- Membangun model regresi yang dapat memprediksi harga jual mobil bekas berdasarkan fitur-fitur seperti merek, model, tahun produksi, dan jarak tempuh.
- Mengidentifikasi fitur-fitur yang paling berkontribusi dalam penentuan harga jual.
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
Dataset untuk prediksi risiko kredit ini dapat diunduh dari Kaggle: **[Link Dataset]**(https://www.kaggle.com/datasets/makslypko/cars-dataset/data)

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

- Menampilkan ringkasan statistik deskriptif untuk kolom numerik.
- Mengamati distribusi harga (`selling_price`) dan mendeteksi outlier.
- Menilai distribusi `km_driven` yang memiliki skewness positif, lalu dilakukan log transformasi.
- Mengubah `year` menjadi `car_age` untuk merepresentasikan umur mobil.
- Menganalisis korelasi antar fitur numerik.

Berikut beberapa tahapan yang telah dilakukan:
![Screenshot](C:\Users\nofen\Submission\Visualiasi\ss1.jpg)

## Data Preprocessing

Langkah-langkah preprocessing yang dilakukan:

1. **Pembersihan Data**:
   - Menghapus data duplikat (sebanyak 109 baris).
   - Tidak ditemukan nilai kosong.

2. **Feature Engineering**:
   - Ekstraksi `brand` dari kolom `name`.
   - Penghapusan kolom `name` setelah ekstraksi `brand`.

3. **Encoding**:
   - Menggunakan `OneHotEncoder` untuk fitur kategorikal: `fuel`, `seller_type`, `transmission`, `owner`, dan `brand`.

4. **Transformasi Numerik**:
   - Melakukan log transformasi pada `km_driven`.
   - Melakukan normalisasi menggunakan `StandardScaler`.

5. **Splitting**:
   - Data dibagi menjadi training dan test set dengan rasio 80:20 menggunakan `train_test_split`.

## Modelling

Model yang digunakan adalah:

Beberapa algoritma regresi digunakan untuk membangun model prediktif:

1. **K-Nearest Neighbors (KNN)**:

   * Parameter: `n_neighbors=10`
   * Kelebihan: Sederhana dan efektif untuk dataset kecil.
   * Kekurangan: Sensitif terhadap skala fitur dan outlier.

2. **Random Forest**:

   * Parameter awal: `n_estimators=50`, `max_depth=16`
   * Kelebihan: Mengurangi overfitting dan menangani fitur non-linear dengan baik.
   * Kekurangan: Memerlukan tuning parameter untuk performa optimal.

3. **AdaBoost**:

   * Parameter awal: `learning_rate=0.05`
   * Kelebihan: Meningkatkan akurasi dengan menggabungkan model lemah.
   * Kekurangan: Sensitif terhadap noise dan outlier.

4. **Hyperparameter Tuning**:

   * Menggunakan GridSearchCV untuk mencari kombinasi parameter terbaik pada Random Forest dan AdaBoost.
   * Parameter yang dituning meliputi `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, dan `learning_rate`.

Model dilatih pada data training, dan digunakan untuk memprediksi harga pada data test. Proses pelatihan berlangsung dengan efisien dan cepat karena linear regression bersifat sederhana dan memiliki kompleksitas rendah.

## Evaluation

Evaluasi dilakukan menggunakan metrik regresi berikut:
* **Root Mean Squared Error (RMSE)**: Mengukur rata-rata kesalahan prediksi model. Semakin rendah nilai RMSE, semakin baik performa model.
* **R² Score**: Menunjukkan proporsi variansi dalam data target yang dapat dijelaskan oleh model. Nilai R² mendekati 1 menunjukkan model yang baik.

### Hasil Evaluasi:
| Model                     | MSE (Train)      | MSE (Test)     |
| :------------------------ | :--------------- | :------------- |
| KNN                       | 33,872,041.65    | 39,428,361.48  |
| Random Forest             | 26,379,393.65    | 40,818,716.80  |
| Boosting (Original)       | 63,513,173.02    | 64,366,053.76  |

| Model               | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
| :------------------ | :----------- | :---------- | :--------- | :-------- |
| Random Forest Tuned | 140,567.89   | 175,234.56  | 0.77       | 0.64      |
| Boosting Tuned      | 176,066.13   | 193,574.27  | 0.72       | 0.67      |

Hasil menunjukkan bahwa model cukup baik dalam menjelaskan variasi harga mobil bekas. Namun, masih terdapat kemungkinan untuk meningkatkan performa dengan algoritma lain seperti Ridge Regression, Lasso Regression, XGBoost, serta melakukan hyperparameter tuning.

## Conclusion and Future Work

Model linear regression berhasil membangun prediksi harga mobil bekas dengan akurasi yang cukup tinggi (R² = 0.77). Fitur-fitur seperti `km_driven`, `brand`, dan `km_driven` menjadi faktor dominan dalam penentuan harga.
Model ini memberikan dasar yang baik untuk pengembangan sistem pricing otomatis dalam platform jual-beli mobil bekas.

