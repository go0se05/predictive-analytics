# Laporan Proyek Machine Learning - Nofendra Tahta Dirgantara

---

## Domain Proyek

Pasar mobil bekas terus mengalami pertumbuhan yang signifikan seiring dengan meningkatnya kebutuhan kendaraan pribadi yang terjangkau. Menentukan harga jual mobil bekas menjadi tantangan, karena bergantung pada berbagai faktor seperti merek, model, jarak tempuh, dan kepemilikan sebelumnya. Kesalahan dalam estimasi harga dapat berdampak pada kerugian finansial dan efisiensi transaksi.
Dengan perkembangan teknologi AI, khususnya machine learning, memungkinkan pembuatan model prediktif yang lebih akurat dan adaptif. Model ini sangat dibutuhkan oleh dealer, konsumen, dan pelaku industri otomotif untuk mengambil keputusan yang lebih cerdas dan efisien.

---

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi harga jual mobil bekas dengan akurasi tinggi berdasarkan atribut kendaraan?
2. Fitur apa saja yang paling berpengaruh terhadap harga jual mobil bekas?
3. Model prediktif mana yang paling optimal untuk kebutuhan ini?

### Goals

* Membangun model regresi yang akurat untuk memprediksi harga jual mobil bekas.
* Mengidentifikasi fitur-fitur utama yang memengaruhi harga jual.
* Mengevaluasi dan memilih model terbaik berdasarkan metrik regresi (RMSE & R²).

---

## Data Understanding

### Sumber Data

Dataset diperoleh dari Kaggle: [Cars Dataset](https://www.kaggle.com/datasets/makslypko/cars-dataset/data).

### Karakteristik Dataset

* **Jumlah Baris**: 8.128
* **Jumlah Kolom**: 5
* **Fitur**:

  * `brand` (kategori)
  * `fuel` (kategori)
  * `owner` (kategori)
  * `km_driven` (numerik)
  * `selling_price` (numerik, target)

Dataset tidak memiliki missing values.

---

### Exploratory Data Analysis (EDA)

* **Distribusi Brand Mobil**
  ![Brand Distribution](https://github.com/user-attachments/assets/a58385a3-eb34-4ebb-a4c5-9d460dcf3ece)
* **Rata-rata Harga Jual per Brand**
  ![Avg Price by Brand](https://github.com/user-attachments/assets/ce9f613e-2093-4f0b-9a22-c57d316f4ead)
* **Harga Jual Berdasarkan Bahan Bakar**
  ![Avg Price by Fuel](https://github.com/user-attachments/assets/9e90e095-d776-4c15-85e8-7937f63e2a91)
* **Harga Jual Berdasarkan Jumlah Kepemilikan**
  ![Avg Price by Owner](https://github.com/user-attachments/assets/d50c8e20-ac72-44cd-a7ff-fc3e993b3570)
* **Heatmap Korelasi**
  ![Correlation](https://github.com/user-attachments/assets/717d2471-f6b1-433f-b22d-a80e8b66b9db)

EDA ini membantu memahami pola dan hubungan antar fitur sebelum modeling.

---

## Data Preparation

### 1. Pemeriksaan Awal

* Dataset tidak memiliki missing values.
* Terdapat 1.678 data duplikat → dihapus.

### 2. Pemisahan Target

* Variabel target (`selling_price`) dipisahkan dari dataset agar tidak ikut termodifikasi pada tahap preprocessing.

### 3. Deteksi & Penanganan Outlier

* **Outlier pada `km_driven`**: 166 data ekstrem (contoh: taksi lama, data entry error).
* **Outlier pada `selling_price`**: 167 data (mobil mewah/langka, data promo).
* **Metode**: **Winsorization** berbasis IQR.
* **Tujuan**: mengurangi pengaruh outlier ekstrem tanpa menghapus data penting.

### 4. Encoding Fitur Kategorikal

* **One-Hot Encoding** (`pd.get_dummies`) untuk fitur: `brand`, `fuel`, dan `owner`.

### 5. Splitting Data

* Dataset dibagi menjadi **training set (90%)** dan **test set (10%)**.
* Menghindari data leakage dan memastikan evaluasi model yang fair.

### 6. Standardisasi Fitur Numerik

* Fitur numerik `km_driven` di-standardisasi menggunakan **StandardScaler**.
* **fit** dilakukan pada **training set** saja → **transform** pada **training** & **test set**.

### Urutan & Alasan

| No | Langkah          | Alasan                                                         |
| -- | ---------------- | -------------------------------------------------------------- |
| 1  | Pemisahan Target | Hindari target ikut termodifikasi.                             |
| 2  | Pembersihan Data | Dataset bersih & konsisten.                                    |
| 3  | Outlier Handling | Kurangi pengaruh data ekstrem tanpa buang data.                |
| 4  | One-Hot Encoding | Model regresi tidak bisa handle data kategorikal langsung.     |
| 5  | Splitting Data   | Cegah data leakage, evaluasi fair.                             |
| 6  | StandardScaler   | Model (seperti KNN) sensitif pada skala → perlu standardisasi. |

---

## Modeling

### Algoritma yang Digunakan

1. **K-Nearest Neighbors (KNN)**

   * **Kelebihan**: Sederhana, efektif untuk data non-linear.
   * **Kekurangan**: Sensitif pada skala data & memori.

2. **Random Forest**

   * **Kelebihan**: Robust terhadap outlier, menangani non-linearitas & interaksi fitur.
   * **Kekurangan**: Interpretasi lebih sulit dibanding model linear.

3. **AdaBoost**

   * **Kelebihan**: Boosting fokus pada data sulit.
   * **Kekurangan**: Lebih sensitif pada outlier & noise.

### Improvement: Hyperparameter Tuning

* **Random Forest**: Tuning `n_estimators`, `max_depth`, `min_samples_split`.
* **AdaBoost**: Tuning `n_estimators`, `learning_rate`.
* **Metode**: **GridSearchCV** dengan 5-fold cross-validation.

### Model Terbaik

* Berdasarkan RMSE & R², **Random Forest** (setelah tuning) memberikan hasil paling stabil dan akurat.
* Random Forest lebih cocok menangani data tabular & fitur-fitur yang saling berinteraksi.

---

## Evaluation

### Metrik Evaluasi

1. **Root Mean Squared Error (RMSE)**

   $$
   \text{RMSE} = \sqrt{\frac{1}{n} \sum (y_{true} - y_{pred})^2}
   $$

   → Mengukur rata-rata kesalahan prediksi, makin kecil makin baik.

2. **R² Score**

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

   → Proporsi variansi yang dijelaskan model, makin mendekati 1 makin baik.

### Hasil Evaluasi

| Model               | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
| ------------------- | ------------ | ----------- | ---------- | --------- |
| Random Forest Tuned | 188,352.78   | 215,080.72  | 0.60       | 0.49      |
| AdaBoost Tuned      | 189,749.05   | 207,779.42  | 0.59       | 0.52      |

### Visualisasi Hasil

![ss10](https://github.com/user-attachments/assets/16e5947a-c2da-406e-9627-5627c7fee222)

---

## Conclusion & Recommendation

### Ringkasan

* Model **Random Forest** terpilih sebagai solusi terbaik.
* Model mampu memprediksi harga mobil bekas dengan cukup baik (R² test = 0.49).
* Fitur `brand`, `km_driven`, dan `owner` paling berpengaruh.

### Rekomendasi Implementasi

1. Gunakan model Random Forest untuk membantu dealer/penjual mobil bekas menentukan harga jual.
2. Lakukan monitoring & retraining berkala agar model adaptif terhadap tren pasar.
3. Eksperimen lanjutan: Tambah data historis & fitur kondisi mobil (misalnya: tahun produksi).

---

