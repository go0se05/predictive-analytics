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
* Tiga kolom bertipe **object (kategori)**: `brand`, `fuel`, dan `owner`.
* Dua kolom bertipe **numerik** (`int64`): `km_driven` dan `selling_price`.
  
* **Fitur**:

  * `brand` : Merk mobil(kategori)
  * `fuel` : Jenis bahan bakar(kategori)
  * `owner` : Jumlah kepemilikan mobil sebelumnya(kategori)
  * `km_driven` : Jarak tempuh --> dalam kilometer(numerik)
  * `selling_price` : Harga jual(numerik, target)

* Tidak ditemukan **missing values**, yang merupakan hal positif karena tidak perlu melakukan imputasi data atau penghapusan baris/kolom.

---

### Exploratory Data Analysis (EDA)

Setelah memahami gambaran umum dataset melalui `cars.info()` dan `cars.describe()`, kita melanjutkan ke **Exploratory Data Analysis (EDA)**. EDA membantu kita menganalisis distribusi data, menemukan hubungan antar variabel, dan mengungkap insight yang tersembunyi sebelum membangun model. Berikut langkah-langkah yang telah dijalankan:  


* **Visualisasi Ditribusi Outlier**

![ss1](https://github.com/user-attachments/assets/2bfbad08-5a87-4210-bc08-5cc5c358a238)

Visuaslisasi diatas menunjukan outlier terdeteksi pada data

---

* **Distribusi Brand Mobil**
  
![Brand Distribution](https://github.com/user-attachments/assets/971b6736-5b0d-4ba8-a284-9107f1efbf38)

Visualisasi ini menunjukan distribusi jumlah mobil berdasarkan brand

---

* **Rata-rata Harga Jual per Brand**
  
![ss4](https://github.com/user-attachments/assets/f2460a55-5905-4101-a1da-249556d70dc5)

jumlah pemilik sebelumnya (_owner_), setelah dilakukan winsorization (penanganan outlier).

---

* **Harga Jual Berdasarkan Bahan Bakar**

![ss5](https://github.com/user-attachments/assets/b4a74b0a-d313-47d8-af91-1ab205ec085f)

Bar plot ini memvisualisasikan rata-rata harga jual ('selling_price') untuk mobil berdasarkan jenis bahan bakarnya ('fuel').

---

* **Harga Jual Berdasarkan Jumlah Kepemilikan**
  
![ss6](https://github.com/user-attachments/assets/af367983-f104-49fb-aac8-fd360fd1e640)

Plot ini menunjukkan rata-rata harga jual ('selling_price') mobil berdasarkan kategori 'owner' (kepemilikan sebelumnya).

---

* **Pair Plot Target Feature**
  
![ss7](https://github.com/user-attachments/assets/9ea52182-b090-4d23-b2b8-fa691ea4b820)

Memvisualisasikan distribusi masing-masing variabel dan mengidentifikasi pola/korelasi antara dua variabel numerik

---

* **Heatmap Korelasi**
  
![ss8](https://github.com/user-attachments/assets/c0c29745-8c25-406a-8ff4-d1cba3aa323d)
Heatmap ini menampilkan matriks korelasi antara dua variabel: 'selling_price' dan 'km_driven'. Penggunaan .abs() pada kode menunjukkan bahwa nilai yang ditampilkan adalah nilai absolut dari koefisien korelasi. Artinya, kita melihat kekuatan hubungan tanpa mempertimbangkan arahnya (positif atau negatif).


EDA ini membantu memahami pola dan hubungan antar fitur sebelum modeling.

---

## Data Preparation

### 1. Pemeriksaan Awal

* Dataset tidak memiliki missing values.
* Terdapat 1.678 data duplikat → dihapus.

### 2. Pemisahan Target

* Variabel target (`selling_price`) dipisahkan dari dataset agar tidak ikut termodifikasi pada tahap preprocessing.

### 3. Deteksi & Penanganan Outlier

###### `km_driven` (Jarak Tempuh)

* **Jumlah outlier**: 166
* **Rentang nilai outlier**: 0 - 2.0 juta km
* **Interpretasi**:

  * Mobil tua (taksi/truk) atau kesalahan input data (contoh: 150,000 km → 1,500,000 km).

###### `selling_price` (Harga Jual)

* **Jumlah outlier**: 167
* **Rentang nilai outlier**: ₹0 - ₹10 juta
* **Interpretasi**:

  * Harga sangat rendah (promo/gratis) atau sangat tinggi (mobil mewah/langka).

---
* **Visualisasi Sebelum Penanganan Outlier**
  
![ss1](https://github.com/user-attachments/assets/2bfbad08-5a87-4210-bc08-5cc5c358a238)

* **Data setelah Handling Outlier**

![ss2](https://github.com/user-attachments/assets/7803b120-3f2c-40cc-95c1-191200255db7)

#### Penanganan Outlier dengan Winsorization

Metode berbasis **IQR (Interquartile Range)**:

* **Q1** = persentil ke-25
* **Q3** = persentil ke-75
* **IQR** = Q3 - Q1
* Batas bawah: Q1 - 1.5 \* IQR
* Batas atas: Q3 + 1.5 \* IQR

Semua nilai di luar batas bawah/atas akan diganti dengan nilai batas tersebut.

##### Tujuan:

* Mengurangi pengaruh nilai ekstrem.
* Menjaga integritas data (tidak kehilangan banyak data).
* Meningkatkan stabilitas model.

---

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

## Modelling

Model yang digunakan adalah:

* #### Ringkasan

  * **Random Forest**: Tuning `n_estimators`, `max_depth`, `min_samples_split`.
  * **AdaBoost**: Tuning `n_estimators`, `learning_rate`.
  * **Metode**: **GridSearchCV** dengan 5-fold cross-validation.
---

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

### Hasil Evaluasi:

* **Visualisasi sebelum Hyperparameter Tunning**

| Model                     | MSE (Train)      | MSE (Test)     |
| :------------------------ | :--------------- | :------------- |
| KNN                       | 35,702,044.93    | 45,486,426.17  |
| Random Forest             | 27,268,530.11    | 47,109,176.79  |
| Boosting (Original)       | 57,408,545.46    | 56,083,611.26  |

![ss9](https://github.com/user-attachments/assets/2e991d43-5743-4e12-b52b-2f8fed2aa85c)

*Visualisasi MSE*

---

* **Visualisasi sesudah Hyperparameter Tunning**

| Model               | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
| :------------------ | :----------- | :---------- | :--------- | :-------- |
| Random Forest Tuned | 188,352.78   | 215,080.72  | 0.60       | 0.49      |
| Boosting Tuned      | 189,749.05   | 207,779.42  | 0.59       | 0.52      |

![ss10](https://github.com/user-attachments/assets/16e5947a-c2da-406e-9627-5627c7fee222)

*Visualisasi RMSE & R2*

---

Hasil menunjukkan bahwa model cukup baik dalam menjelaskan variasi harga mobil bekas. Namun, masih terdapat kemungkinan untuk meningkatkan performa dengan algoritma lain seperti Ridge Regression, Lasso Regression, XGBoost, serta melakukan hyperparameter tuning.

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

> Model yang dikembangkan telah mencapai standar Minimum Viable Product (MVP) yang ditetapkan, sehingga dapat segera digabungkan dengan infrastruktur penilaian kredit yang sudah berjalan.
