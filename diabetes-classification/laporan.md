# Laporan Proyek Machine Learning - Sherly Santiadi

## Domain Proyek

Penderita diabetes beresiko lebih tinggi pada kasus pasien COVID-19. Oleh karena itu, pentingnya untuk mendeteksi sedini mungkin penyakit diabetes berdasarkan latar belakang historis sehingga dapat diberi perawatan yang tepat.

[Sumber Referensi](https://bapin-ismki.e-journal.id/jimki/article/view/342)

## Business Understanding

Bagian laporan ini mencakup:

### Problem Statements
- Bagaimana membuat model untuk mengidentifikasi pasien yang beresiko diabetes?
- Bagaimana mengukur performa model untuk memprediksi pasien yang beresiko diabetes?

### Goals
- Membuat model untuk mengidentifikasi pasien yang beresiko diabetes.
- Mengukur performa model untuk memprediksi pasien yang beresiko diabetes.

### Solution statements
- Membangun model `random forest` dengan menerapkan `hyperparameter tuning`
- Mengukur performa model dengan metrik evaluasi.

## Data Understanding
Data yang digunakan dalam proyek ini bersumber dari [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).
| gender | age  | hypertension | heart_disease | smoking_history | bmi   | HbA1c_level | blood_glucose_level | diabetes |
|--------|------|--------------|---------------|-----------------|-------|-------------|---------------------|----------|
| Female | 80.0 | 0            | 1             | never           | 25.19 | 6.6         | 140                 | 0        |
| Male   | 54.0 | 1            | 0             | No Info         | 27.32 | 5.7         | 80                  | 1        |
| ...    | ...  | ...          | ...           | ...             | ...   | ...         | ...                 | ...      |

### Variabel-variabel pada Diabetes dataset adalah sebagai berikut:
- gender: mengacu pada jenis kelamin biologis individu, yang dapat berdampak pada kerentanan mereka terhadap diabetes. Ada tiga kategori di dalamnya pria, wanita dan lainnya.
- age: merupakan faktor penting karena diabetes lebih sering didiagnosis pada orang yang lebih tua. Rentang usia dari 0-80.
- hypertension: merupakan suatu kondisi medis di mana tekanan darah di arteri terus-menerus meningkat. Di dalam dataset ini terdapat nilai 0 atau 1 di mana 0 menunjukkan mereka tidak memiliki hipertensi dan untuk 1 itu berarti mereka memiliki hipertensi.
- heart_disease: merupakan suatu kondisi medis lain yang dikaitkan dengan peningkatan risiko diabetes. Di dalam dataset ini terdapat nilai 0 atau 1 dimana 0 menunjukkan mereka tidak memiliki penyakit jantung dan untuk 1 itu berarti mereka memiliki penyakit jantung.
- smoking_history: riwayat merokok juga dianggap sebagai faktor risiko diabetes dan dapat memperburuk komplikasi yang terkait dengan diabetes. Dalam dataset, terdapat 5 kategori yaitu tidak terkini, mantan, Tanpa Info, terkini, tidak pernah, dan selamanya.
- bmi: merupakan ukuran lemak tubuh berdasarkan berat dan tinggi badan. Nilai BMI yang lebih tinggi terkait dengan risiko diabetes yang lebih tinggi.
  - Kisaran BMI dalam dataset adalah dari 10.16 hingga 71.55.
  - BMI kurang dari 18.5 adalah kurus
  - BMI 18.5-24.9 adalah normal
  - BMI 25-29.9 adalah kelebihan berat badan
  - BMI 30 atau lebih adalah obesitas.
- HbA1c_level: kadar HbA1c (Hemoglobin A1c) adalah ukuran kadar gula darah rata-rata seseorang selama 2-3 bulan terakhir. Tingkat yang lebih tinggi menunjukkan risiko lebih besar terkena diabetes. Sebagian besar lebih dari 6,5% dari Tingkat HbA1c menunjukkan diabetes.
- blood_glucose_level: tingkat glukosa darah mengacu pada jumlah glukosa dalam aliran darah pada waktu tertentu. Kadar glukosa darah yang tinggi merupakan indikator utama diabetes.
- diabetes: diabetes adalah variabel target yang akan diprediksi, dengan nilai 1 menunjukkan adanya diabetes dan 0 menunjukkan tidak adanya diabetes.

## Exploratory Data Analysis
Tahapan yang akan dilakukan dalam mengeksplor dataset yaitu:
- Gathering Data
- Assessing Data
- Describing Data

## Data Preparation
Tahapan yang akan dilakukan dalam mempersiapkan dataset yaitu:
- One Hot Encoding: mengubah variabel kategorikal menjadi representasi numerik yang sesuai, di dalam dataset yang digunakan dalam proyek ini terdapat dua kolom kategorikal yaitu kolom `gender` dan kolom `smoking_history`. Hal ini penting untuk memastikan bahwa semua variabel dalam dataset bersifat numerik agar dapat digunakan dalam model.
- Standarisasi: mengubah skala variabel numerik agar memiliki mean (rerata) nol dan standar deviasi satu. Tujuan standarisasi adalah untuk menghilangkan perbedaan skala antar variabel, sehingga variabel dengan skala yang lebih besar tidak mendominasi kontribusinya dalam pembelajaran mesin. Standarisasi juga membantu meningkatkan interpretasi model, karena koefisien yang dihasilkan dapat dibandingkan secara langsung. Standarisasi akan diimplementasikan ke dalam kolom numerik yaitu `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`. Dengan mengimplementasikan standarisasi dapat dilihat pada mean bernilai 0 dan standar deviasi bernilai 1.
- Resampling menggunakan SMOTE dan Random Under Sampler: mengubah distribusi kelas pada dataset dengan tujuan untuk mengatasi ketidakseimbangan kelas (class imbalance). SMOTE (Synthetic Minority Over-sampling Technique) adalah metode resampling yang membuat sampel sintetis untuk kelas minoritas dengan menggabungkan fitur-fitur dari sampel kelas minoritas yang ada. Dengan menggunakan SMOTE, jumlah sampel pada kelas minoritas akan meningkat sehingga dapat mengurangi ketidakseimbangan kelas. Metode ini membantu meningkatkan kinerja model dalam memprediksi kelas minoritas. Sedangkan, Random Under Sampler adalah metode resampling yang secara acak mengurangi jumlah sampel pada kelas mayoritas agar sebanding dengan jumlah sampel pada kelas minoritas. Dengan mengurangi jumlah sampel pada kelas mayoritas, metode ini membantu mengatasi ketidakseimbangan kelas dengan mempertahankan sampel pada kelas minoritas yang ada. Tujuan dari Random Under Sampler adalah untuk mencegah overfitting pada kelas mayoritas dan meningkatkan performa model dalam memprediksi kelas minoritas.

## Modeling
Pada tahapan pemodelan ini, Penulis menggunakan algoritma Random Forest Classifier untuk menyelesaikan permasalahan dalam proyek "Diabetes Prediction".

Tahapan yang dilakukan dalam pemodelan ini adalah sebagai berikut:
- Membentuk model awal: Pertama, Penulis membuat model awal dengan menggunakan RandomForestClassifier tanpa mengatur parameter apapun. Model awal ini digunakan untuk melatih data yang telah diresampling `(X_train_resampled)` dan `(y_train_resampled)`.
- Penentuan parameter: Setelah itu, Penulis menentukan parameter-parameter yang akan digunakan pada model Random Forest. Dalam contoh ini, Penulis menggunakan RandomizedSearchCV untuk mencari parameter terbaik dengan mencoba beberapa kombinasi secara acak. Parameter yang diuji adalah jumlah estimators `(n_estimators)`, kedalaman maksimum `(max_depth)`, minimum sampel split `(min_samples_split)`, dan minimum sampel leaf `(min_samples_leaf)`. Parameter-parameter ini digunakan untuk meningkatkan performa model dan menghindari overfitting.
- Pencarian parameter terbaik: Pada langkah ini, RandomizedSearchCV mencoba beberapa kombinasi parameter yang dijelaskan di atas dengan melakukan validasi silang `(cross-validation)` menggunakan 5-fold cross-validation `(cv=5)`. Dalam hal ini, RandomizedSearchCV mencoba 10 kombinasi parameter yang berbeda `(n_iter=10)` dan mencatat performa model untuk setiap kombinasi.
- Memilih model terbaik: Setelah melakukan pencarian parameter, Penulis memilih model terbaik berdasarkan performa yang diukur. Model terbaik ini akan digunakan untuk melatih data yang telah dilakukan one hot encoding `(X_train_encoded)` dan target `(y_train)`.

## Hyperparameter Tuning
Proses improvement melalui hyperparameter tuning:

Dalam langkah hyperparameter tuning, RandomizedSearchCV digunakan untuk mencari kombinasi terbaik dari parameter-parameter yang diuji. Dalam contoh ini, Penulis mencari nilai terbaik untuk n_estimators, max_depth, min_samples_split, dan min_samples_leaf.

Dengan melakukan hyperparameter tuning, Penulis dapat meningkatkan performa model dengan menemukan parameter yang optimal. Parameter yang optimal dapat membantu menghindari overfitting, meningkatkan akurasi, dan menghasilkan model yang lebih baik secara keseluruhan.

Setelah proses hyperparameter tuning, Penulis mendapatkan model terbaik `(rf_best)` yang kemudian dilatih menggunakan data yang telah dilakukan one hot encoding `(X_train_encoded)` dan target `(y_train)`. Model terbaik ini kemudian dapat digunakan untuk melakukan prediksi pada data yang baru.

## Evaluation
Dalam proyek ini, Penulis menggunakan beberapa metrik evaluasi untuk kasus klasifikasi, yaitu akurasi, presisi, recall, dan skor F1. Berikut adalah penjelasan singkat mengenai metrik-metrik tersebut:

- Akurasi: Metrik ini mengukur sejauh mana model dapat mengklasifikasikan data dengan benar secara keseluruhan. Akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total jumlah prediksi.

- Presisi: Presisi mengukur sejauh mana model memberikan hasil positif yang benar dari semua prediksi positif yang dilakukan. Presisi dihitung dengan membagi jumlah prediksi positif yang benar dengan jumlah total prediksi positif.

- Recall: Recall, juga dikenal sebagai sensitivitas, mengukur sejauh mana model dapat mengidentifikasi secara benar semua instance positif yang ada dalam dataset. Recall dihitung dengan membagi jumlah prediksi positif yang benar dengan jumlah total instance positif yang sebenarnya.

- F1 Score: F1 Score adalah rata-rata harmonik antara presisi dan recall. Skor F1 berguna untuk memperhitungkan baik presisi maupun recall dalam satu metrik yang konsisten. Skor F1 dihitung dengan menggunakan formula:
$$\frac{2 \cdot (presisi \cdot recall)}{presisi + recall}$$

## Conclusion
- Membuat model random forest dengan menerapkan hyperparameter tuning yaitu menggunakan randomized search dapat mengidentifikasi pasien yang beresiko diabetes.
- Hasil `F1-Score` model random forest pada test set sebesar 97%.
