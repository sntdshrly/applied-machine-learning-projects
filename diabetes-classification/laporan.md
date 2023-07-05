# Laporan Proyek Machine Learning - Sherly Santiadi


## Domain Proyek
Diabetes merupakan penyakit serius sehingga pentingnya untuk mendeteksi sedini mungkin penyakit diabetes berdasarkan latar belakang historis sehingga dapat diberi perawatan yang tepat. Selain itu, penderita diabetes beresiko lebih tinggi pada kasus pasien COVID-19 [1]. WHO sendiri memprediksi bahwa jumlah penderita diabetes dari rentang waktu tahun 1980 sampai dengan 2014 meningkat kurang lebih empat kali lebih tinggi [2]. Oleh karena itu, penting sekali untuk menanggulani permasalahan diabetes ini. Dengan menerapkan teknologi *machine learning* untuk memprediksi berdasarkan historis pengguna akan lebih mudah untuk mendeteksi permasalahan diabetes sedini mungkin.


## Business Understanding

### Problem Statements

- Bagaimana membuat model *machine learning* untuk mengidentifikasi pasien yang beresiko diabetes?
- Bagaimana mengukur performa model untuk memprediksi pasien yang beresiko diabetes?

### Goals
- Membuat model untuk mengidentifikasi pasien yang beresiko diabetes.
- Mengukur performa model untuk memprediksi pasien yang beresiko diabetes.

### Solution statements
- Membangun model *random forest* dengan menerapkan *hyperparameter tuning*.
- Mengukur performa model dengan menggunakan metrik evaluasi.


## Data Understanding
Data yang digunakan dalam proyek ini bersumber dari [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

### Variabel-variabel pada Diabetes Dataset adalah sebagai berikut:
- `gender`: mengacu pada jenis kelamin biologis individu, yang dapat berdampak pada kerentanan mereka terhadap diabetes. Ada tiga kategori di dalamnya pria, wanita dan lainnya.
- `age`: merupakan faktor penting karena diabetes lebih sering didiagnosis pada orang yang lebih tua. Rentang usia dari 0-80.
- `hypertension`: merupakan suatu kondisi medis di mana tekanan darah di arteri terus-menerus meningkat. Di dalam dataset ini terdapat nilai 0 atau 1 di mana 0 menunjukkan mereka tidak memiliki hipertensi dan untuk 1 itu berarti mereka memiliki hipertensi.
- `heart_disease`: merupakan suatu kondisi medis lain yang dikaitkan dengan peningkatan risiko diabetes. Di dalam dataset ini terdapat nilai 0 atau 1 dimana 0 menunjukkan mereka tidak memiliki penyakit jantung dan untuk 1 itu berarti mereka memiliki penyakit jantung.
- `smoking_history`: riwayat merokok juga dianggap sebagai faktor risiko diabetes dan dapat memperburuk komplikasi yang terkait dengan diabetes. Dalam dataset, terdapat 5 kategori yaitu *never*, *No Info*, *current*, *former*, *not current*.
- `bmi`: merupakan ukuran lemak tubuh berdasarkan berat dan tinggi badan. Nilai BMI yang lebih tinggi terkait dengan risiko diabetes yang lebih tinggi.
  - Kisaran BMI dalam dataset adalah dari 10.16 hingga 71.55.
  - BMI kurang dari 18.5 adalah kurus
  - BMI 18.5-24.9 adalah normal
  - BMI 25-29.9 adalah kelebihan berat badan
  - BMI 30 atau lebih adalah obesitas.
- `HbA1c_level`: kadar HbA1c (Hemoglobin A1c) adalah ukuran kadar gula darah rata-rata seseorang selama 2-3 bulan terakhir. Tingkat yang lebih tinggi menunjukkan risiko lebih besar terkena diabetes. Sebagian besar lebih dari 6.5% dari Tingkat HbA1c menunjukkan diabetes.
- `blood_glucose_level`: tingkat glukosa darah mengacu pada jumlah glukosa dalam aliran darah pada waktu tertentu. Kadar glukosa darah yang tinggi merupakan indikator utama diabetes.
- `diabetes`: diabetes adalah variabel target yang akan diprediksi, dengan nilai 1 menunjukkan adanya diabetes dan 0 menunjukkan tidak adanya diabetes.

## Exploratory Data Analysis
Tahapan yang akan dilakukan dalam mengeksplor dataset yaitu:
- *Gathering Data*

*Gathering Data* merupakan tahap awal sebelum membuat model *machine learning*, pada tahapan ini Penulis mencoba *load* dataset yang digunakan dan memperhatikan secara general kolom serta baris seperti pada Tabel 1.

Tabel 1. Diabetes Dataset

| gender | age  | hypertension | heart_disease | smoking_history | bmi   | HbA1c_level | blood_glucose_level | diabetes |
|--------|------|--------------|---------------|-----------------|-------|-------------|---------------------|----------|
| Female | 80.0 | 0            | 1             | never           | 25.19 | 6.6         | 140                 | 0        |
| Male   | 54.0 | 1            | 0             | No Info         | 27.32 | 5.7         | 80                  | 1        |
| ...    | ...  | ...          | ...           | ...             | ...   | ...         | ...                 | ...      |

<sub>100000 rows × 9 columns</sub>

Kemudian hal lainnya yang perlu diperhatikan adalah perhitungan statistika dari dataset yang digunakan seperti pada Tabel 2.
- *Assessing Data*
  - Apakah ada *missing value*? Pada dataset yang digunakan dalam proyek ini tidak ada *missing value*.
  - Apakah ada data duplikat? Pada dataset yang digunakan dalam proyek ini terdapat duplikasi data sebaganyak 3854 data.
- *Describing Data*
  - Count adalah jumlah sampel pada data.
  - Mean adalah nilai rata-rata.
  - Std adalah standar deviasi.
  - Min yaitu nilai minimum setiap kolom.
  - 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
  - 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
  - 75% adalah kuartil ketiga.
  - Max adalah nilai maksimum.

Tabel 2. Hasil Perhitungan Statistik Dataset

|       | age          | ... | diabetes     |
|-------|--------------|-----|--------------|
| count | 96146.000000 | ... | 96146.000000 |
| mean  | 41.794326    | ... | 0.088220     |
| std   | 22.462948    | ... | 0.283616     |
| min   | 0.080000     | ... | 0.000000     |
| 25%   | 24.000000    | ... | 0.000000     |
| 50%   | 43.000000    | ... | 0.000000     |
| 75%   | 59.000000    | ... | 0.000000     |
| max   | 80.000000    | ... | 1.000000     |

- Melihat Distribusi Kolom

Pada Gambar 1 merupakan salah satu contoh visualisasi menggunakan *box plot*, dari hasil visualisasi di bawah ini dapat diketahui bahwa  rentang umur seseorang mengalami diabetes yaitu diantara 50 - 80.

![Box Plot](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/00da1487-03fb-4707-b1d3-9f17b6dd3cc9)

Gambar 1. Distribusi Kolom dengan Variabel Diabetes dan Age

Selain itu, Penulis juga melakukan visualiasi terhadap variabel `gender` seperti pada Gambar 2. Visualisasi pada Gambar 2, merupakan hasil manipulasi data yang sebelumnya data tersebut memiliki tiga *unique value* yaitu:
- Female sebanyak 56161 data.
- Male sebanyak 39967 data.
- Other sebanyak 18 data.
Namun, karena pada nilai `other` proporsi datanya relatif sedikit dibanding `female` dan `male` maka nilai tersebut dihapus dari dataframe.

![Data Gender](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/4e684737-d550-4e1c-8379-8490448cd18b)

Gambar 2. Distribusi Gender

Gambar 3 merupakan hasil visualisasi terhadap kolom `age` untuk mengetahui distribusi atau sebaran data yang digunakan dalam proyek ini. Hasilnya adalah rentang usia yang ada pada proyek ini tersebar dari 0 hingga 80.

![Data Age](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/9c55646a-2a80-4dad-8192-c1272f0b2a30)


Gambar 3. Distribusi Age

Gambar 4 menunjukan bahwa label di dalam proyek sangatlah tidak seimbang, oleh karena itu pada tahap selanjutnya akan dilakukan *resampling* dataset untuk menangani masalah *imbalanced dataset*.

![Data Diabetes](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/10d588ee-433e-44eb-abc3-bb018112993b)


Gambar 4. Distribusi Diabetes

Gambar 5 menunjukan korelasi atau hubungan antar variabel, hubungan yang kuat ditandai dengan nilai yang mendekati +1 (korelasi positif) atau -1 (korelasi negatif) sedangkan, tidak adanya korelasi antar variabel ditandai dengan nilai mendekati 0. Dari matriks korelasi ditemukan bahwa kolom yang memiliki korelasi positif dengan kolom diabetes diurutkan dari yang paling kuat hingga lemah sebagai berikut:

1. `blood_glucose_level`
2. `HbA1c_level`
3. `age`
4. `bmi`
5. `hypertension`
6. `heart_disease`


![Matriks Korelasi](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/a04645ff-127c-4fbd-bfb4-5fc29a298ea1)


Gambar 5. Matriks Korelasi


## Data Preparation

Tahapan yang akan dilakukan dalam mempersiapkan dataset yaitu:
- *One Hot Encoding*: mengubah variabel kategorikal menjadi representasi numerik yang sesuai, di dalam dataset yang digunakan dalam proyek ini terdapat dua kolom kategorikal yaitu kolom `gender` dan kolom `smoking_history`. Hal ini penting untuk memastikan bahwa semua variabel dalam dataset bersifat numerik agar dapat digunakan dalam model.
- Standarisasi: mengubah skala variabel numerik agar memiliki mean (rerata) nol dan standar deviasi satu. Tujuan standarisasi adalah untuk menghilangkan perbedaan skala antar variabel, sehingga variabel dengan skala yang lebih besar tidak mendominasi kontribusinya dalam pembelajaran mesin. Standarisasi juga membantu meningkatkan interpretasi model, karena koefisien yang dihasilkan dapat dibandingkan secara langsung. Standarisasi akan diimplementasikan ke dalam kolom numerik yaitu `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`. Dengan mengimplementasikan standarisasi dapat dilihat pada mean bernilai 0 dan standar deviasi bernilai 1.
- *Resampling* menggunakan SMOTE dan *Random Under Sampler*: mengubah distribusi kelas pada dataset dengan tujuan untuk mengatasi ketidakseimbangan kelas (*class imbalance*). SMOTE (*Synthetic Minority Over-sampling Technique*) adalah metode resampling yang membuat sampel sintetis untuk kelas minoritas dengan menggabungkan fitur-fitur dari sampel kelas minoritas yang ada. Dengan menggunakan SMOTE, jumlah sampel pada kelas minoritas akan meningkat sehingga dapat mengurangi ketidakseimbangan kelas. Metode ini membantu meningkatkan kinerja model dalam memprediksi kelas minoritas. Sedangkan, Random Under Sampler adalah metode resampling yang secara acak mengurangi jumlah sampel pada kelas mayoritas agar sebanding dengan jumlah sampel pada kelas minoritas. Dengan mengurangi jumlah sampel pada kelas mayoritas, metode ini membantu mengatasi ketidakseimbangan kelas dengan mempertahankan sampel pada kelas minoritas yang ada. Tujuan dari Random Under Sampler adalah untuk mencegah *overfitting* pada kelas mayoritas dan meningkatkan performa model dalam memprediksi kelas minoritas.

## Modeling
Pada tahapan pemodelan ini, Penulis menggunakan algoritma **Random Forest Classifier** untuk menyelesaikan permasalahan dalam proyek "Diabetes Classification".

Tahapan yang dilakukan dalam pemodelan ini adalah sebagai berikut:
- Membentuk model awal: Pertama, Penulis membuat model awal dengan menggunakan RandomForestClassifier tanpa mengatur parameter apapun. Model awal ini digunakan untuk melatih data yang telah diresampling `(X_train_resampled)` dan `(y_train_resampled)`.
- Penentuan parameter: Setelah itu, Penulis menentukan parameter-parameter yang akan digunakan pada model Random Forest. Dalam contoh ini, Penulis menggunakan RandomizedSearchCV untuk mencari parameter terbaik dengan mencoba beberapa kombinasi secara acak. Parameter yang diuji adalah jumlah estimators `(n_estimators)`, kedalaman maksimum `(max_depth)`, minimum sampel split `(min_samples_split)`, dan minimum sampel leaf `(min_samples_leaf)`. Parameter-parameter ini digunakan untuk meningkatkan performa model dan menghindari overfitting.
- Pencarian parameter terbaik: Pada langkah ini, RandomizedSearchCV mencoba beberapa kombinasi parameter yang dijelaskan di atas dengan melakukan validasi silang `(cross-validation)` menggunakan 5-fold cross-validation `(cv=5)`. Dalam hal ini, RandomizedSearchCV mencoba 10 kombinasi parameter yang berbeda `(n_iter=10)` dan mencatat performa model untuk setiap kombinasi.
- Memilih model terbaik: Setelah melakukan pencarian parameter, Penulis memilih model terbaik berdasarkan performa yang diukur. Model terbaik ini akan digunakan untuk melatih data yang telah dilakukan one hot encoding `(X_train_encoded)` dan target `(y_train)`.

## Hyperparameter Tuning
Proses *improvement* melalui *hyperparameter tuning*:

- Dalam langkah hyperparameter tuning, **RandomizedSearchCV** digunakan untuk mencari kombinasi terbaik dari parameter-parameter yang diuji. Dalam contoh ini, Penulis mencari nilai terbaik untuk n_estimators, max_depth, min_samples_split, dan min_samples_leaf.
- Dengan melakukan hyperparameter tuning, Penulis dapat meningkatkan performa model dengan menemukan parameter yang optimal. Parameter yang optimal dapat membantu menghindari overfitting, meningkatkan akurasi, dan menghasilkan model yang lebih baik secara keseluruhan.
- Setelah proses hyperparameter tuning, Penulis mendapatkan model terbaik `(rf_best)` yang kemudian dilatih menggunakan data yang telah dilakukan one hot encoding `(X_train_encoded)` dan target `(y_train)`. Model terbaik ini kemudian dapat digunakan untuk melakukan prediksi pada data yang baru.

## Evaluation
Dalam proyek ini, Penulis menggunakan beberapa metrik evaluasi untuk kasus klasifikasi, yaitu akurasi, presisi, recall, dan skor F1. Berikut adalah penjelasan singkat mengenai metrik-metrik tersebut:

- Akurasi: Metrik ini mengukur sejauh mana model dapat mengklasifikasikan data dengan benar secara keseluruhan. Akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total jumlah prediksi.
- Presisi: Presisi mengukur sejauh mana model memberikan hasil positif yang benar dari semua prediksi positif yang dilakukan. Presisi dihitung dengan membagi jumlah prediksi positif yang benar dengan jumlah total prediksi positif.
- Recall: Recall, juga dikenal sebagai sensitivitas, mengukur sejauh mana model dapat mengidentifikasi secara benar semua instance positif yang ada dalam dataset. Recall dihitung dengan membagi jumlah prediksi positif yang benar dengan jumlah total instance positif yang sebenarnya.
- F1 Score: F1 Score adalah rata-rata harmonik antara presisi dan recall. Skor F1 berguna untuk memperhitungkan baik presisi maupun recall dalam satu metrik yang konsisten. Skor F1 dihitung dengan menggunakan formula:
$$\frac{2 \cdot (presisi \cdot recall)}{presisi + recall}$$

## Conclusion

- Model *random forest* dengan menerapkan *hyperparameter tuning* (*randomized search*) dapat mengidentifikasi pasien yang beresiko diabetes.
- Hasil F1-Score model random forest pada test set sebesar 97% dapat dilihat pada Gambar 6.

![Grafik ROC](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/14979018-24b1-4a42-8766-2ac7786687c5)


Gambar 6. Grafik ROC

Berdasarkan hasil proyek dan metrik evaluasi yang digunakan seperti pada Tabel 3, dapat dilihat bahwa model yang telah dilatih memberikan hasil yang baik. Berikut adalah analisis berdasarkan metrik-metrik evaluasi yang digunakan:

- Akurasi: Model memiliki akurasi sebesar 97% pada data validasi (15235 sampel) dan 97% pada data pengujian (19044 sampel). Ini berarti model dapat mengklasifikasikan data dengan benar dengan tingkat akurasi yang tinggi.
- Presisi: Model memiliki presisi sebesar 97% untuk kelas 0 (tidak diabetes) dan 98% untuk kelas 1 (diabetes) pada data validasi dan data pengujian. Hal ini menunjukkan bahwa model memberikan sedikit sekali kesalahan dalam mengklasifikasikan data sebagai kelas 0 atau kelas 1.
- Recall: Model memiliki recall sebesar 100% untuk kelas 0 dan 68% untuk kelas 1 pada data validasi dan data pengujian. Ini berarti model mampu mengidentifikasi secara benar semua instance yang termasuk dalam kelas 0, tetapi memiliki tingkat recall yang lebih rendah untuk kelas 1.
- F1 Score: F1 Score untuk kelas 0 adalah 98% pada data validasi dan data pengujian, sementara untuk kelas 1 adalah 81% pada data validasi dan data pengujian. F1 Score yang tinggi menunjukkan keseimbangan antara presisi dan recall.

Secara keseluruhan, model *Random Forest Classifier* yang telah dilatih memberikan hasil yang baik dengan tingkat akurasi yang tinggi dan presisi yang baik untuk kedua kelas. Namun, terdapat penurunan recall dan F1 Score untuk kelas 1, yang dapat menunjukkan bahwa **model memiliki kesulitan dalam mengidentifikasi instance-instance yang termasuk dalam kelas 1**.

Tabel 3. Metrik Evaluasi

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.97      | 1.00   | 0.98     | 17349   |
| 1            | 0.98      | 0.68   | 0.81     | 1695    |
| accuracy     |           |        | 0.97     | 19044   |
| macro avg    | 0.98      | 0.84   | 0.90     | 19044   |
| weighted avg | 0.97      | 0.97   | 0.97     | 19044   |


---

**Daftar Pustaka**

[1] [A. A. Panua, R. Zainuddin, Ekayanti Hafidah Ahmad, and Fitriani Sangkala, “Faktor Risiko Terjadinya Covid-19 Pada Penderita Diabetes Melitus Tipe 2,” vol. 10, no. 2, pp. 624–634, Dec. 2021, doi: https://doi.org/10.35816/jiskh.v10i2.668.](https://bapin-ismki.e-journal.id/jimki/article/view/342)

[2] [Sartika Sumangkut, Wenny Supit, and Franly Onibala, JURNAL KEPERAWATAN, vol. 1, no. 1, 2013, doi: https://doi.org/10.35790/jkp.v1i1.2235.‌](https://ejournal.unsrat.ac.id/v3/index.php/jkp/article/view/2235)
