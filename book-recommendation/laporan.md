# Laporan Proyek Machine Learning - Sherly Santiadi

## Project Overview
Sistem rekomendasi dapat dimanfaatkan dalam kunjungan perpustakaan. Tujuan memanfaatkan sistem rekomendasi adalah supaya pengunjung dapat diberikan rekomendasi buku yang sesuai dengan minat bacaan pengunjung. Hal ini dapat memudahkan pustakawan untuk lebih efektif dan efisien dalam memberikan rekomendasi kepada pengunjung perpustakaan [1]. Penerapan sistem rekomendasi menggunakan *content based filtering* sudah pernah diimplementasikan dalam sistem rekomendasi film oleh Muhammad Fajriansyah, dkk. Di dalam jurnal, dituliskan bahwa untuk memndapatkan hasil rekomendasi dengan algoritma *content based filtering* perlu dilakukan pembobotan menggunakan metode TF-IDF dan juga *cosine similarity* dalam upaya mencari kesamaan kata kunci [2]. Oleh karena itu, dengan menerapkan teknologi *machine learning* yaitu *content-based filtering* diharapkan dapat membantu memberikan rekomendasi yang tepat berdasarkan historis pengguna ketika meminjam buku di perpustakaan.

## Business Understanding

### Problem Statements
- Bagaimana cara mengimplementasikan pembobotan menggunakan TF-IDF dalam dataset *Book Recommendation*?
- Bagaimana cara menghitung derajat kesamaan antar judul buku dengan teknik *cosine similarity*?
- Bagaimana cara mengimplementasikan algoritma *content based filtering*?

### Goals
- Mengimplementasikan pembobotan menggunakan TF-IDF dalam dataset *Book Recommendation*.
- Menghitung derajat kesamaan antar judul buku dengan teknik *cosine similarity*.
- Mengimplementasikan algoritma *content based filtering*.

### Solution statements
- Membangun algoritma *content based filtering* dengan menggunakan TF-IDF untuk pembobotan dan menghitung derajat kesamaan menggunakan *cosine similarity*.

## Data Understanding
Data yang digunakan dalam proyek ini bersumber dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

Di dalam proyek ini terdapat tiga buah dataset yang dapat digunakan yaitu:
- Books.csv
- Ratings.csv
- Users.csv
Ketiga dataset tersebut akan disimpan ke dalam tiga variabel berbeda yaitu `books`, `ratings`, dan `users`.

Variabel-variabel pada `Books.csv` adalah sebagai berikut:
- ISBN: Kode unik dari sebuah buku
- Book-Title: Judul buku
- Book-Author: Pengarang buku
- Year-Of-Publication: Tahun terbit buku
- Publisher: Penerbit buku
- Image-URL-S: Cover buku berukuran kecil
- Image-URL-M: Cover buku berukuran sedang
- Image-URL-L: Cover buku berukuran besar

Variabel-variabel pada `Ratings.csv` adalah sebagai berikut:
- User-ID: Kode unik dari pengguna yang memberikan penilaian
- ISBN: Kode unik dari sebuah buku
- Book-Rating: Penilaian buku nilai terendah dimulai dari angka 0

Variabel-variabel pada `Users.csv` adalah sebagai berikut:
- User-ID: Kode unik dari pengguna
- Location: Lokasi pengguna
- Age: Usia pengguna

## Exploratory Data Analysis
Tahapan yang akan dilakukan dalam mengeksplor dataset yaitu:
- *Gathering Data*

*Gathering Data* merupakan tahap awal sebelum membuat model *machine learning*, pada tahapan ini Penulis mencoba *load* dataset yang digunakan, menggabungkan dataset tersebut ke dalam `merged_df` dan memperhatikan secara general kolom serta baris seperti pada Tabel 1.

Tabel 1. Book Recommendation Dataset
| User-ID | ISBN       | Book-Rating | Book-Title           | Book-Author | Year-Of-Publication | Publisher        | Image-URL-S                                       | Image-URL-M                                       | Image-URL-L                                       | Location              | Age  |
|---------|------------|-------------|----------------------|-------------|---------------------|------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|-----------------------|------|
| 276725  | 034545104X | 0           | Flesh Tones: A Novel | M. J. Rose  | 2002                | Ballantine Books | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | tyler, texas, usa     | NaN  |
| 2313    | 034545104X | 5           | Flesh Tones: A Novel | M. J. Rose  | 2002                | Ballantine Books | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | cincinnati, ohio, usa | 23.0 |
| ...     | ...        | ...         | ...                  | ...         | ...                 | ...              | ...                                               | ...                                               | ...                                               | ...                   | ...  |

<sub>1031136 rows × 12 columns</sub>

Kemudian hal lainnya yang perlu diperhatikan adalah perhitungan statistika dari dataset yang digunakan seperti pada Tabel 2, Tabel 3, dan Tabel 4.

- Assessing Data
Apakah ada missing value? Pada dataset yang digunakan dalam proyek ini terdapat missing value sebanyak 6 baris data.
Apakah ada data duplikat? Pada dataset yang digunakan dalam proyek ini tidak terdapat duplikasi data.

- Describing Data
  - Count adalah jumlah sampel pada data.
  - Mean adalah nilai rata-rata.
  - Std adalah standar deviasi.
  - Min yaitu nilai minimum setiap kolom.
  - 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
  - 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
  - 75% adalah kuartil ketiga.
  - Max adalah nilai maksimum.

Tabel 2. Hasil Perhitungan Statistik Dataset pada `Books.csv`
|       | Year-Of-Publication |
|------:|---------------------|
| count |        20000.000000 |
|  mean |         1961.527000 |
|   std |          257.291152 |
|   min |            0.000000 |
|   25% |         1991.000000 |
|   50% |         1997.000000 |
|   75% |         2001.000000 |
|   max |         2011.000000 |

Tabel 3. Hasil Perhitungan Statistik Dataset pada `Ratings.csv`
|       | User-ID      | Book-Rating  |
|------:|-------------:|--------------|
| count | 1.149780e+06 | 1.149780e+06 |
|  mean | 1.403864e+05 | 2.866950e+00 |
|   std | 8.056228e+04 | 3.854184e+00 |
|   min | 2.000000e+00 | 0.000000e+00 |
|   25% | 7.034500e+04 | 0.000000e+00 |
|   50% | 1.410100e+05 | 0.000000e+00 |
|   75% | 2.110280e+05 | 7.000000e+00 |
|   max | 2.788540e+05 | 1.000000e+01 |

Tabel 4. Hasil Perhitungan Statistik Dataset pada `Users.csv`
|       | User-ID      | Age           |
|-------|--------------|---------------|
| count | 278858.00000 | 168096.000000 |
| mean  | 139429.50000 | 34.751434     |
| std   | 80499.51502  | 14.428097     |
| min   | 1.00000      | 0.000000      |
| 25%   | 69715.25000  | 24.000000     |
| 50%   | 139429.50000 | 32.000000     |
| 75%   | 209143.75000 | 44.000000     |
| max   | 278858.00000 | 244.000000    |

- Melihat Distribusi Kolom

Pada Gambar 1 merupakan salah satu contoh visualisasi dari hasil *grouping* `Book-Author` dan menghitung jumlah buku yang ditulis oleh masing-masing penulis, dari hasil visualisasi di bawah ini dapat diketahui bahwa Penulis Agatha Christie menulis paling banyak buku yaitu sebanyak > 600 buku.

![20_penulis_teratas](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/455de535-fb83-449b-bab6-f5990cbcf24f)
Gambar 1. Dua Puluh Penulis Teratas Berdasarkan Jumlah Buku

Pada Gambar 2 merupakan salah satu contoh visualisasi dari hasil *grouping* `Year-Of-Publication` dan menghitung jumlah buku yang diterbitkan, dari hasil visualisasi di bawah ini dapat diketahui bahwa pada tahun 2022 jumlah buku yang diterbitkan paling banyak yaitu sebanyak > 17.500 buku.

![20_tahun_terbit_teratas](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/c18f89b9-ec6e-4c2b-85c5-505f74b28b38)
Gambar 2. Dua Puluh Tahun Terbit Teratas Berdasarkan Jumlah Buku

Pada Gambar 3 merupakan salah satu contoh visualisasi dari hasil *grouping* `Book-Rating` dan menghitung jumlah buku yang diberikan rating dari 0-10, dari hasil visualisasi di bawah ini dapat diketahui bahwa rating buku sebagian besar mendapaptkan rating 0.

![rating_buku](https://github.com/sntdshrly/applied-machine-learning-projects/assets/71547739/27c0c568-7553-4a7f-bf13-039de8085450)
Gambar 3. Rating Buku

Pada Tabel 5 merupakan salah satu contoh visualisasi dari hasil *grouping* `Book-Rating` dan menghitung rata-rata rating yang diberikan rating dari 0-10, dari hasil visualisasi di bawah ini dapat diketahui bahwa lima judul buku yang diberikan rating tertinggi.

Tabel 5. Rating Buku Tertinggi
| Book-Title                                                                                                           | Rating |
|----------------------------------------------------------------------------------------------------------------------|--------|
| Dark Justice                                                                                                         | 10.0   |
| California Historical Landmarks                                                                                      | 10.0   |
| Isms: a dictionary of words ending in -ism, -ology, and -phobia,: With some similar terms, arranged in subject order | 10.0   |
| Round the Corner (Sister Circle)                                                                                     | 10.0   |
| 006781: Bk.1 Gags De Boule Et Bil                                                                                    | 10.0   |

Pada Tabel 6 merupakan salah satu contoh visualisasi dari hasil *grouping* `Book-Rating` dan menghitung rata-rata rating yang diberikan rating dari 0-10, dari hasil visualisasi di bawah ini dapat diketahui bahwa lima judul buku yang diberikan rating terendah.

Tabel 6. Rating Buku Terendah
| Book-Title                                                                                                           | Rating |
|----------------------------------------------------------------------------------------------------------------------|--------|
| Dark Justice                                                                                                         | 10.0   |
| California Historical Landmarks                                                                                      | 10.0   |
| Isms: a dictionary of words ending in -ism, -ology, and -phobia,: With some similar terms, arranged in subject order | 10.0   |
| Round the Corner (Sister Circle)                                                                                     | 10.0   |
| 006781: Bk.1 Gags De Boule Et Bil                                                                                    | 10.0   |

## Data Preparation
Tahapan yang akan dilakukan dalam mempersiapkan dataset yaitu:
- Mengubah Tipe Data: Di dalam proyek ini diketahui bahwa value pada `Year-Of-Publication`ada yang bernilai 'DK Publishing Inc' dan 'Gallimard'. Sepertinya terdapat kesalahan input. Setelah dilihat, ternyata ada 3 baris data yang salah input. Oleh karena itu, pada proyek ini data-data yang salah input akan dibuang.
- Mengkaji Kolom `Age`: Di dalam proyek ini, kolom `Age` memiliki *missing value* sebanyak 110.762 data dikarenakan jumlahnya yang begitu banyak maka kolom ini akan dibuang.
- Mengkaji Dataset: Di dalam proyek ini terdapat 3 baris yang memiliki *missing value* oleh karena jumlah yang tidak begitu banyak, maka data yang memiliki *missing value* akan dibuang.
- Mengonversi Data Series Menjadi Bentuk List: Tahap ini dilakukan untuk memasukan list tersebut ke dalam dictionary.
- Membuat Dictionary: Digunakan untuk menentukan pasangan *key-value* pada dataset.

## Modeling
Pada tahapan pemodelan ini, Penulis menggunakan algoritma *Content-based Filerting* untuk menyelesaikan permasalahan dalam proyek "Book Recommendation".

- TF-IDF Vectorizer: Digunakan untuk melakukan pembobotan frekuensi kata tehadap *inverse document frequency*. Nilai di dalam vektor akan merepresentasikan seberapa penting kata tertentu terhadapt dokumennya.
- Cosine Similarity: Digunakan untuk menghitung kesamaan antar vektor, sehingga apabila skor cosine semakin tinggi, maka buku tersebut semakin baik untuk direkomendasikan
- Membangun Fungsi `content_based_filtering`: Membangun fungsi dengan menerapkan TF-IDF dan Cosine similarity, selanjutnya mengurutkan output yang dihasilkan sebanyak 5 rekomendasi buku dari nilai tertinggi ke terendah.

### Hasil Top-5 Recommendation
Judul buku: "Chicken Soup for the Preteen Soul - 101 Stories of Changes, Choices and Growing Up for Kids, ages 10-13"

Tabel 7. Hasil Top-5 Recommendation
| Recommended Books                                 | Similarity |
|---------------------------------------------------|------------|
| A 6th Bowl of Chicken Soup for the Soul (Chick... | 1.0        |
| Chicken Soup for the Mother's Soul 2 : 101 Mor... | 1.0        |
| Chicken Soup for the Soul                         | 1.0        |
| A Second Chicken Soup for the Woman's Soul (Ch... | 1.0        |
| Chicken Soup from the Soul of Hawaii: Stories ... | 1.0        |


## Evaluation
Dalam proyek ini, Penulis menggunakan beberapa metrik evaluasi untuk kasus rekomendasi, yaitu akurasi, presisi, recall, dan skor F1. Berikut adalah penjelasan singkat mengenai metrik-metrik tersebut:

---
**Daftar Pustaka**

[1] [S. Saefudin and D. Fernando, “PENERAPAN DATA MINING REKOMENDASI BUKU MENGGUNAKAN ALGORITMA APRIORI,” JSiI (Jurnal Sistem Informasi), vol. 7, no. 1, p. 50, Mar. 2020, doi: https://doi.org/10.30656/jsii.v7i1.1899‌.](https://core.ac.uk/download/pdf/327232562.pdf)

[2] [M. Fajriansyah, P. P. Adikara, and A. W. Widodo, "Sistem Rekomendasi Film Menggunakan Content Based Filtering," Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer, vol. 5, no. 6, pp. 2188-2199, 2021.](https://jptiik.multi.web.id/index.php/j-ptiik/article/download/9163/4159)
