# Laporan Proyek Machine Learning - Sherly Santiadi

## Project Overview
Sistem rekomendasi dapat dimanfaatkan dalam kunjungan perpustakaan. Tujuan memanfaatkan sistem rekomendasi adalah supaya pengunjung dapat diberikan rekomendasi buku yang sesuai dengan minat bacaan pengunjung. Dengan menggunakan sistem ini, pengunjung dapat menemukan buku yang relevan sesuai dengan preferensi pengunjung. Hal ini dapat memudahkan pustakawan untuk lebih efektif dan efisien dalam memberikan rekomendasi kepada pengunjung perpustakaan [1]. Pustakawan memiliki tanggung jawab untuk memasukan data-data buku ke dalam sistem seperti informasi judul buku, pengarang, dan lain sebagainya untuk membuat sistem menjadi semakin *up-to-date*, selain itu pustakawan dapat memberikan panduang terkait bagaimana cara menggunakan sistem rekomendasi. Penerapan sistem rekomendasi menggunakan *content based filtering* sudah pernah diimplementasikan dalam sistem rekomendasi film oleh Muhammad Fajriansyah, dkk. Di dalam jurnal, dituliskan bahwa untuk memndapatkan hasil rekomendasi dengan algoritma *content based filtering* perlu dilakukan pembobotan menggunakan metode TF-IDF dan juga *cosine similarity* dalam upaya mencari kesamaan kata kunci [2]. Oleh karena itu, dengan menerapkan teknologi *machine learning* yaitu *content-based filtering* diharapkan dapat membantu memberikan rekomendasi yang tepat berdasarkan historis pengguna ketika meminjam buku di perpustakaan.

## Business Understanding

### Problem Statements
- Bagaimana cara menghitung pembobotan menggunakan TF-IDF untuk setiap kata terhadap frekuensi kemunculan kata tersebut di dalam dataset *Book Recommendation*?
- Bagaimana cara merepresentasikan vektorisasi judul buku sehingga dapat menentukan kemunculan kata dalam judul buku kemudian menggunakan vektor tersebut untuk menghitung derajat kesamaan antar judul buku dengan teknik *cosine similarity*?
- Bagaimana cara mengimplementasikan algoritma *content based filtering* menggunakan TF-IDF dan *cosine similarity* untuk menghasilkan rekomendasi buku yang sesuai berdasarkan histori buku yang pernah dipinjam pengguna?

### Goals
- Menghitung pembobotan menggunakan TF-IDF untuk setiap kata terhadap frekuensi kemunculan kata tersebut di dalam dataset *Book Recommendation*.
- Merepresentasikan vektorisasi judul buku sehingga dapat menentukan kemunculan kata dalam judul buku kemudian menggunakan vektor tersebut untuk menghitung derajat kesamaan antar judul buku dengan teknik *cosine similarity*.
- Mengimplementasikan algoritma *content based filtering* menggunakan TF-IDF dan *cosine similarity* untuk menghasilkan rekomendasi buku yang sesuai berdasarkan histori buku yang pernah dipinjam pengguna.

### Solution statements
- Membangun algoritma *content based filtering* dengan menggunakan TF-IDF untuk pembobotan dan menghitung derajat kesamaan menggunakan *cosine similarity*.

## Data Understanding
Data yang digunakan dalam proyek ini bersumber dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset terbut diperoleh dari **Amazon Web Services**, buku-buku yang dipakai adalah buku-buku yang memiliki ISBN valid dan tautan URL yang diberikan akan mengarah juga akan mengarah ke website Amazon.

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

Tabel 1. Dataset `Books.csv`
| ISBN       | Book-Title                                        | Book-Author          | Year-Of-Publication | Publisher                  | Image-URL-S                                       | Image-URL-M                                       | Image-URL-L                                       |
|------------|---------------------------------------------------|----------------------|---------------------|----------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| 0195153448 | Classical Mythology                               | Mark P. O. Morford   | 2002                | Oxford University Press    | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... |
| 0002005018 | Clara Callan                                      | Richard Bruce Wright | 2001                | HarperFlamingo Canada      | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... |
| 0060973129 | Decision in Normandy                              | Carlo D'Este         | 1991                | HarperPerennial            | http://images.amazon.com/images/P/0060973129.0... | http://images.amazon.com/images/P/0060973129.0... | http://images.amazon.com/images/P/0060973129.0... |
| 0374157065 | Flu: The Story of the Great Influenza Pandemic... | Gina Bari Kolata     | 1999                | Farrar Straus Giroux       | http://images.amazon.com/images/P/0374157065.0... | http://images.amazon.com/images/P/0374157065.0... | http://images.amazon.com/images/P/0374157065.0... |
| 0393045218 | The Mummies of Urumchi                            | E. J. W. Barber      | 1999                | W. W. Norton &amp; Company | http://images.amazon.com/images/P/0393045218.0... | http://images.amazon.com/images/P/0393045218.0... | http://images.amazon.com/images/P/0393045218.0... |
| ...        | ...                                               | ...                  | ...                 | ...                        | ...                                               | ...                                               | ...                                               |

<sub>271360 rows × 8 columns</sub>

Variabel-variabel pada `Ratings.csv` adalah sebagai berikut:
- User-ID: Kode unik dari pengguna yang memberikan penilaian
- ISBN: Kode unik dari sebuah buku
- Book-Rating: Penilaian buku nilai terendah dimulai dari angka 0

Tabel 2. Dataset `Ratings.csv`
| User-ID | ISBN       | Book-Rating |
|---------|------------|-------------|
| 276725  | 034545104X | 0           |
| 276726  | 0155061224 | 5           |
| 276727  | 0446520802 | 0           |
| 276729  | 052165615X | 3           |
| 276729  | 0521795028 | 6           |
| ...     | ...        | ...         |

<sub>1149780 rows × 3 columns</sub>

Variabel-variabel pada `Users.csv` adalah sebagai berikut:
- User-ID: Kode unik dari pengguna
- Location: Lokasi pengguna
- Age: Usia pengguna

Tabel 3. Dataset `Users.csv`
| User-ID | Location                           | Age  |
|---------|------------------------------------|------|
| 1       | nyc, new york, usa                 | NaN  |
| 2       | stockton, california, usa          | 18.0 |
| 3       | moscow, yukon territory, russia    | NaN  |
| 4       | porto, v.n.gaia, portugal          | 17.0 |
| 5       | farnborough, hants, united kingdom | NaN  |
| ...     | ...                                | ...  |

<sub>278858 rows × 3 columns</sub>

## Exploratory Data Analysis
Tahapan yang akan dilakukan dalam mengeksplor dataset yaitu:
- *Gathering Data*

*Gathering Data* merupakan tahap awal sebelum membuat model *machine learning*, pada tahapan ini Penulis mencoba *load* dataset yang digunakan, menggabungkan dataset tersebut ke dalam `merged_df` dan memperhatikan secara general kolom serta baris seperti pada Tabel 4.

Tabel 4. Book Recommendation Dataset
| User-ID | ISBN       | Book-Rating | Book-Title           | Book-Author | Year-Of-Publication | Publisher        | Image-URL-S                                       | Image-URL-M                                       | Image-URL-L                                       | Location              | Age  |
|---------|------------|-------------|----------------------|-------------|---------------------|------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|-----------------------|------|
| 276725  | 034545104X | 0           | Flesh Tones: A Novel | M. J. Rose  | 2002                | Ballantine Books | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | tyler, texas, usa     | NaN  |
| 2313    | 034545104X | 5           | Flesh Tones: A Novel | M. J. Rose  | 2002                | Ballantine Books | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | http://images.amazon.com/images/P/034545104X.0... | cincinnati, ohio, usa | 23.0 |
| ...     | ...        | ...         | ...                  | ...         | ...                 | ...              | ...                                               | ...                                               | ...                                               | ...                   | ...  |

<sub>1031136 rows × 12 columns</sub>

Kemudian hal lainnya yang perlu diperhatikan adalah perhitungan statistika dari dataset yang digunakan seperti pada Tabel 5, Tabel 6, dan Tabel 7.

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

Tabel 5. Hasil Perhitungan Statistik Dataset pada `Books.csv`
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

Tabel 6. Hasil Perhitungan Statistik Dataset pada `Ratings.csv`
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

Tabel 7. Hasil Perhitungan Statistik Dataset pada `Users.csv`
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

Pada Tabel 8 merupakan salah satu contoh visualisasi dari hasil *grouping* `Book-Rating` dan menghitung rata-rata rating yang diberikan rating dari 0-10, dari hasil visualisasi di bawah ini dapat diketahui bahwa lima judul buku yang diberikan rating tertinggi.

Tabel 8. Rating Buku Tertinggi
| Book-Title                                                                                                           | Rating |
|----------------------------------------------------------------------------------------------------------------------|--------|
| Dark Justice                                                                                                         | 10.0   |
| California Historical Landmarks                                                                                      | 10.0   |
| Isms: a dictionary of words ending in -ism, -ology, and -phobia,: With some similar terms, arranged in subject order | 10.0   |
| Round the Corner (Sister Circle)                                                                                     | 10.0   |
| 006781: Bk.1 Gags De Boule Et Bil                                                                                    | 10.0   |

Pada Tabel 9 merupakan salah satu contoh visualisasi dari hasil *grouping* `Book-Rating` dan menghitung rata-rata rating yang diberikan rating dari 0-10, dari hasil visualisasi di bawah ini dapat diketahui bahwa lima judul buku yang diberikan rating terendah.

Tabel 9. Rating Buku Terendah
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
- Mengkaji Kolom `Age`: Di dalam proyek ini, kolom `Age` memiliki *missing value* sebanyak 110.762 data. Untuk menangani *missing value* terdapat berbagai cara seperti mengisi dengan nilai tertentu misal dengan nilai 0 atau nilai rata-rata, bisa juga dengan menggunakan hasil prediksi dengan model tertentu seperti dengan algoritma *K-Nearest Neighbors* namun dikarenakan jumlah *missing value* yang begitu banyak dan juga kolom `Age` tidak akan digunakan dalam proyek ini maka kolom ini akan dibuang.
- Mengkaji Dataset: Di dalam proyek ini terdapat 3 baris yang memiliki *missing value* oleh karena jumlah yang tidak begitu banyak, maka data yang memiliki *missing value* akan dibuang.
- Mengonversi Data Series Menjadi Bentuk List: Tahap ini dilakukan untuk memasukan list tersebut ke dalam dictionary.

  ```
  isbn = books['ISBN'].tolist()
  ```

  Snippet di atas digunakan untuk mengkonversi kolom `ISBN` dari DataFrame `books` menjadi list. Proses yang sama dilakukan juga untuk kolom `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L` menggunakan metode `tolist()`.

- Membuat Dictionary: Digunakan untuk menentukan pasangan *key-value* pada dataset.

## Modeling
Pada tahapan pemodelan ini, Penulis menggunakan algoritma *Content-based Filerting* untuk menyelesaikan permasalahan dalam proyek "Book Recommendation".

- TF-IDF Vectorizer
  - Menggunakan fungsi `fit_transform()` pada objek TF-IDF Vectorizer untuk melakukan pembobotan frekuensi kata menggunakan metode *Term Frequency-Inverse Document Frequency (TF-IDF)*. Hal ini menggunakan perhitungan frekuensi kata dalam setiap dokumen (TF) dan mengalinya dengan nilai inverse document frequency (IDF) untuk memberikan bobot yang lebih tinggi pada kata-kata yang jarang muncul secara umum di dalam dataset.
  - Luaran yang dihasilkan berupa representasi vektor TF-IDF untuk setiap buku dalam dataset.

- Cosine Similarity
  - Menggunakan representasi vektor TF-IDF yang telah dibuat sebelumnya.
  - Menggunakan fungsi *cosine similarity* untuk menghitung kesamaan antara dua vektor TF-IDF.
  - Semakin tinggi nilai *cosine similarity* antara dua buku, semakin mirip atau serupa kedua buku tersebut.

- Membangun Fungsi `content_based_filtering`
  - Membangun fungsi dengan memanfaatkan langkah-langkah sebelumnya, yaitu TF-IDF Vectorizer dan perhitungan *cosine similarity*.
  - Dalam fungsi ini, dilakukan pembobotan TF-IDF untuk setiap buku dalam dataset menggunakan TF-IDF Vectorizer.
  - Kemudian perhitungan *cosine similarity* dilakukan antara vektor buku yang ingin direkomendasikan dengan vektor representasi buku lainnya dalam dataset.
  - Hasil *cosine similarity* diurutkan dari nilai tertinggi ke terendah untuk mendapatkan buku-buku dengan kesamaan terbesar.
  - Fungsi ini akan menghasilkan output berupa daftar rekomendasi buku berdasarkan kesamaan cosine similarity, dengan jumlah rekomendasi yang diinginkan (di sini Penulis menggunakan k = 5 ).

### Hasil Top-5 Recommendation
Berikut merupakan salah satu contoh dari hasil rekomendasi yang diberikan oleh model dengan judul buku **"Chicken Soup for the Preteen Soul - 101 Stories of Changes, Choices and Growing Up for Kids, ages 10-13"**

Tabel 10. Hasil Top-5 Recommendation
| Recommended Books                                 | Similarity |
|---------------------------------------------------|------------|
| A 6th Bowl of Chicken Soup for the Soul (Chick... | 1.0        |
| Chicken Soup for the Mother's Soul 2 : 101 Mor... | 1.0        |
| Chicken Soup for the Soul                         | 1.0        |
| A Second Chicken Soup for the Woman's Soul (Ch... | 1.0        |
| Chicken Soup from the Soul of Hawaii: Stories ... | 1.0        |

Dalam Tabel 10. terlihat bahwa model dapat memberikan rekomendasi buku serupa dari judul buku *Chicken Soup for the Soul* dengan nilai kesamaan 1.0 atau sama dengan 100% serupa.

## Evaluation
Dalam proyek ini, Penulis menggunakan beberapa metrik evaluasi untuk kasus rekomendasi, yaitu presisi, recall, MAP@0.5 seperti pada Tabel 11. Untuk *ground truth* yang digunakan menggunakan filtering seperti snippet di bawah ini:
```
books[books['book_title'].str.contains("Chicken Soup")]['book_title'].tolist()
```
Tabel 11. Hasil Evaluasi

| Presisi | Recall              | MAP@5               |
|---------|---------------------|---------------------|
| 1.0     | 0.13157894736842105 | 0.13157894736842105 |

- Presisi
  - Presisi mengukur proporsi dari item yang relevan (buku yang benar-benar relevan dengan buku referensi/ *ground truth*) di antara item yang direkomendasikan.
  - Dalam proyek ini, presisi bernilai 1.0, yang berarti semua buku yang direkomendasikan untuk judul buku "Chicken Soup for the Preteen Soul - 101 Stories of Changes, Choices and Growing Up for Kids, ages 10-13" relevan secara tepat.
  - Presisi yang tinggi menunjukkan bahwa rekomendasi yang diberikan oleh sistem memiliki tingkat keakuratan yang tinggi, dengan sebagian besar rekomendasi yang relevan.

- Recall
  - Recall mengukur proporsi dari item yang relevan yang berhasil ditemukan oleh sistem rekomendasi di antara seluruh item yang relevan.
  - Dalam proyek ini, recall memiliki nilai 0.13, yang berarti sistem rekomendasi berhasil menemukan 13% dari buku-buku yang relevan dengan judul buku referensi.
  - Recall yang tinggi menunjukkan bahwa sistem rekomendasi mampu menemukan sebagian besar buku yang relevan, dengan jumlah *false negative* (buku yang relevan tetapi tidak direkomendasikan) yang relatif rendah.

- MAP@5 (Mean Average Precision at 5)
  - MAP@5 adalah rata-rata dari presisi pada 5 posisi teratas dalam daftar rekomendasi.
  - Dalam proyek ini, MAP@5 memiliki nilai 0.13, yang menunjukkan rata-rata presisi pada 5 buku teratas dalam daftar rekomendasi sebesar 13%.
  - MAP@5 yang tinggi menunjukkan bahwa rekomendasi yang lebih relevan cenderung berada pada peringkat atas dalam daftar rekomendasi.

## Conclution
Proyek ini berhasil mengimplementasikan sisstem rekomendasi buku menggunakan metode *content-based filtering*. Model dapat memberikan rekomendasi buku yang relevan berdasarkan judul buku, dengan tingkat presisi yang tinggi. Namun, terdapat beberapa keterbatasan dalam proyek ini, misalnya dataset yang digunakan hanya sebanyak 20.000 data. Proyek ini dapat dikembangkan misalnya dengan penambahan jumlah data ataupun menggunakan teknik filtering lainnya seperti *collaborative fitlering*.

---
**Daftar Pustaka**

[1] [S. Saefudin and D. Fernando, “PENERAPAN DATA MINING REKOMENDASI BUKU MENGGUNAKAN ALGORITMA APRIORI,” JSiI (Jurnal Sistem Informasi), vol. 7, no. 1, p. 50, Mar. 2020, doi: https://doi.org/10.30656/jsii.v7i1.1899‌.](https://core.ac.uk/download/pdf/327232562.pdf)

[2] [M. Fajriansyah, P. P. Adikara, and A. W. Widodo, "Sistem Rekomendasi Film Menggunakan Content Based Filtering," Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer, vol. 5, no. 6, pp. 2188-2199, 2021.](https://jptiik.multi.web.id/index.php/j-ptiik/article/download/9163/4159)
