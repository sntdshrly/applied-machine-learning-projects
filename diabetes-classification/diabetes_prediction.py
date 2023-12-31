# -*- coding: utf-8 -*-
"""diabetes-prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EI9FULXv6ZNK2XdJFucvqm-B2T5Ecylm

# Predictive Analytics - Diabetes Prediction 🍔

by: Sherly Santiadi

# Domain Proyek
---
Penderita diabetes beresiko lebih tinggi pada kasus pasien COVID-19. Oleh karena itu, pentingnya untuk mendeteksi sedini mungkin penyakit diabetes berdasarkan latar belakang historis sehingga dapat diberi perawatan yang tepat.

[Sumber Referensi](https://bapin-ismki.e-journal.id/jimki/article/view/342)

# Business Understanding
---

## Problem Statements
- Bagaimana membuat model untuk mengidentifikasi pasien yang beresiko diabetes?
- Bagaimana mengukur performa model untuk memprediksi pasien yang beresiko diabetes?

## Goals
- Membuat model untuk mengidentifikasi pasien yang beresiko diabetes.
- Mengukur performa model untuk memprediksi pasien yang beresiko diabetes.

## Solution Statements
- Membangun model `random forest` dengan menerapkan `hyperparameter tuning`
- Mengukur performa model dengan metrik evaluasi.

# Data Understanding
---
Data yang digunakan dalam proyek ini bersumber dari [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

Variabel-variabel pada Diabetes dataset adalah sebagai berikut:
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

# Library
---
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
path="/content/drive/My Drive/Colab Notebooks/diabetes_prediction_dataset.csv"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.tree import plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

"""# Exploratory Data Analysis (EDA)
---

## Gathering Data
"""

df = pd.read_csv(path)
df

"""Pada data yang digunakan dalam proyek ini terdapat `100.000 baris` dengan `9 kolom`.

## Assessing Data
- Apakah ada missing value?
"""

df.isnull().sum()

"""Pada data yang digunakan dalam proyek ini tidak terdapat missing value.

- Apakah ada data duplikat?
"""

df.duplicated().sum()

"""Pada data yang digunakan dalam proyek ini terdapat 3.854 data yang duplikat."""

df.info()

"""- Terdapat 2 kolom dengan tipe object, yaitu: `gender` dan `smoking_history`. Kolom ini merupakan categorical features (fitur non-numerik).
- Terdapat 3 kolom numerik dengan tipe data float64 yaitu: `age`,`bmi`,`HbA1c_level`. Ini merupakan fitur numerik yang merupakan hasil pengukuran bertipe desimal.
- Terdapat 4 kolom numerik dengan tipe data int64, yaitu: `hypertension`,`heart_disease`,`blood_glucose_level`,`diabetes`. Ini merupakan fitur numerik yang merupakan hasil pengukuran bertipe integer.

## Menangani Data Duplikat
Pada proyek ini data duplikat akan dihapus dari dataframe.
"""

df = df.drop_duplicates(keep='last')
df.duplicated().sum()

len(df)

"""Setelah menghapus data duplikat data berkurang dari yang awalnya terdapat 10.000 data → 96.146 data.

## Describing Data
"""

df.describe()

"""Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.
"""

df.nunique()

"""`df.nunique()` digunakan untuk mengetahui berapa banyak nilai unik di dalam setiap kolom."""

df.gender.value_counts()

"""Kolom `gender` terdapat 3 unik value yaitu `female`,`male`,`other`. Karena proporsi `other` jauh lebih sedikit dibandingkan `female` dan `male` maka akan dihapus."""

# inplace = True untuk mengimplementasikan drop langsung ke dataframe original
df.drop(df[df['gender']=='Other'].index , inplace=True)
df.gender.value_counts()

df.age.value_counts()

"""## Mengubah Tipe Data

Kolom `age` terdapat usia dengan tipe data desimal, oleh karena itu tipe data kolom `age` akan diubah menjadi integer.
"""

df['age'] = df['age'].astype(int)
df.age.value_counts()

df.smoking_history.value_counts()

corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()

"""Dari matriks korelasi ditemukan bahwa kolom yang memiliki korelasi positif dengan kolom `diabetes` diurutkan dari yang paling kuat hingga lemah sebagai berikut:
1. `blood_glucose_level`
2. `HbA1c_level`
3. `age`
4. `bmi`
5. `hypertension`
6. `heart_disease`

Pada kolom `smoking_history` terdapat value `No Info` yang cukup banyak yaitu sebesar 32.881.

## Mengubah Kolom Kategorikal

Karena pada kolom `smoking_history` terdapat pengkategorian yang mirip maka untuk kategori yang mirip akan dikelompokan ke dalam satu kategori yang sama sehingga data tersebut lebih terorganisir.
"""

def kategori_smoking_history(label):
    if label in ["not current", "former","ever"]:
        return 'Past'
    elif label in["never","No info", "None"]:
        return "Never"
    elif label =='current':
        return 'Current'
df["smoking_history"] = df["smoking_history"].apply(kategori_smoking_history)
df.smoking_history.value_counts()

"""## Melihat Distribusi Kolom"""

df["gender"].value_counts().plot(kind="bar", figsize=(10,6))
plt.title("Gender Distribution")
plt.ylabel("Count")
plt.xlabel("Name")

sns.histplot(df["age"],kde=True,bins=25)
plt.title("Age Distribution");

df["hypertension"].value_counts().plot(kind="bar", figsize=(10,6))
plt.title("Hypertension Distribution")
plt.ylabel("Count")
plt.xlabel("Name")

df["heart_disease"].value_counts().plot(kind="bar", figsize=(10,6))
plt.title("Heart Disease Distribution")
plt.ylabel("Count")
plt.xlabel("Name")

df["smoking_history"].value_counts().plot(kind="bar", figsize=(10,6))
plt.title("Smoking History Distribution")
plt.ylabel("Count")
plt.xlabel("Name")

sns.histplot(df["bmi"],kde=True,bins=25)
plt.title("Bmi Distribution");

sns.histplot(df["HbA1c_level"],kde=True,bins=25)
plt.title("HbA1c Level Distribution");

sns.histplot(df["blood_glucose_level"],kde=True,bins=25)
plt.title("Blood Glucose Level Distribution");

df["diabetes"].value_counts().plot(kind="bar", figsize=(10,6))
plt.title("Diabetes Distribution")
plt.ylabel("Count")
plt.xlabel("Name")

"""Dari grafik di atas ditemukan bahwa kolom `diabetes` sangat tidak seimbang mengingat kolom dengan nilai 0 berjumlah > 80.000 sedangkan kolom dengan nilai 1 berjumlah < 20.000"""

sns.boxplot(data= df, x=df["diabetes"], y=df["age"])

"""Dari grafik di atas dapat diketahui bahwa rentang umur seseorang mengalami diabetes yaitu diantara `50 - 80`."""

df.reset_index(drop=True, inplace=True)
df.describe()

"""Pada data yang digunakan dalam proyek ini terdapat outliers, namun outliers tersebut tidak akan dihapus dikarenakan akan mempengaruhi kolom-kolom biner lainnya seperti pada kolom `hypertension` menjadi hanya memiliki nilai 0 saja."""

# Drop baris dengan nilai 'age' = 0 dikarenakan usia 0 tidak relevan untuk digunakan
df = df.loc[(df[['age']]!=0).all(axis=1)]
df.describe()

"""# Data Preparation
---

Sebelum melakukan manipulasi data atau konversi data ke dalam data numerik, data sebaiknya dipisahkan terlebih dahulu untuk mencegah kebocoran data.
"""

df

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# split data menjadi training dan testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# split data training menjadi training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print("Jumlah X_train: ", X_train.shape)
print("Jumlah y_train: ", y_train.shape)
print("Jumlah X_val: ", X_val.shape)
print("Jumlah y_val: ", y_val.shape)
print("Jumlah X_test: ", X_test.shape)
print("Jumlah y_test: ", y_test.shape)

X_train.info()

"""## One Hot Encoding"""

obj = (X.dtypes == 'object')
object_column_X = list(obj[obj].index)
object_column_X

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_columns = encoder.fit_transform(X_train[object_column_X])
df_encoded = pd.DataFrame(encoded_columns)
df_encoded.index = X_train.index
X_train_numerical = X_train.drop(object_column_X, axis=1)
X_train_encoded = pd.concat([X_train_numerical, df_encoded], axis=1)
X_train_encoded

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_columns = encoder.fit_transform(X_val[object_column_X])
df_encoded = pd.DataFrame(encoded_columns)
df_encoded.index = X_val.index
X_val_numerical = X_val.drop(object_column_X, axis=1)
X_val_encoded = pd.concat([X_val_numerical, df_encoded], axis=1)
X_val_encoded

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_columns = encoder.fit_transform(X_test[object_column_X])
df_encoded = pd.DataFrame(encoded_columns)
df_encoded.index = X_test.index
X_test_numerical = X_test.drop(object_column_X, axis=1)
X_test_encoded = pd.concat([X_test_numerical, df_encoded], axis=1)
X_test_encoded

"""## Standarisasi"""

X_train_encoded.info()

numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()

scaler.fit(X_train_encoded[numerical_features])
X_train_encoded[numerical_features] = scaler.transform(X_train_encoded.loc[:, numerical_features])
X_train_encoded[numerical_features].describe().round(4)

scaler.fit(X_val[numerical_features])
X_val_encoded[numerical_features] = scaler.transform(X_val_encoded.loc[:, numerical_features])
X_val_encoded[numerical_features].describe().round(4)

scaler.fit(X_test[numerical_features])
X_test_encoded[numerical_features] = scaler.transform(X_test_encoded.loc[:, numerical_features])
X_test_encoded[numerical_features].describe().round(4)

"""Perhatikan tabel di atas, sekarang nilai mean = 0 dan standar deviasi = 1.

## Resampling
"""

X_train_encoded

X_train_encoded.columns = X_train_encoded.columns.astype(str)
X_val_encoded.columns = X_val_encoded.columns.astype(str)
X_test_encoded.columns = X_test_encoded.columns.astype(str)

# Kelas minoritas akan ditingkatkan 10% dari total dataset
smote = SMOTE(sampling_strategy = 0.1)
# Kelas majoritas akan ditingkatkan 50% dari total dataset
random_under_sampling = RandomUnderSampler(sampling_strategy=0.5)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)
X_train_resampled, y_train_resampled = random_under_sampling.fit_resample(X_train_resampled, y_train_resampled)

"""# Modeling
---

Pada tahapan pemodelan ini, kita menggunakan algoritma Random Forest Classifier untuk menyelesaikan permasalahan dalam proyek "Diabetes Prediction".

Tahapan yang dilakukan dalam pemodelan ini adalah sebagai berikut:
- Membentuk model awal: Pertama, kita membuat model awal dengan menggunakan RandomForestClassifier tanpa mengatur parameter apapun. Model awal ini digunakan untuk melatih data yang telah diresampling `(X_train_resampled)` dan `(y_train_resampled)`.
- Penentuan parameter: Setelah itu, kita menentukan parameter-parameter yang akan digunakan pada model Random Forest. Dalam contoh ini, kita menggunakan RandomizedSearchCV untuk mencari parameter terbaik dengan mencoba beberapa kombinasi secara acak. Parameter yang diuji adalah jumlah estimators `(n_estimators)`, kedalaman maksimum `(max_depth)`, minimum sampel split `(min_samples_split)`, dan minimum sampel leaf `(min_samples_leaf)`. Parameter-parameter ini digunakan untuk meningkatkan performa model dan menghindari overfitting.
- Pencarian parameter terbaik: Pada langkah ini, RandomizedSearchCV mencoba beberapa kombinasi parameter yang dijelaskan di atas dengan melakukan validasi silang `(cross-validation)` menggunakan 5-fold cross-validation `(cv=5)`. Dalam hal ini, RandomizedSearchCV mencoba 10 kombinasi parameter yang berbeda `(n_iter=10)` dan mencatat performa model untuk setiap kombinasi.
- Memilih model terbaik: Setelah melakukan pencarian parameter, kita memilih model terbaik berdasarkan performa yang diukur. Model terbaik ini akan digunakan untuk melatih data yang telah dilakukan one hot encoding `(X_train_encoded)` dan target `(y_train)`.
"""

random_forest = RandomForestClassifier()
random_forest.fit(X_train_resampled, y_train_resampled)

"""## Hyperparameter Tuning
Proses improvement melalui hyperparameter tuning:

Dalam langkah hyperparameter tuning, RandomizedSearchCV digunakan untuk mencari kombinasi terbaik dari parameter-parameter yang diuji. Dalam contoh ini, kita mencari nilai terbaik untuk n_estimators, max_depth, min_samples_split, dan min_samples_leaf.

Dengan melakukan hyperparameter tuning, kita dapat meningkatkan performa model dengan menemukan parameter yang optimal. Parameter yang optimal dapat membantu menghindari overfitting, meningkatkan akurasi, dan menghasilkan model yang lebih baik secara keseluruhan.

Setelah proses hyperparameter tuning, kita mendapatkan model terbaik `(rf_best)` yang kemudian dilatih menggunakan data yang telah dilakukan one hot encoding `(X_train_encoded)` dan target `(y_train)`. Model terbaik ini kemudian dapat digunakan untuk melakukan prediksi pada data yang baru.
"""

param_distributions = {'n_estimators': [10, 50, 100, 200],
                       'max_depth': [None, 5, 10, 20],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4]}

random_search = RandomizedSearchCV(random_forest, param_distributions, n_iter=10, cv=5, error_score='raise')
random_search.fit(X_train_resampled, y_train_resampled)

rf_best = random_search.best_estimator_
rf_best.fit(X_train_encoded, y_train)

"""# Evaluation
---
Dalam proyek ini, kita menggunakan beberapa metrik evaluasi untuk kasus klasifikasi, yaitu akurasi, presisi, recall, dan skor F1. Berikut adalah penjelasan singkat mengenai metrik-metrik tersebut:

- Akurasi: Metrik ini mengukur sejauh mana model dapat mengklasifikasikan data dengan benar secara keseluruhan. Akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total jumlah prediksi.

- Presisi: Presisi mengukur sejauh mana model memberikan hasil positif yang benar dari semua prediksi positif yang dilakukan. Presisi dihitung dengan membagi jumlah prediksi positif yang benar dengan jumlah total prediksi positif.

- Recall: Recall, juga dikenal sebagai sensitivitas, mengukur sejauh mana model dapat mengidentifikasi secara benar semua instance positif yang ada dalam dataset. Recall dihitung dengan membagi jumlah prediksi positif yang benar dengan jumlah total instance positif yang sebenarnya.

- Skor F1: Skor F1 adalah rata-rata harmonik antara presisi dan recall. Skor F1 berguna untuk memperhitungkan baik presisi maupun recall dalam satu metrik yang konsisten. Skor F1 dihitung dengan menggunakan formula: 2 * (presisi * recall) / (presisi + recall).
"""

y_pred = rf_best.predict(X_val_encoded)
print(classification_report(y_val, y_pred))

y_test_pred = rf_best.predict(X_test_encoded)
print(classification_report(y_test, y_test_pred))

"""Berdasarkan hasil proyek dan metrik evaluasi yang digunakan, kita dapat melihat bahwa model yang telah dilatih memberikan hasil yang baik. Berikut adalah analisis berdasarkan metrik-metrik evaluasi yang digunakan:

- Akurasi: Model memiliki akurasi sebesar 97% pada data validasi (15235 sampel) dan 97% pada data pengujian (19044 sampel). Ini berarti model dapat mengklasifikasikan data dengan benar dengan tingkat akurasi yang tinggi.

- Presisi: Model memiliki presisi sebesar 97% untuk kelas 0 (tidak diabetes) dan 100% untuk kelas 1 (diabetes) pada data validasi dan data pengujian. Hal ini menunjukkan bahwa model memberikan sedikit sekali kesalahan dalam mengklasifikasikan data sebagai kelas 0 atau kelas 1.

- Recall: Model memiliki recall sebesar 100% untuk kelas 0 dan 68% untuk kelas 1 pada data validasi dan data pengujian. Ini berarti model mampu mengidentifikasi secara benar semua instance yang termasuk dalam kelas 0, tetapi memiliki tingkat recall yang lebih rendah untuk kelas 1.

- F1 Score: F1 Score untuk kelas 0 adalah 98% pada data validasi dan data pengujian, sementara untuk kelas 1 adalah 81% pada data validasi dan data pengujian. F1 Score yang tinggi menunjukkan keseimbangan antara presisi dan recall.

Secara keseluruhan, model `Random Forest Classifier` yang telah dilatih memberikan hasil yang baik dengan tingkat akurasi yang tinggi dan presisi yang baik untuk kedua kelas.

Namun, terdapat penurunan recall dan F1 Score untuk kelas 1, yang dapat menunjukkan bahwa model memiliki kesulitan dalam mengidentifikasi instance-instance yang termasuk dalam kelas 1.

---

Dalam kasus prediksi diabetes, perlu mempertimbangkan apakah fokus yang lebih penting:
- Mengidentifikasi sebanyak mungkin pasien yang benar-benar menderita diabetes (recall tinggi).
- Meminimalkan jumlah kesalahan dalam mengklasifikasikan pasien yang sehat sebagai pasien diabetes (presisi tinggi).

Jika kita lebih peduli untuk mengidentifikasi sebanyak mungkin pasien yang benar-benar menderita diabetes, maka recall yang tinggi lebih penting. Recall yang tinggi berarti model memiliki kemampuan yang baik dalam menemukan pasien diabetes yang sebenarnya dan mengklasifikasikannya dengan benar. Dalam hal ini, kita berusaha untuk menghindari kesalahan mengklasifikasikan pasien diabetes sebagai sehat.

Namun, jika fokus lebih pada menghindari kesalahan mengklasifikasikan pasien sehat sebagai pasien diabetes, maka presisi yang tinggi lebih penting. Presisi yang tinggi berarti model memiliki kemampuan yang baik dalam memastikan bahwa pasien yang diklasifikasikan sebagai diabetes adalah pasien yang benar-benar menderita diabetes. Dalam hal ini, kita berusaha untuk menghindari kesalahan mengklasifikasikan pasien sehat sebagai diabetes.

## ROC Curve
ROC Curve digunakan untuk mengilustasikan trade-off antara true positive rate (TPR) dan false positive rate (FPR) pada berbagai threshold klasifikasi.
"""

# Plot decision tree pertama di random forest
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(rf_best.estimators_[0], max_depth=2, ax=ax)

# Untuk memvisualisasikan probabilitas kelas di setiap node, gunakan metode `predict_proba` di seluruh random forest
# Misalnya untuk memprediksi probabilitas kelas untuk contoh pertama dalam test set:
probs = rf_best.predict_proba(X_test_encoded[:1])
print(probs)

X_test[:1]

# Memprediksi probabilitas kelas untuk kelas positif (kelas 1)
y_pred_proba = rf_best.predict_proba(X_test_encoded)[:, 1]

# Hitung kurva ROC dan AUC untuk kelas positif
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Class 1 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()

# Hitung confusion matrix
y_pred = rf_best.predict(X_test_encoded)
conf_matrix = confusion_matrix(y_test, y_pred)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

"""# Conclusion
---
- Membuat model random forest dengan menerapkan hyperparameter tuning yaitu menggunakan randomized search dapat mengidentifikasi pasien yang beresiko diabetes.
- Hasil `F1-Score` model random forest pada test set sebesar 97%.
"""