# XAI Post-hoc Classification for Diabetes Prediction

## Deskripsi Proyek

Proyek ini mengimplementasikan sistem **klasifikasi diabetes** berbasis *machine learning* yang dilengkapi dengan metode **Explainable Artificial Intelligence (XAI) post-hoc**. Sistem tidak hanya menghasilkan prediksi kelas (diabetes atau non-diabetes), tetapi juga menyediakan **penjelasan interpretatif** mengenai faktor-faktor (fitur klinis) yang memengaruhi keputusan model.

Pendekatan XAI digunakan untuk meningkatkan **transparansi, interpretabilitas, dan kepercayaan** terhadap sistem AI, khususnya pada domain kesehatan yang bersifat kritis.

---

## Dataset

Dataset yang digunakan adalah **Pima Indians Diabetes Dataset** (diakses melalui OpenML), yang berisi data klinis pasien dengan atribut:

* **preg** (*Number of Pregnancies*): jumlah kehamilan yang pernah dialami oleh pasien.
* **plas** (*Plasma Glucose Concentration*): kadar glukosa plasma dalam darah, yang merupakan indikator utama risiko diabetes.
* **pres** (*Diastolic Blood Pressure*): tekanan darah diastolik pasien (mm Hg).
* **skin** (*Triceps Skin Fold Thickness*): ketebalan lipatan kulit pada bagian triceps, yang digunakan sebagai estimasi kandungan lemak tubuh.
* **insu** (*Serum Insulin*): kadar insulin dalam serum darah pasien.
* **mass** (*Body Mass Index / BMI*): indeks massa tubuh yang menggambarkan proporsi berat badan terhadap tinggi badan.
* **pedi** (*Diabetes Pedigree Function*): ukuran risiko diabetes berdasarkan riwayat genetik keluarga.
* **age** (*Age*): usia pasien dalam satuan tahun.

Tugas yang dilakukan adalah **klasifikasi biner**, yaitu:

* `0` : Non-Diabetes
* `1` : Diabetes

---

## Teknik Machine Learning

Model klasifikasi yang digunakan adalah **Multilayer Perceptron (MLP)** yang dibangun menggunakan **PyTorch**, dengan karakteristik:

* Fully connected neural network
* Fungsi aktivasi ReLU
* Binary classification menggunakan `BCEWithLogitsLoss`
* Penanganan *class imbalance* dengan `pos_weight`
* Optimasi menggunakan Adam Optimizer

Evaluasi model dilakukan menggunakan metrik:

* Accuracy
* F1-score
* Confusion Matrix (dalam bentuk heatmap)

---

## Metode Explainable AI (XAI)

Tiga metode XAI post-hoc diterapkan dan dibandingkan:

1. **LIME-style (Local Surrogate Model)**
   Menggunakan regresi linear lokal untuk mengaproksimasi perilaku model di sekitar satu data masukan.

2. **Integrated Gradients (IG)**
   Mengukur kontribusi fitur berdasarkan integrasi gradien dari baseline ke input aktual.

3. **SHAP (SHapley Additive exPlanations)**
   Menghitung kontribusi fitur berdasarkan teori nilai Shapley untuk memberikan penjelasan yang konsisten secara teori.

Ketiga metode digunakan untuk menjelaskan prediksi pada beberapa sampel data uji dan dibandingkan hasilnya.

---

## Struktur Folder

Struktur proyek disusun secara modular sebagai berikut:

```
.
├── figures/              # Visualisasi (XAI plots, confusion matrix, dll)
├── models/               # Model terlatih
├── outputs/              # Output CSV (ringkasan XAI)
├── train/                # Modul utama pipeline
│   ├── data.py           # Load & preprocessing dataset
│   ├── helper.py         # Helper visualisasi
│   ├── metrics.py        # Evaluasi model
│   ├── mlp.py            # Arsitektur MLP
│   ├── sample.py         # Analisis XAI per sampel
│   └── xai.py            # Implementasi metode XAI
├── venv/                 # Virtual environment
├── main.py               # Entry point pipeline
├── requirements.txt      # Dependency
└── README.md             # Dokumentasi
```

---

## Cara Menjalankan Program

### 1️. Membuat Virtual Environment

```bash
python -m venv venv
```

### 2️. Mengaktifkan Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3️. Install Dependency

```bash
pip install -r requirements.txt
```

### 4️. Menjalankan Pipeline
Variabel n_samples dapat disesuaikan keinginan

```bash
python main.py --n_samples 10
```

Pipeline akan secara otomatis:

* Melatih model
* Mengevaluasi performa
* Menentukan threshold terbaik
* Menghasilkan penjelasan XAI
* Menyimpan tabel dan visualisasi

---

## Output yang Dihasilkan

### 📁 Folder `outputs/`

* `xai_global_summary.csv`
  Tabel ringkasan perbandingan hasil XAI untuk setiap sampel.

### 📁 Folder `figures/`

* `confusion_matrix.png`
* `lime_<i>.png`
* `ig_<i>.png`
* `shap_<i>.png`

Setiap gambar menjelaskan kontribusi fitur terhadap prediksi model.
Visualisasi bar yang mengarah ke kanan (bernilai positif) menunjukkan peningkatan risiko diabetes, sedangkan bar ke kiri (bernilai negatif) menunjukkan penurunan risiko diabetes, dengan panjang bar merepresentasikan besar kecilnya pengaruh fitur terhadap prediksi model.

---

## Kesimpulan Singkat

Hasil eksperimen menunjukkan bahwa meskipun ketiga metode XAI sering mengidentifikasi fitur yang sama sebagai faktor dominan, terdapat perbedaan perspektif penjelasan antar metode. Hal ini menegaskan bahwa interpretasi model sangat bergantung pada pendekatan XAI yang digunakan, sehingga penggunaan lebih dari satu metode XAI memberikan pemahaman yang lebih komprehensif terhadap perilaku model.

---

## Author 

> 13522104 Diana Tri Handayani
