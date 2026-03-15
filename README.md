# BSI Mobile App Sentiment Analysis

Sebuah proyek *Natural Language Processing* (NLP) untuk menganalisis sentimen ulasan pengguna aplikasi BSI Mobile menggunakan teknik *Machine Learning* dan *Word Embeddings*.

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk mengekstraksi dan mengklasifikasikan sentimen pengguna  (Positif/Negatif/Netral) dari ulasan BSI Mobile. Eksperimen dilakukan dengan membandingkan tiga algoritma klasifikasi (ANN, LightGBM, SVC) terhadap tiga representasi *Word Embeddings* yang berbeda untuk mencari model dengan pemahaman semantik dan performa klasifikasi terbaik, yaitu:
- **Word2Vec**
- **FastText**
- **GloVe**

Proyek ini merupakan *submission* untuk kelas **Belajar Fundamental Deep Learning** pada program **Coding Camp 2026 dari Dicoding x DBS Foundation (Learning Path: AI Engineer)**. Repositori ini disusun dengan mengedepankan prinsip reproduktibilitas dan modularitas kode sesuai standar industri.

## 🛠️ Tech Stack & Libraries
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-20232A.svg?style=for-the-badge)
![Gensim](https://img.shields.io/badge/Gensim-F19122.svg?style=for-the-badge)
![NLTK](https://img.shields.io/badge/NLTK-315A8A.svg?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

- **Bahasa:** Python
- **Libraries NLP & ML:** `nltk`, `gensim` (Word2Vec, FastText), `scikit-learn`, `lightgbm`
- **Libraries Deep Learning:** `tensorflow`, `torch`
- **Data Manipulation:** `pandas`, `numpy`

## 🗂️ Struktur Direktori

Project ini mengikuti standar [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

    ├── data/                  
    │   ├── external/          <- Kamus eksternal (root words, stop words)
    │   ├── interim/           <- Data yang telah dibersihkan (clean_review_bsi_mobile.xlsx)
    │   ├── processed/         <- Data final berlabel (labelled_review_bsi_mobile.xlsx)
    │   └── raw/               <- Data ulasan asli hasil scraping (review_bsi_mobile.xlsx)
    │
    ├── models/                <- Model yang telah dilatih (FastText SVC, Stemmer Cache, Stopwords)
    │                             Note: File model (.joblib, .pkl) tidak diunggah ke repositori ini.
    │                             [Unduh Model di Sini](https://drive.google.com/drive/folders/1M3wcLBIN8-Skw3jSLVutaiusYz94kt3T?usp=sharing)
    │
    ├── notebooks/             <- Jupyter notebooks untuk eksplorasi dan pemodelan
    │   ├── 01_scraping.ipynb
    │   ├── 02_data_labelling.ipynb
    │   ├── 03_pre_processing.ipynb
    │   ├── 04_word2vec.ipynb
    │   ├── 05_fasttext.ipynb
    │   ├── 06_glove.ipynb
    │   └── 07_inference.ipynb
    │
    ├── src/                   <- Source code untuk digunakan dalam proyek ini
    │   ├── __init__.py           
    │   ├── utils.py           <- Fungsi bantuan (helper functions)
    │   └── slang_words.py     <- Pemetaan dan penanganan kata gaul/slang
    │
    ├── .gitignore             <- Mengabaikan file sistem, cache, dan model berukuran besar
    ├── requirements.txt       <- Daftar dependensi untuk mereproduksi environment analisis
    └── README.md              <- Dokumentasi utama proyek

## 🚀 Cara Menjalankan Proyek (Reproduction)

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/11erlangga/sentiment-analysis-bsi-mobile.git
   cd sentiment-analysis-bsi-mobile
    ```

2. **Buat Virtual Environment (Opsional tapi direkomendasikan):**
    ```Bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    venv\Scripts\activate     # Untuk Windows
    ```

3. **Install dependensi:**
    ```Bash
    pip install -r requirements.txt
    ```

4. **Unduh Model (Wajib untuk Inference):**
    Unduh file model `.joblib` dan `.pkl` dari [Tautan Google Drive](https://drive.google.com/drive/folders/1M3wcLBIN8-Skw3jSLVutaiusYz94kt3T?usp=sharing) dan letakkan di dalam folder `models/` sebelum menjalankan notebook `07_inference.ipynb`.

## 📊 Hasil Eksperimen
Eksperimen dilakukan dengan membandingkan performa model Artificial Neural Network (ANN), LightGBM (LGBM), dan Support Vector Classifier (SVC) pada masing-masing *word embedding*. Karena indikasi adanya ketidakseimbangan kelas (*imbalanced data*) pada ulasan, **F1-Macro** dijadikan metrik evaluasi utama.

Berikut adalah performa model terbaik (berdasarkan F1-Macro) untuk masing-masing *embedding*:

| Word Embedding | Best Model | Accuracy | Weighted Precision | Weighted Recall | F1-Macro |
|----------------|-----------|----------|-------------------|----------------|----------|
| Word2Vec       | SVC       | 0.9190   | 0.9100            | 0.9190         | 0.7057   |
| FastText       | SVC       | 0.9212   | 0.9159            | 0.9212         | 0.7312   |
| GloVe          | SVC       | 0.9162   | 0.9117            | 0.9162         | 0.7427   |



Kesimpulan: Secara keseluruhan, model Support Vector Classifier (SVC) konsisten menghasilkan keseimbangan presisi dan recall (F1-Score) terbaik di semua skenario representasi kata dibandingkan ANN dan LGBM. Kombinasi GloVe + SVC ditetapkan sebagai model terbaik dalam proyek ini karena berhasil meraih F1-Score tertinggi sebesar 0.7427, menjadikannya model yang paling optimal untuk menangani klasifikasi pada dataset ulasan ini.

👨‍💻 Penulis
Erlangga Sri Heryanto - [LinkedIn Profile](https://www.linkedin.com/in/erlangga-sri-heryanto/)

*Proyek ini diselesaikan sebagai bagian dari Dicoding x DBS Foundation Coding Camp 2026.*