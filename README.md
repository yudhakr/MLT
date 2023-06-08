
# Laporan Proyek Machine Learning - Ayudha Kusuma Rahmadhani

## Domain Proyek
Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai kesehatan dengan judul proyek "Analisis Prediktif Data Air Bersih Di Masyarakat".
[![Analisis-Prediktif-Data-Air-Bersih-Di-Masyarakat.jpg](https://i.postimg.cc/pTbscM81/Analisis-Prediktif-Data-Air-Bersih-Di-Masyarakat.jpg)](https://postimg.cc/DW6QmDxq)

**Latar Belakang**

Kualitas air yang baik adalah faktor penting bagi kesehatan manusia.
Manusia sangat bergantung pada air yang aman dan bersih untuk memenuhi kebutuhan sehari-hari, seperti minum, memasak, dan menjaga kebersihan pribadi.
Air yang terkontaminasi oleh polusi, seperti logam berat, pestisida, bakteri, dan virus, dapat menyebabkan berbagai masalah kesehatan yang serius, termasuk penyakit perut, masalah pernapasan, keracunan, dan gangguan sistem kekebalan tubuh. 

Masalah kualitas air yang umum terjadi termasuk kontaminasi feses, penyakit yang disebabkan oleh air yang tidak aman, dan tantangan akibat perubahan iklim dan faktor lainya yang menurunnya kualitas air
terutama untuk dikomsumsi.Kualitas fisikokimia, bakteriologis, dan konsentrasi logam jejak sampel air dari sumber yang diolah, keran jalan, dan wadah penyimpanan rumah tangga sebagian besar berada dalam kisaran yang diizinkan standar air minum WHO dan SANS.
HQ untuk anak-anak dan orang dewasa kurang dari satu, menunjukkan bahwa air minum menimbulkan ancaman kesehatan yang kurang signifikan bagi anak-anak dan orang dewasa. 

Format Referensi: [Judul Referensi](https://www.nature.com/articles/s41598-022-10092-4)

## Business Understanding
---
#### Problem Statements
Berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini diantaranya:
* Bagaimana cara membuat menprediksi potabilitas air berdasarkan atribut-atribut yang ada ?
* Bagaimana cara membuat model untuk memprediksi kualitas air bersih dimasyarakat
* Berapa mengetahui air tersebut layak atau tidak dikomsumsi ?


#### Goals
* Untuk mengetahui kualitas air dan potabilitasnya dengan  pembuatan modelnya
* Mengetahui cara pembuatan model machine learning untuk Analisis Prediktif Data Air Bersih Di Masyarakat.
* Membuat model _machine learning_ untuk mengklasifikasi data air yang layak dikonsumsi dengan nilai akurasi mencapai 90%

#### Solution Statements
Solusi yang dapat dilakukan untu memenuhi tujuan diantaranya :
- Untuk  melakukan pemerosesan data dilakukan beberapa teknik yaitu :
  - Mengisi data yang kosong dengan nilai rata - rata **_(mean substition)_**.
  - Mengatasi data yang tidak seimbang dengan **_(resample)_**.
  - Melakukan **_pembagian dataset_** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
  - Melakukan penghapusan data pencilan pada data latih dengan metode LOF **_(Local Outlier Factor)_**.
  - Melakukan standardisasi data pada semua fitur data **_(Standar Scaler)_**.

* Untuk pembuatan model menggunakan model K-Nearest Neighbor.Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. Berikut cara kerja, kelebihan dan kekurangan algoritma Random Forest dan K-Nearest Neighbor:
    * Cara kerja Algoritma K-Nearest Neighbor [[1]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/):
        * Menentukan jumlah tetangga terdekat K
        * Menghitung jarak dokumen _testing_ ke dokumen _training_
        * Urutkan data berdasarkan data yang mempunyai jarak Euclidean terkecil
        * Tentukan kelompok testing berdasarkan label pada K.
    * Kelebihan dan kekurangan Algoritma K-Nearest Neighbor [[2]](https://simdos.unud.ac.id/uploads/file_penelitian_1_dir/721bdb509a6f0bb9ccca6d7374b86759.pdf):
        * KNN memiliki beberapa kelebihan yaitu bahwa algoritmanya tangguh terhadap _training_ data yang _noisy_ dan efektif apabila data latihnya besar.
        * Kekurangan pada algoritma KKN yaitu perlu menentukan nilai dari parameter K (jumlah dari tetangga terdekat), Pembelajaran berdasarkan jarak tidak jelas mengenai jenis jarak apa yang harus digunakan dan atribut mana yang harus digunakan untuk mendapatkan hasil yang terbaik dan Biaya komputasi cukup tinggi karena diperlukan perhitungan dari jarak tiap sample uji pada keseluruhan sample latih.

## Data Understanding
[![dataset.png](https://i.postimg.cc/2689THMZ/dataset.png)](https://postimg.cc/gwQK2KYz)

Untuk informasi mengenai dataset  yang digunakan dalam proyak menggunakan data untuk mengembangakn proyek ini berasal dari
|Sumber                | [Kaggle Dataset : Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability) |
|-----------------------|----------------------------------------------------------------------------------|
| Lisensi               | Public Domain                                                                    |
| Kategori              | Kesehatan, Lingkungan, Kesehatan Masyarakat                                      |
| Jenis dan Ukuran Berkas| CSV (257 kB)                                                                    |


Pada dataset yang digunakan " Analisis Prediktif Data Air Bersih Di Masyarakat" berisi 3277 baris dan 12 kolom.tipe data yang digunakan adalah float64 dan 1 buah data bertipe int64. Untuk penjelasan menganai variabel - variabel yang ada dalam dataset _ water quality_ sebagai berikut :










