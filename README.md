
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

* Untuk pembuatan model menggunakan model K-Nearest Neighbor.Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. Berikut cara kerja, 
untuk Hitungan euclidin distance dengan rumus
[![rumus1.png](https://i.postimg.cc/44S3ZzWc/rumus1.png)](https://postimg.cc/mh74yFfZ)
kelebihan dan kekurangan algoritma Random Forest dan K-Nearest Neighbor:
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


### Variable-variabel pada Water quality

Pada dataset yang digunakan " Analisis Prediktif Data Air Bersih Di Masyarakat" berisi 3277 baris dan 12 kolom.tipe data yang digunakan adalah float64 dan 1 buah data bertipe int64. Untuk penjelasan menganai variabel - variabel yang ada dalam dataset **_water quality_** sebagai berikut :

1. **Nilai pH**: adalah aparameter penting dalam mengevaluasi keseimbangan asam-basa air.merupakan nini indikator apakah air asam/basa dengan menggunakan skala 0 hingga 14.
2. **Hardness**: adalah parameter yang disebabkan oleh garam kalsium dan magnesium,kekerasan dalam kapasitas air untuk mengendapkan sabun dalam mg/L
3. **Solids**: adalah kemampuan air untuk melarutkan mineral/garam anorganik dan beberapa mineral organik seperti kalium,kalsium,dan lain-lainya.
4. **Chloramines**: adalah satu jenis disinfektan yang sering digunakan dalam sistem penyediaan air minum. Biasanya, kloramin terbentuk ketika amonia dicampurkan dengan klorin dalam proses pengolahan air minum. Konsentrasi klorin hingga 4 miligram per liter (mg/L) dianggap aman dalam air minum.
5. **Sulfate** : adalah senyawa alami yang ditemukan dalam mineral, tanah, dan batuan. Senyawa ini juga hadir dalam udara di sekitar kita, air tanah, tumbuhan, dan makanan. Penggunaan utama sulfat terdapat dalam industri kimia. Konsentrasi sulfat di air laut sekitar 2.700 miligram per liter (mg/L).
6. **Conductivity**: adalah jumlah zat terlarut dalam air menentukan konduktivitas listriknya. Konduktivitas listrik (EC) sebenarnya mengukur proses ionik suatu larutan yang memungkinkannya menghantarkan arus. Menurut standar WHO, nilai konduktivitas listrik (EC) tidak boleh melebihi 400 μS/cm.
7. **Organic_carbon**: adalah  ukuran jumlah total karbon dalam senyawa organik dalam air murni. Menurut US EPA, nilai TOC dalam air yang telah diolah/minum seharusnya <2 mg/L, dan <4 mg/L dalam sumber air yang digunakan untuk pengolahan.
8. **Trihalomethanes**:adalah bahan kimia yang dapat ditemukan dalam air yang diolah dengan klorin. Konsentrasi THM dalam air minum bervariasi tergantung pada tingkat bahan organik dalam air, jumlah klorin yang diperlukan untuk mengolah air, dan suhu air yang sedang diolah.
9. **Turbidity**: adalah Kekeruhan yang berasal sifat air dalam memancarkan cahaya, dan uji ini digunakan untuk menunjukkan kualitas pembuangan limbah terkait dengan zat koloid. Nilai kekeruhan rata-rata yang diperoleh untuk Kampus Wondo Genet (0,98 NTU) lebih rendah dari nilai yang direkomendasikan oleh WHO yaitu 5,00 NTU.
10. **Potability**: adalah air aman untuk dikonsumsi oleh manusia, di mana 1 berarti dapat diminum dan 0 berarti tidak dapat diminum.

Selain itu disini melkukan visualisasi data yang kosng
[![visualisasi-data.png](https://i.postimg.cc/0jYX95F5/visualisasi-data.png)](https://postimg.cc/xNd3RQ9B)
[![visualisasi-data-2.png](https://i.postimg.cc/zXJY34j2/visualisasi-data-2.png)](https://postimg.cc/Z0DMM7X6)

## Data Preparation
Pada proyek ini teknik data preparation yang dilakukan diantaranya :

Untuk mengatasi data kosong dengan nilai rata-rata kolom (mean substition),maka rata-rata yang memiliki data kosong dengan membuat kolom Potability = 0
[![prepation-1.png](https://i.postimg.cc/8CdRNh7w/prepation-1.png)](https://postimg.cc/CZ1B4nHf)

Ini berdasarkan rata-rata data kosong, dengan kolom Portability = 1

[![prepation-1.png](https://i.postimg.cc/8CdRNh7w/prepation-1.png)](https://postimg.cc/CZ1B4nHf)

Ini berdasarkan data pada kolom yang memiliki data kosong (data keseluruhan)
[![prepation-3.png](https://i.postimg.cc/GhkGkFQR/prepation-3.png)](https://postimg.cc/F7HYvJ8C)











