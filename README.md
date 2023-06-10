
# Laporan Proyek Machine Learning - Ayudha Kusuma Rahmadhani
Judul proyek : "Analisis Prediktif Data Air Bersih Di Masyarakat".

## Domain Proyek
[![Analisis-Prediktif-Data-Air-Bersih-Di-Masyarakat.jpg](https://i.postimg.cc/pTbscM81/Analisis-Prediktif-Data-Air-Bersih-Di-Masyarakat.jpg)](https://postimg.cc/DW6QmDxq)
   Kualitas air yang baik adalah faktor penting bagi kesehatan manusia.
Manusia sangat bergantung pada air yang aman dan bersih untuk memenuhi kebutuhan sehari-hari, seperti minum, memasak, dan menjaga kebersihan pribadi.
Air yang terkontaminasi oleh polusi, seperti logam berat, pestisida, bakteri, dan virus, dapat menyebabkan berbagai masalah kesehatan yang serius, termasuk penyakit perut, masalah pernapasan, keracunan, dan gangguan sistem kekebalan tubuh. 

   Berdasarkan terori masalah kualitas air yang umum terjadi termasuk kontaminasi feses, penyakit yang disebabkan oleh air yang tidak aman, dan tantangan akibat perubahan iklim dan faktor lainya yang menurunnya kualitas air
terutama untuk dikomsumsi.Kualitas fisikokimia, bakteriologis, dan konsentrasi logam jejak sampel air dari sumber yang diolah, keran jalan, dan wadah penyimpanan rumah tangga sebagian besar berada dalam kisaran yang diizinkan standar air minum WHO dan SANS.

Maka dari permasalahan tersebut penulis ingin membuat sistem model nalisis Prediktif Data Air Bersih Di Masyarakat berdasarkan variabel-variabel tertentu,sehingga masyarakat akan tahu dan berhati-hati dalam penggunaan air dan tahu cara menglola air yang terkontaminasi material tertentu.[[1]](https://www.mendeley.com/search/?page=1&query=%20K-Nearest%20Neighbor%20air%20bersih&sortBy=relevance)


## Business Understanding
---
#### Problem Statements
Berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini diantaranya:
* Bagaimana cara melakukan pra-pemrosesan data air agar dapat digunakan untuk membuat model yang baik ?
* Bagaimana cara membuat model machine learning untuk mengklasifikasikan data air yang dapat dikonsumsi oleh manusia ?

#### Goals
* Melakukan preprocessing data sehingga data tersebut siap untuk di latih oleh model Machine Learning
* Membuat model _machine learning_ untuk mengklasifikasi data air yang layak dikonsumsi dengan nilai akurasi mencapai 90%

#### Solution Statements
Solusi yang dapat dilakukan untu memenuhi tujuan diantaranya :
- Untuk  melakukan pemerosesan data dilakukan beberapa teknik yaitu :
  - Mengisi data yang kosong dengan nilai rata - rata **_(mean substition)_**.
  - Mengatasi data yang tidak seimbang dengan **_(resample)_**.
  - Melakukan **_pembagian dataset_** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
  - Melakukan penghapusan data pencilan pada data latih dengan metode LOF **_(Local Outlier Factor)_**.
  - Melakukan standardisasi data pada semua fitur data **_(Standar Scaler)_**.

* Untuk pembuatan model menggunakan model K-Nearest Neighbor sebagai model baseline.Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini.Algoritma ini mngasumsikan bahwa sesuatu yang serupa serta selalu berdekatan Berikut cara kerja algoritmanya, 
- Melakukan pemuatan data
- Menginisialisasikan nilai K(banyak tetangga/kelompok)
- Melakukan penambahkan jarak dan urutan dari contoh pada koleksi yang berururutan (hitungan euclidin distance) dengan rumus

$$d(xi, x1) = √(x₁ − xu)² + (xi2 − X12)² + ... + (Tip − Xlp) $$
- Memilih entri K paling awal pada koleksi yang beurutan
- Dapatkan label dari dari entri K yang dipilih
- Apabila kasus regresi, kembalikan nilai rata-ratanya. Apabila kasus klasifikasi, kembalikan labelnya

kelebihan dan kekurangan algoritma Random Forest dan K-Nearest Neighbor:
    * Cara kerja Algoritma K-Nearest Neighbor 
        * Menentukan jumlah tetangga terdekat K
        * Menghitung jarak dokumen _testing_ ke dokumen _training_
        * Urutkan data berdasarkan data yang mempunyai jarak Euclidean terkecil
        * Tentukan kelompok testing berdasarkan label pada K.
    * Kelebihan dan kekurangan Algoritma K-Nearest Neighbor 
        * KNN memiliki beberapa kelebihan yaitu bahwa algoritmanya tangguh terhadap _training_ data yang _noisy_ dan efektif apabila data latihnya besar.
        * Kekurangan pada algoritma KKN yaitu perlu menentukan nilai dari parameter K (jumlah dari tetangga terdekat), Pembelajaran berdasarkan jarak tidak jelas mengenai jenis jarak apa yang harus digunakan dan atribut mana yang harus digunakan untuk mendapatkan hasil yang terbaik dan Biaya komputasi cukup tinggi karena diperlukan perhitungan dari jarak tiap sample uji pada keseluruhan sample latih.

## Data Understanding

[![dataset.png](https://i.postimg.cc/FRvZwDbn/dataset.png)](https://postimg.cc/t1rWxdMh)
Untuk informasi mengenai dataset  yang digunakan dalam proyak menggunakan data untuk mengembangakn proyek ini berasal dari
|Sumber               |[Kaggle Dataset : Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability) |
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

Selain itu disini melakukan visualisasi data yang kosng
<img width="850" alt="visualisasi data" src="https://github.com/yudhakr/MLT/assets/84507343/8fbf937d-06ed-497b-b343-8c38dc5bb9e5">
<img width="734" alt="visualisasi data 2" src="https://github.com/yudhakr/MLT/assets/84507343/52cb286c-9712-4fa1-812d-14d9e6b469a3">


## Data Preparation
Pada proyek ini teknik data preparation yang dilakukan diantaranya :
<img width="113" alt="data kosong" src="https://github.com/yudhakr/MLT/assets/84507343/6d177780-0d43-4820-8969-1db359fbaa9a">
* Karena data tidak memiliki column categorical/object jadi skip langkah
* Karena data yang kosong pada dataset cukup banyak, pemilihan metode untuk menghapus data saja bukanlah hal yang bijak. Hal tersebut akan mengakibatkan model yang nantinya akan dibuat kehilangan banyak informasi. Sehingga dipilihlah cara untuk memanipulasi datanya, dengan mengisi data yang kosong dengan nilai rata-rata kolomnya. Data rata-rata kolom dipilih karena merupakan data yang dipastikan bukan data pencilan. Sehingga dengan menganggap data kosong sebagai data rata-rata, model tetap dapat memperoleh informasi dari data yang ada pada kolom lainnya. Proses yang dilakukan pertama-tama dengan cara mengambil nilai rata-rata dari kolom yang memiliki data kosong, kemudian memasukannya kepada setiap data kosong sebagai pengganti dari datanya. Semua proses tersebut dilakukan dengan slicing data dengan kondisi 
* Untuk mengatasi data kosong dengan nilai rata-rata kolom (mean substition),maka rata-rata yang memiliki data kosong dengan membuat kolom Potability = 0 dan selanjutnya membuat kolom Potability = 1

menggunakan panda
* Ini melakukan pengganti data kosong dengan nilai rata-rata kolom dengan memasukan kedalam variable df. Dalam hal ini, kita menggunakan nilai rata-rata dari masing-masing kolom sebagai nilai pengganti untuk nilai-nilai yang hilang. Argumen inplace=True digunakan untuk melakukan perubahan langsung pada dataframe df tanpa perlu menyimpan hasilnya dalam variabel baru,juga mengecek kembali nilai yang kosong pada dataset.

* Disini perbedaan antara nilai rata-rata dan median Air Minum dan Air Non-Minum sangat kecil. Jadi menggunakan median keseluruhan fitur untuk menghubungkan nilai

* Disini melakukan pengecekan total baris dalam kolom dari dataset
[![prepation-5.png](https://i.postimg.cc/zBcyTj3J/prepation-5.png)](https://postimg.cc/njqVJqv5)

* Ini juga  menghitung masing- masing nilai pada kolom 'Potability' dalam variable 'df' dengan menunjukan apakah suatu sample air layak/tidak.Untuk mengatasi masalah ini, dilakukan resampling pada data dengan label 1 sehingga jumlah data pada label 1 menjadi seimbang dengan jumlah data pada label 0. Setelah resampling, dataset yang sudah seimbang tersebut disimpan kembali dalam dataframe df.
[![resample.png](https://i.postimg.cc/Wzm68CZG/resample.png)](https://postimg.cc/D4ZXyCB0)

* Melakukan pembagian dataset pada dataset train_test_split menjadi 80% dan 20% untuk data uji setelah melakukan pra-pemrosesan ke dataset, sehingga perbandingan ratio menjadi 80:20.Untuk Data latih sendiri hanya melatih model,Pembagian ini menggunakan modul train_test_split dan scikit-learn.[![pembagian-dataset.png](https://i.postimg.cc/2jHRmSKN/pembagian-dataset.png)](https://postimg.cc/PN8FQTHS)

* Melakukan pencicilan pada data latih dengan [metode LOF(Local Outlier Factor)](http://etd.repository.ugm.ac.id/penelitian/detail/183405).Data pencilan merupakan nilai yang yang tidak normal dalam dataset yang mengakibatkan distorsi pada analisis statistika dan berujung pada pembuatan model yang kurang optimal.Metode ini cocok digunakan untuk data bertipe runtutan waktu. Data yang telah bersih dan telah dihaluskan dapat digunakan untuk melakukan prediksi. Penelitian ini menggunakan metode Artificial Neural Network (ANN), Gated Recurrent Unit (GRU) dan Long-Short Term Memories (LSTM) untuk melakukan prediksi.[![LOF.png](https://i.postimg.cc/66cFdSMB/LOF.png)](https://postimg.cc/7bCB4BvR)

* Melakukan **standardisasi data** pada semua fitur data.Tahap terakhir dengan melakukan standardisasi data.Hal ini akan membuat semua fitur numerik dataset memiliki skala yang sama dengan menggunakan MinMaxScaler.[![standarisasi-data.png](https://i.postimg.cc/90Cw0KSf/standarisasi-data.png)](https://postimg.cc/sQHg063F)

## Modeling

Setelah  melakukan pra-pemrosesan pada dataset. Untuk selanjutnya adalah modeling terhadap data. Pada tahap ini menggunakan 2 algoritma K-Nearest dengan  menggunakan data model terlatih sehingga data diukur bilai akurasinya.

* Model baseline adalah model awal yang digunakan sebagai pembanding atau titik awal dalam membangun model yang lebih kompleks atau dioptimalkan.melatih model baseline menggunakan data latih (X_train dan y_train). Metode fit() digunakan untuk mengajarkan model untuk mempelajari pola atau hubungan antara fitur-fitur dalam data latih dan label yang sesuai.[[3]](https://stephenallwright.com/baseline-machine-learning-models/)
[![model-1.png](https://i.postimg.cc/V6FS21sF/model-1.png)](https://postimg.cc/MMvKjCQc) Untuk menyimpan prediksi confussion matrix [![model-2.png](https://i.postimg.cc/ZKLGPLSz/model-2.png)](https://postimg.cc/qzzD4yq1)
Pada Model berbandingan dengan algoritma K-Nearest Neighbor,dimana membuktikan apakah kedua model dapat diuji dan divisualisasikan pada confussion matrix.
* Hasil Model baseline
[![model-5.png](https://i.postimg.cc/5ttfT2Jy/model-5.png)](https://postimg.cc/jDGVJKLY)
* Hasil Model yang dikembangkan (model yang dapat digunakan
[![model-6.png](https://i.postimg.cc/BvRRpSQd/model-6.png)](https://postimg.cc/5HqPNMz3)

## Evalution
Pada proyek ini, model yang dikembangkan adalah Dalam proyek ini, sistem yang dikembangkan adalah suatu jenis klasifikasi dan mengukur performanya menggunakan metrik akurasi, f1-skor, recall, dan presisi. Berikut adalah hasil pengukuran dari model yang dipilih, yakni model yang menggunakan algoritma Pohon Acak (Random Forest), dengan metrik akurasi, f1-skor, recall, dan precision.
* Laporan hasil klasifikasi model_baseline
[![Evaluation-1.png](https://i.postimg.cc/52GpLzHc/Evaluation-1.png)](https://postimg.cc/18GpZfcM)

* Rumus Akurasi
[![akurasi-1.png](https://i.postimg.cc/BnzbC1Yw/akurasi-1.png)](https://postimg.cc/9DGcmzjG)

Rumus diatas merupakan metrik akurasi yang menghitung ketepatan model dalam hal ini meprediksi data dengan data yang sebenarnya.Untuk kelebihan sendiri dalam pembuatan model klasifikasi baik itu klasifikasi antar dua kelas maupun kategori, selain itu perthitungan ini memiliki kekurangan yang biasanya dapat menyesatkan terutama data yang tidak seimbang.

* _Precision_ merupakan numerik untuk melakukan prediksi benar positifnya hasil suatu prediksi,untuk rumus sendiri _Precision_ = (TP)/(TP + TP)

 *_Recall_ merupakan metrik untuk memprediksi benar positifnya berdasarkan keseluruhan data,untuk rumus sendiri _Recall_ = (TP)/(TP + FN)
 * _f1-score f1-score merupakan metrik perbandingan antara precision dan recall yang dibobotkan,sedangkan Rumus f1-score sebagai berikut:
 [![f1.png](https://i.postimg.cc/VvpxxcJq/f1.png)](https://postimg.cc/GTPgDVJt)

## Referensi
- [[1]](http://publikasi.dinus.ac.id/index.php/technoc/article/view/5901)Hardiana Said, Nur Hafifah Matondang, Helena Nurramdhani Irmanda,PENERAPAN ALGORITMA K-NEAREST NEIGHBOR UNTUK MEMPREDIKSI KUALITAS AIR YANG DAPAT DIKONSUMSI,publikasi dinus,Vol 21, No. 2,2020.
- [[2]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/)Ramadhan Rakhmat Sani, Junta Zeniarja, Ardytha Luthfiarta,Penerapan Algoritma K-Nearest Neighbor pada Information Retrieval dalam Penentuan Topik Referensi Tugas Akhir,publikasi dinus,Vol 1,No. 2,(2016).
- [[3]]























