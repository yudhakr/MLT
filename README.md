
# Laporan Proyek Machine Learning - Ayudha Kusuma Rahmadhani
Judul proyek : "Analisis Prediktif Data Air Bersih Di Masyarakat".

## Domain Proyek
[![Analisis-Prediktif-Data-Air-Bersih-Di-Masyarakat.jpg](https://i.postimg.cc/pTbscM81/Analisis-Prediktif-Data-Air-Bersih-Di-Masyarakat.jpg)](https://postimg.cc/DW6QmDxq)
   Kualitas air yang baik adalah faktor penting bagi kesehatan manusia.
Manusia sangat bergantung pada air yang aman dan bersih untuk memenuhi kebutuhan sehari-hari, seperti minum, memasak, dan menjaga kebersihan pribadi.
Air yang terkontaminasi oleh polusi, seperti logam berat, pestisida, bakteri, dan virus, dapat menyebabkan berbagai masalah kesehatan yang serius, termasuk penyakit perut, masalah pernapasan, keracunan, dan gangguan sistem kekebalan tubuh. 

   Berdasarkan terori masalah kualitas air yang umum terjadi termasuk kontaminasi feses, penyakit yang disebabkan oleh air yang tidak aman, dan tantangan akibat perubahan iklim dan faktor lainya yang menurunnya kualitas air
terutama untuk dikomsumsi.Kualitas fisikokimia, bakteriologis, dan konsentrasi logam jejak sampel air dari sumber yang diolah, keran jalan, dan wadah penyimpanan rumah tangga sebagian besar berada dalam kisaran yang diizinkan standar air minum WHO dan SANS.

Maka dari permasalahan tersebut penulis ingin membuat sistem model nalisis Prediktif Data Air Bersih Di Masyarakat berdasarkan variabel-variabel tertentu,sehingga masyarakat akan tahu dan berhati-hati dalam penggunaan air dan tahu cara menglola air yang terkontaminasi material tertentu.Juga diharapkan model machine learning ini diharapkan dapat memudahkan ahli seperti ahli hidrologi dan para pencari sumber air dalam mencari air dan mengujinya secara cepat sebelum menunggu hasil dari laboratorium. Implementasinya model ini dapat dijalankan dan diterapkan pada sebuah aplikasi [[1]](https://www.mendeley.com/search/?page=1&query=%20K-Nearest%20Neighbor%20air%20bersih&sortBy=relevance)


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
* 
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
<img width="875" alt="dataset" src="https://github.com/yudhakr/MLT/assets/84507343/a07fef2d-96a7-4384-944a-c4a4b19281f9">



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
* Mengatasi data yang tidak seimbang jumlahnya dengan label lain menggunakan teknik resample.Dataset yang tidak seimbang pada data kategori akan menyebabkan model yang dibuat menjadi bias terhadap suatu kategori yang memiliki data lebih banyak. Oleh karena itu diperlukan teknik manipulasi data, dan yang digunakan di sini adalah teknik resample. Prosesnya adalah dengan memasukan kolom yang memiliki data paling sedikit pada fungsi resample, kemudian fungsi resample akan menghasilkan data baru dari data yang sudah ada sebelumnya sampai jumlah datanya sama dengan data mayoritas dari label selainnya. Setelah itu selesai, masukan datanya kedalam dataset agar menjadi satu kesatuan data.
* Agar dapat menguji performa model pada data sebenarnya, maka perlu dilakukan pembagian dataset kedalam dua atau tiga bagian. Pada proyek ini dilakukan dua bagian saja yakni pada data latih dan data uji dengan rasio 80:20. Data latih dilakukan sepenuhnya untuk melatih model, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. Pada bagian ini dipastikan juga pembagian label kategorikal haruslah sama banyak pada data latih dan data uji. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn
* Menghapus data pencilan pada data latih dengan metode LOF Local Outlier Factor, pada bagian ini diterapkan metode Local Outlier Factor untuk mendeteksi nilai outlier dan kemudian menghapusnya dari data latih. Mengapa data latih saja? Agar kita dapat melihat bagaimana performa model pada data yang belum pernah dilihat model sebelumnya termasuk juga data pencilan. Mengapa dipilih metode LOF? karena metode ini berhubungan erat prosesnya dengan algoritma nearest neighbor. Mengutip dari dokumentasi LocalOutlierFactor, fungsi tersebut bekerja dengan cara menganalisis nilai lokalitas yang ada pada k-tetangga terdekat, yang jaraknya digunakan untuk memperkirakan kepadatan lokal. Dengan membandingkan kepadatan lokal sampel dengan kepadatan lokal tetangganya, seseorang dapat mengidentifikasi sampel yang memiliki kepadatan jauh lebih rendah daripada tetangganya. Apabila kepadatannya rendah maka ini dianggap outlier.
* Tahap terakhir dengan melakukan standardisasi data. Hal ini akan membuat semua fitur numerik berada dalam skala data yang sama juga membuat komputasi dari pembuatan model dapat berjalan lebih cepat karena rentang datanya hanya antara 0-1. Untuk melakukan standardisasi data, digunakan fungsi MinMaxScaler yang perhitungannya kurang lebih seperti rumus di bawah ini.


## Modeling

Setelah  melakukan pra-pemrosesan pada dataset. Untuk selanjutnya adalah modeling terhadap data. Pada tahap ini menggunakan 2 algoritma K-Nearest dengan  menggunakan data model terlatih sehingga data diukur bilai akurasinya.

* Model baseline adalah model awal yang digunakan sebagai pembanding atau titik awal dalam membangun model yang lebih kompleks atau dioptimalkan.melatih model baseline menggunakan data latih (X_train dan y_train). Metode fit() digunakan untuk mengajarkan model untuk mempelajari pola atau hubungan antara fitur-fitur dalam data latih dan label yang sesuai.

<img width="430" alt="model 1" src="https://github.com/yudhakr/MLT/assets/84507343/05580f53-6a31-4c80-855d-a8505908a7e1">

 Untuk menyimpan prediksi confussion matrix 
 
 <img width="236" alt="model 2" src="https://github.com/yudhakr/MLT/assets/84507343/ed66b314-6235-4bc4-b500-0d748ae25a93">

Pada Model berbandingan dengan algoritma K-Nearest Neighbor,dimana membuktikan apakah kedua model dapat diuji dan divisualisasikan pada confussion matrix.
* Hasil Model baseline
<img width="369" alt="model 5" src="https://github.com/yudhakr/MLT/assets/84507343/0597d1d3-3f69-4bba-99cf-55f569ba2aee">

* Hasil Model yang dikembangkan (model yang dapat digunakan
<img width="338" alt="model 6" src="https://github.com/yudhakr/MLT/assets/84507343/49088d17-3906-4c69-b613-d45f3b69c0ad">


## Evalution
Pada proyek ini, model yang dikembangkan adalah Dalam proyek ini, sistem yang dikembangkan adalah suatu jenis klasifikasi dan mengukur performanya menggunakan metrik akurasi, f1-skor, recall, dan presisi. Berikut adalah hasil pengukuran dari model yang dipilih, yakni model yang menggunakan algoritma Pohon Acak (Random Forest), dengan metrik akurasi, f1-skor, recall, dan precision.
* Laporan hasil klasifikasi model_baseline

|   |              | precision |  recall | f1-score |   support |
|---|-------------:|----------:|--------:|---------:|----------:|
|   |  Not Potable |  0.707124 | 0.67000 | 0.688062 | 400.00000 |
|   |    Potable   |  0.686461 | 0.72250 | 0.704019 | 400.00000 |
|   |   accuracy   |  0.696250 | 0.69625 | 0.696250 |   0.69625 |
|   |   macro avg  |  0.696792 | 0.69625 | 0.696041 | 800.00000 |
|   | weighted avg |  0.696792 | 0.69625 | 0.696041 | 800.00000 |

* Rumus Akurasi
<img width="251" alt="akurasi 1" src="https://github.com/yudhakr/MLT/assets/84507343/9c4abd6f-704e-475d-b343-f647934bb0a8">

Rumus diatas merupakan metrik akurasi yang menghitung ketepatan model dalam hal ini meprediksi data dengan data yang sebenarnya.Untuk kelebihan sendiri dalam pembuatan model klasifikasi baik itu klasifikasi antar dua kelas maupun kategori, selain itu perthitungan ini memiliki kekurangan yang biasanya dapat menyesatkan terutama data yang tidak seimbang.

* _Precision_ merupakan numerik untuk melakukan prediksi benar positifnya hasil suatu prediksi. Kelebihan dari metriks ini berfokus pada bagaimana performa (prediksi) model terhadap label data positif, kekurangannya metriks ini tidak memperhitungkan label negatifnya,untuk rumus sendiri _Precision_ = (TP)/(TP + TP)

 *_Recall_ merupakan metrik untuk memprediksi benar positifnya berdasarkan keseluruhan data. Recall merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua label data positif, untuk rumus sendiri _Recall_ = (TP)/(TP + FN)
 * _f1-score f1-score merupakan metrik perbandingan antara precision dan recall yang dibobotkan.metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik hasil prediksi model (precision) dan seberapa lengkap hasil prediksinya (recall).sedangkan Rumus f1-score sebagai berikut:
$$F1 Score = 2* (Recall*Precission) / (Recall + Precission) $$

*** _Catatan : Nilai beta = 1 (f1-skor)_***

Kelebihan dari metriks ini menutup semua kekurangan yang ada pada precision dan recall. Namun kekurangannya adalah f1-score tidak memperhitungkan hasil prediksi benar pada label negatif.
 

## Referensi
- [[1]](http://publikasi.dinus.ac.id/index.php/technoc/article/view/5901)Hardiana Said, Nur Hafifah Matondang, Helena Nurramdhani Irmanda,Penerapan Algoritma K-Nearest Neighbors Untuk Memprediksi Kualitas Air Yang Dapat Dikonsumsi,publikasi dinus,Vol 21, No. 2,2020.
- [[2]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/)Ramadhan Rakhmat Sani, Junta Zeniarja, Ardytha Luthfiarta,Penerapan Algoritma K-Nearest Neighbor pada Information Retrieval dalam Penentuan Topik Referensi Tugas Akhir,publikasi dinus,Vol 1,No. 2,2016.
- [[3]] Kelleher, John D, et al. "Machine Learning for Predictive Data Analytics". MIT Press.2020.
- [[4]](https://www.kaggle.com/datasets/adityakadiwal/water-potability) Aitya Kadiwak,Water Quality Drinking water potability,2021, from https://www.kaggle.com/datasets/adityakadiwal/water-potability
- [[5]] Khoiri. (2020). Cara Menghitung Mean Squared Error (MSE). Khoiri. Retrieved September 16, 2022, from https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html.























