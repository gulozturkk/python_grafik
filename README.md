# 📊 Çalışanların İşten Ayrılma Durumunu Tahmin Etme (Employee Attrition Prediction)

Bu proje, çalışanların demografik ve iş yeri bilgilerine (aylık gelir, yaş, iş memnuniyeti ve ev-iş mesafesi) dayanarak işten ayrılma (attrition) olasılıklarını tahmin eden bir makine öğrenmesi sınıflandırma modelidir[cite: 2, 5]. Şirketlerin işten ayrılma riski yüksek çalışanları önceden tespit ederek gerekli önlemleri alabilmesine yardımcı olmayı amaçlar[cite: 5].

---

## 🎯 Problem Tanımı
Çalışanların işten ayrılması şirketler için hem yüksek maliyetli hem de iş süreçlerini aksatan kritik bir problemdir[cite: 5]. Bu projede çalışan verileri analiz edilerek ikili sınıflandırma (binary classification) yapabilen bir model geliştirilmiştir[cite: 2, 5]:
* **`1` (Evet):** Çalışan işten ayrıldı[cite: 2, 5]
* **`0` (Hayır):** Çalışan işte kaldı[cite: 2, 5]

---

## 📁 Veri Seti ve Ön İşleme
* **Kaynak:** Kaggle (IBM HR Analytics Employee Attrition & Performance)[cite: 6]
* **Boyut:** 1.470 çalışan kaydı ve 35 özellik[cite: 2, 6]
* **Veri Temizleme:** Eksik değer (null/NaN) kontrolü yapılmış ve veri setinde eksik kayıt bulunmadığı doğrulanmıştır[cite: 2, 6]. Hedef değişken olan `Attrition` sütunu sayısal formata (`Yes: 1`, `No: 0`) dönüştürülmüştür[cite: 2].
* **Seçilen Özellikler (Features):**
  * `Age` (Yaş)[cite: 2, 5]
  * `MonthlyIncome` (Aylık Gelir)[cite: 2, 5]
  * `JobSatisfaction` (İş Memnuniyeti Derecesi: 1-4)[cite: 2, 5]
  * `DistanceFromHome` (Ev-İş Arası Mesafe)[cite: 2, 5]

---

## 🔍 Keşifçi Veri Analizi (EDA) Bulguları
Veri görselleştirme ve istatistiksel incelemeler sonucunda öne çıkan temel bulgular şunlardır[cite: 2, 6]:
* **Gelir ve Ayrılma İlişkisi:** Düşük aylık gelire sahip çalışanların işten ayrılma eğiliminin belirgin şekilde daha yüksek olduğu gözlemlenmiştir[cite: 2, 6].
* **Memnuniyet Ortalaması:** Veri setindeki çalışanların genel iş memnuniyeti puanları ortalama **2.7** civarındadır[cite: 2, 6].
* **Yaş ve Gelir Dağılımı:** Yaş ilerledikçe çalışanların aylık gelirlerinin genel olarak artış gösterdiği tespit edilmiştir[cite: 2, 6].
* **Özellik Önem Dereceleri:** Model kararlarında en belirleyici faktör açık ara farkla **Aylık Gelir (`MonthlyIncome`)** olurken, onu sırasıyla Yaş (`Age`), Ev-İş Mesafesi (`DistanceFromHome`) ve İş Memnuniyeti (`JobSatisfaction`) takip etmektedir[cite: 3].

---

## ⚙️ Yöntem ve Modelleme
1. **Veri Bölünmesi:** Modelin genelleme yeteneğini ölçmek ve ezberlemesini (overfitting) önlemek amacıyla veri seti `train_test_split` ile **%80 Eğitim (1.176 örnek)** ve **%20 Test (294 örnek)** olarak ikiye ayrılmıştır[cite: 2, 7].
2. **Algoritma Seçimi:** Karar mekanizmasının açıklanabilir olması ve sınıflandırma problemine uygunluğu nedeniyle **Karar Ağacı Sınıflandırıcı (`DecisionTreeClassifier`)** tercih edilmiştir[cite: 2, 7].
3. **Eğitim ve Tahmin:** Model eğitim seti üzerinde eğitilmiş (`model.fit`)[cite: 2, 7] ve hem test seti hem de yeni oluşturulan çalışan profilleri üzerinde tahminler gerçekleştirilmiştir[cite: 2]. Ayrıca kullanıcıdan terminal üzerinden girdi alarak anlık tahmin yapabilen interaktif bir fonksiyon (`calisan_ayrilma_tahmin_etme()`) eklenmiştir.

---

## 📈 Sonuçlar ve Değerlendirme
* **Test Doğruluğu (Accuracy):** Görülmemiş test verileri üzerinde **%81.63** doğruluk oranı elde edilmiştir[cite: 2, 8].
* **Değerlendirme:** Doğruluk oranının %100 olmaması, modelin eğitim verisini ezberlemediğini (overfitting riskinden kaçınıldığını) ve temel bir Karar Ağacı modeli için güvenilir bir genelleme performansı sunduğunu göstermektedir[cite: 3, 8].

---

## 🚀 Gelecek Geliştirmeler
Model performansını daha da ileriye taşımak için planlanan adımlar[cite: 3]:
* **Özellik Mühendisliği:** Fazla mesai (`OverTime`), departman bilgisi (`Department`), şirketteki çalışma süresi gibi ek değişkenlerin modele dahil edilmesi[cite: 3].
* **Model Çeşitliliği:** `RandomForestClassifier`, `LogisticRegression` veya `XGBoost` gibi farklı sınıflandırma algoritmalarının denenerek karşılaştırılması[cite: 3].
* **Çapraz Doğrulama:** Model kararlılığını artırmak için k-katlı çapraz doğrulama (`cross_val_score`) uygulanması[cite: 3].

---

## 🛠️ Kurulum ve Çalıştırma

1. Repoyu bilgisayarınıza klonlayın:
   git clone [https://github.com/gulozturkk/repo-adi.git](https://github.com/gulozturkk/repo-adi.git)
   cd repo-adi
2.Gerekli kütüphaneleri yükleyin:
  pip install pandas numpy matplotlib scikit-learn jupyter

3.Jupyter Notebook dosyasını başlatın ve hücreleri sırayla çalıştırın:
  jupyter notebook "Basitleştirilmiş_Python_Proje (1).ipynb"
