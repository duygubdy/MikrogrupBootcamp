# HRWELL - Machine Learning Integrated HR Tool

## Proje Amacı
HRWELL, İnsan Kaynakları süreçlerinde veri analizini ve makine öğrenmesini entegre eden bir araçtır. Bu proje, çalışan verilerini analiz ederek çalışan yıpranmasını ve performansını önceden tahmin etmeyi hedefler. Dinamik veri analizi toolu ile çalışan yıpranma analizine doğrudan erişim sağlar ve yüksek doğruluklu bir model sunar.

## Proje Hedeflerimiz
- Çalışan yıpranmasını ve performansını doğru bir şekilde tahmin etmek.
- İnsan Kaynakları süreçlerini veri odaklı hale getirmek.
- Geliştirilen model ve analiz araçlarıyla iş verimliliğini artırmak.

## Ekibimiz
- **Duygu Baday**
  - **Görevler:** Flask, Veri Analiz Toolu, Local Storage, Model Entegrasyonu
  - **Kullandığı Teknolojiler: **
    - Frontend: HTML, CSS, JavaScript, Toastr
    - Backend: Flask, Python
    - Veri Depolama: Local Storage
      
- **Efnan Durmazer**
  - **Görevler:** Veri Seti Araştırması, Veri İnceleme ve Analiz
  - **Kullandığı Teknolojiler: **
    - Veri Analizi: Pandas, NumPy
    - Görselleştirme: Matplotlib, Seaborn, Plotly
      
- **Hatice Akkuş**
  - **Görevler:** Veri Seti Hazırlığı, Model Eğitimi, Veri Temizleme, Görselleştirme
  - **Kullandığı Teknolojiler: **
    - Veri İşleme: Pandas(One-Hot Encoding)
    - Veri Dengesi: Imbalanced-learn (ADASYN)
    - Modelleme: Scikit-learn (Logistic Regression, Random Forest, Support Vector Machine (SVM), Gradient Boosting)
    - Model Değerlendirmesi: Scikit-learn (Accuracy, Classification Report, Cross-Validation)

- **Cansu Ardıç**
  - **Görevler:** Veri Analizi ve UI Tasarımı, Raporlama
  - **Kullandığı Teknolojiler: **
    - UI Tasarımı: HTML, CSS
    - Veri Analizi: Pandas, Matplotlib


## Verilerimiz
### Veri Seti
- **Kaynak:** Kaggle.com
- **Boyut:** 1323 satır ve 26 sütun
- **Değişkenler:**
  - Yaş
  - İş Seyahati Sıklığı
  - Departman
  - Evin İşe Uzaklığı
  - Eğitim Seviyesi
  - Eğitim Alanı
  - Çevre Memnuniyeti
  - Cinsiyet
  - İşe Bağlılık
  - İş Seviyesi
  - İş Rolü
  - İş Memnuniyeti
  - Medeni Durum
  - Aylık Gelir
  - Fazla Mesai
  - Performans Değerlendirmesi
  - İlişki Memnuniyeti
  - Hisse Senedi Opsiyon Seviyesi
  - Toplam Çalışma Yılları
  - Geçen Yılki Eğitim Sayısı
  - İş-Yaşam Dengesi
  - Şirketteki Yıllar
  - Mevcut Roldeki Yıllar
  - Son Terfiden Bu Yana Geçen Yıllar
  - Mevcut Yöneticisiyle Geçirdiği Yıllar
  - İşten Ayrılma
  - Yıpranma

## Modelimiz ve Çıktılarımız
### Veri Analizi 
1. Keşifçi Veri Analizi (EDA)
2. Değişkenlerin İncelenmesi ve Görselleştirilmesi
3. Aykırı Değer ve Eksik Veri Analizi
4. Raporlama


### Model Performansı
| Yıpranma           | Eğitim Doğruluğu | Test Doğruluğu | Çapraz Doğrulama |
|--------------------|------------------|----------------|------------------|
| Logistic Regression| 0.914            | 0.901          | 0.900            |
| Random Forest      | 0.831            | 0.809          | 0.810            |
| SVM Linear         | 0.910            | 0.901          | 0.896            |
| SVM RBF            | 0.925            | 0.910          | 0.893            |
| Gradient Boosting  | 0.941            | 0.900          | 0.895            |

### Model Performansı
| Performans         | Eğitim Doğruluğu | Test Doğruluğu | Çapraz Doğrulama |
|--------------------|------------------|----------------|------------------|
| Logistic Regression| 0.862            | 0.854          | 0.843            |
| Random Forest      | 0.870            | 0.827          | 0.821            |
| SVM Linear         | 0.861            | 0.861          | 0.848            |
| SVM RBF            | 0.888            | 0.886          | 0.865            |
| Gradient Boosting  | 0.932            | 0.884          | 0.853            |

 En iyi sonuç Yıpranma tahmini için Logistic Regression, Performans tahmini için SVM RBF ile alındı.
 
#### Yıpranma Durum Modeli Sınıflandırma Raporu (Logistic Regression)

|   Sınıf      | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Hayır        | 0.87      | 0.94   | 0.90     | 222     |
| Evet         | 0.93      | 0.87   | 0.90     | 224     |
| accuracy     |           |        | 0.90     | 446     |
| macro avg    | 0.90      | 0.90   | 0.90     | 446     |
| weighted avg | 0.90      | 0.90   | 0.90     | 446     |


#### Performans Değerlendirme Modeli Sınıflandırma Raporu (SVM RBF)

|   Sınıf      | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Düşük        | 0.85      | 0.94   | 0.89     | 224     |
| Yüksek       | 0.93      | 0.83   | 0.88     | 215     |
| accuracy     |           |        | 0.89     | 439     |
| macro avg    | 0.89      | 0.89   | 0.89     | 439     |
| weighted avg | 0.89      | 0.89   | 0.89     | 439     |

## Uygulamamıza Ait Ekran Görüntüleri


## Kurulum
Projenizi yerel olarak çalıştırmak için aşağıdaki adımları izleyin:

### Gereksinimler

### Adımlar
1. Bu repoyu klonlayın:
   ```bash
   git clone https://github.com/haticeakkus/HRWELL.git

2. Proje dizinine gidin:
   ```bash
   cd HRWELL

3. Sanal ortam oluşturun:
   ```bash
   python -m venv env

4. Sanal ortam etkinleşirin(Windows):
   ```bash
   .\env\Scripts\activate

5. Sanal ortam etkinleşirin(macOS/Linux):
   ```bash
   source env/bin/activate

5. Gerekli bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt

6. Flask uygulamasını başlatın:
   ```bash
   flask run


