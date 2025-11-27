# 🏠 House Prices Prediction: End-to-End Machine Learning Project

## 📌 Proje Özeti
Bu proje, Kaggle'ın ünlü "House Prices: Advanced Regression Techniques" veri seti kullanılarak geliştirilmiştir. Amaç, evlerin fiziksel özelliklerine dayanarak satış fiyatlarını (SalePrice) makine öğrenmesi algoritmaları ile tahmin etmektir. Proje; veri temizleme, özellik mühendisliği (feature engineering) ve model optimizasyonu adımlarını uçtan uca kapsamaktadır.

## 🛠️ Kullanılan Teknikler ve Kütüphaneler
* **Python:** Pandas, Numpy
* **Veri Görselleştirme:** Seaborn, Matplotlib
* **Makine Öğrenmesi:** Scikit-learn, Gradient Boosting Regressor (GBM)
* **Veri Ön İşleme:**
    * Outlier Handling (Aykırı Değer Baskılama)
    * Missing Value Imputation (Eksik Değer Atama)
    * Rare Encoding & Label/One-Hot Encoding
* **Model Tuning:** GridSearchCV ile Hiperparametre Optimizasyonu

## 📊 Proje Adımları
1.  **EDA (Keşifçi Veri Analizi):** Veri setinin yapısı incelendi, kategorik ve numerik değişkenler ayrıştırıldı.
2.  **Preprocessing:**
    * Aykırı değerler tespit edildi ve baskılandı.
    * Eksik veriler, değişkenlerin karakterine göre (örneğin havuz yoksa 'Yok' etiketi, metrekare ise medyan değeri ile) dolduruldu.
3.  **Feature Engineering:**
    * `NEW_Total_House_Area`, `NEW_House_Age` gibi model başarısını artıran yeni değişkenler türetildi.
4.  **Log Transformation:** Hedef değişken (`SalePrice`) logaritma dönüşümü ile normal dağılıma yaklaştırıldı.
5.  **Modelleme:** Gradient Boosting Regressor kullanıldı.
6.  **Değerlendirme:** Model, **%89** açıklayıcılık (R2 Score) ve düşük RMSE değeri ile optimize edildi.

## 📈 Sonuçlar
* **RMSE (Root Mean Squared Error):** ~23.000$
* **R2 Score:** ~0.89
* Model, test setindeki ev fiyatlarını %85-90 başarı oranıyla tahmin etmektedir.

---
*Bu proje Miuul Data Science Bootcamp kapsamında geliştirilmiştir.*
