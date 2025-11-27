import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#veri seti okuma
test = pd.read_csv("veri_setleri/test.csv")
train = pd.read_csv("veri_setleri/train.csv")
#veri seti birleştirme(index i tuttuk karışmasın diye)
df = pd.concat([train, test], axis=0).reset_index(drop=True)
df.shape
# veriye genel bakış
def data_ozet( dataframe, head=5 ):
    print("######### Shape ########")
    print(dataframe.shape)
    print("######### Type ########")
    print(dataframe.dtypes)
    print("######### Head ########")
    print(dataframe.head(head))
    print("######### Tail ########")
    print(dataframe.tail(head))
    print("######### NaN ########")
    print(dataframe.isnull().sum())
data_ozet(df)


# verinin değişkenlerini tespit ettik
def degisken_analiz(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = degisken_analiz(df)


#Burada bu sınıf sayısal olarak görünüyor ama aslında inşaat tipi kodları olduğu için sayısal olarak bir anlam ifade etmiyor.
df["MSSubClass"] = df["MSSubClass"].astype(str)
#Burada ise satış ayı değişkeni 12 ye kadar sayısal bir anlam ifade etmiyor çevirdik...
df["MoSold"] = df["MoSold"].astype(str)


 #Kategorik Değişken Analizi
def kategorik_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.savefig("feature_importance.png")
        print("Grafik kaydedildi: feature_importance.png")


# Numerik Değişken Analizi
def numerik_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.savefig("feature_importance.png")
        print("Grafik kaydedildi: feature_importance.png")

print("--- KATEGORİK DEĞİŞKENLERİN DAĞILIMI ---")
for col in cat_cols:
    kategorik_summary(df, col, plot=True)

print("\n--- NUMERİK DEĞİŞKENLERİN DAĞILIMI ---")
for col in num_cols:
    numerik_summary(df, col, plot=False)

# Adım_5: Kategorik ve Nümerik Değişkenlerin Hedef Değişkene Göre Analizi

def hedef_degisken_ile_kategorik_analiz(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

print("--- KATEGORİK DEĞİŞKENLERE GÖRE HEDEF ORTALAMALARI ---")

for col in cat_cols:

    if col != "SalePrice":
        print(f"###### {col} ######")
        hedef_degisken_ile_kategorik_analiz(df, "SalePrice", col)

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
plt.savefig("feature_importance.png")
print("Grafik kaydedildi: feature_importance.png")

#bence gerek yok:) korelasyon a bakalım
def hedef_degisken_ile_numerik_analiz(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


#Aykırı değer var mı yok mu?
cat_cols, num_cols, cat_but_car = degisken_analiz(df)
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    # EMNİYET KİLİDİ BURADA:
    # 1. Sütun 'SalePrice' olmasın (Hedef değişken)
    # 2. Sütun 'Object' (String) OLMASIN. Sadece sayısal olsun.
    if col != "SalePrice" and pd.api.types.is_numeric_dtype(df[col]):
        try:
            if check_outlier(df, col):
                print(f"{col}: VAR")
        except Exception as e:
            # Hata veren sütunu atla ve ismini yazdır ki bilelim
            print(f"ATLANDI: {col} değişkeninde hata oluştu -> {e}")
for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))

#Eksik değer tablosu
def missing_values_table(dataframe, na_name=False):
    # Eksik değeri olan sütunları bul
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    # Sayılarını ve oranlarını hesapla
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    # Tablo haline getir
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print("\n--- EKSİK DEĞER TABLOSU ---")
    print(missing_df)

    if na_name:
        return na_columns
missing_values_table(df)

# BONUS : KORELASYON ANALİZİ

df[num_cols].corr()

f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block = True)
plt.savefig("corr.png")
print("Grafik kaydedildi: feature_importance.png")



# FEATURE ENGİNEERİNG

# AYKIRI DEĞER BASKILAMA(SALEPRİCE HARİÇ)
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # bütün int değerlerini float a çevirdik hata düzeldi.
    if dataframe[variable].dtype == 'int64':
        dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
# baskılama işlemi(salesprice hariç)
for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

# EKSİK DEĞER DOLDURMA

#A) "Yokluk" (No Pool, No Garage vb.) Anlamına Gelen Boşluklar

none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
             "GarageType", "GarageFinish", "GarageQual", "GarageCond",
             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]

for col in none_cols:
    df[col] = df[col].fillna("No")

#B) Sayısal Olup "Yok" Anlamına Gelenler (Alan = 0 vb.)
zero_cols = ["GarageYrBlt", "GarageArea", "GarageCars",
             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
             "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]

for col in zero_cols:
    df[col] = df[col].fillna(0)

# C) LotFrontage (Ev Cephesi): Mahalle kırılımında medyan ile dolduruyoruz
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# D) Kalan Ufak Eksikler (Mod ve Medyan)
# Kategorik olanları en çok tekrar edenle (Mode)
for col in cat_cols:
    if df[col].isnull().sum() > 0 and col != "SalePrice":
        df[col] = df[col].fillna(df[col].mode()[0])

# Numerik olanları ortanca değerle (Median)
for col in num_cols:
    if df[col].isnull().sum() > 0 and col != "SalePrice":
        df[col] = df[col].fillna(df[col].median())

df.isnull().sum()

# Adım_2: RARE ENCODİNG

# 1. RARE ANALYZER (Önce durumu görelim)
# ------------------------------------------------------
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

print("--- RARE ANALİZİ BAŞLIYOR ---")
# SalePrice sadece Train setinde olduğu için analizde hata vermemesi adına filtreliyoruz
# Ancak tüm veri (df) üzerinde işlem yapacağımız için fonksiyonun iç yapısı önemli.
# Basitçe görmek için:
rare_analyser(df, "SalePrice", cat_cols)


# 2. RARE ENCODER (İşlemi uygulayalım)
# ------------------------------------------------------
def rare_encoder(dataframe, rare_perc):
    # kopyasını aldık
    temp_df = dataframe.copy()

    # Rare olacak sınıfları bulup tek çatı altında topladık
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])

    return temp_df


# Nadirlik oranı %1 (0.01) olarak belirleyelim.
# Yani veri setinde %1'den az bulunan sınıflar "Rare" olacak.
df = rare_encoder(df, 0.01)

# Kontrol edelim, bakalım "Rare" gelmiş mi?
print("Rare Encoder sonrası bazı sınıflar:")
print(df["Heating"].value_counts())  # Örnek kontrol

# ADIM_3 YENİ DEĞİŞKENLER TÜRETME

# 1. Toplam Ev Alanı
# 1. Kat + 2. Kat + Bodrum Alanı
df["NEW_Total_House_Area"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

# 2. Toplam Banyo Sayısı
df["NEW_Total_Bath"] = df["FullBath"] + (df["HalfBath"] * 0.5) + df["BsmtFullBath"] + (df["BsmtHalfBath"] * 0.5)

# 3. Toplam Veranda/Balkon Alanı (Porch Area)
df["NEW_Total_Porch_SF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] +
                            df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])

# 4. Evin Yaşı (Satıldığı Yıl - Yapıldığı Yıl)
df["NEW_House_Age"] = df["YrSold"] - df["YearBuilt"]

# 5. Restorasyon Yapılmış mı? (Yapım yılı ile Tadilat yılı farklıysa 1, değilse 0)
df["NEW_Is_Renovated"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

# 6. Restorasyon Üzerinden Geçen Yıl
df["NEW_Restoration_Age"] = df["YrSold"] - df["YearRemodAdd"]

# 7. Kalite Skoru (Kalite * Kondisyon)
df["NEW_Total_Qual"] = df["OverallQual"] * df["OverallCond"]

# 8. Lüks Özellikler: Havuz Var mı? (PoolArea > 0)
df["NEW_Has_Pool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)

# 9. Garaj Var mı?
df["NEW_Has_Garage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)

# --- KONTROL ---
print(df[["NEW_Total_House_Area", "NEW_House_Age", "NEW_Total_Bath"]].head())

# DİKKAT: Yeni değişkenler ürettiğimiz için değişken listelerimizi (cat_cols, num_cols) güncelleme
cat_cols, num_cols, cat_but_car = degisken_analiz(df)
print(f"\nYeni Değişken Sayısı: {df.shape[1]}")

# A) Label-Encoding

from sklearn.preprocessing import LabelEncoder

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
# değişken ürettikten sonra bazıları binary olmuş olabilir.
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
print(len(binary_cols))
print(binary_cols)
for col in binary_cols:
    df = label_encoder(df, col)

# B) ONE-HOT ENCODİNG

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

cat_cols, num_cols, cat_but_car = degisken_analiz(df)

# Label Encoder'dan geçmeyen (hala object olan) sütunları alalım
final_ohe_cols = [col for col in cat_cols if df[col].dtype == 'O']
# Veya num_but_cat olup encode edilmemişleri de eklemek gerekebilir ama şimdilik object'ler yeterli.

print("--- ONE-HOT ENCODING UYGULANIYOR ---")
df = one_hot_encoder(df, final_ohe_cols, drop_first=True)

print("İşlem Tamamlandı!")
print(f"Toplam Değişken Sayısı: {df.shape[1]}")
print(df.head())


# MODELLEME
train_df = df[df['SalePrice'].notnull()] #saleprice boş olmayanlar train
test_df = df[df['SalePrice'].isnull()] #saleprice boş olanlar test

y = train_df['SalePrice'] #bağımlı değişken(y)
X = train_df.drop(["Id", "SalePrice"], axis=1) #bağımsız değişkenler
print(f"Train Seti Boyutu (X): {X.shape}")
print(f"Hedef Değişken (y) Boyutu: {y.shape}")


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
gbm_model = GradientBoostingRegressor(random_state=42) #gbm
# log dönüşümü olmadan
gbm_model.fit(X_train, y_train)


df = pd.get_dummies(df, drop_first=True)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice']
X = train_df.drop(["Id", "SalePrice"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

  #hata çözümü
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

gbm_model = GradientBoostingRegressor(random_state=42)
gbm_model.fit(X_train, y_train)

y_pred = gbm_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Normal Model RMSE: {rmse:.4f}") #rmse= 24523.7129

 #log ile rmse değerine bakma
# Hedef değişkeni logaritma ile küçültüyoruz ki dağılım normale yaklaşsın.
y_train_log = np.log1p(y_train)
# Modeli Log'lu hedef ile eğitiyoruz
gbm_model.fit(X_train, y_train_log)
# Tahmin (Model log scale'de tahmin üretecek)
y_pred_log = gbm_model.predict(X_test)
# Tahminleri Tersine Çevirme (Inverse Transform) sonra büyütüyoruz tekraradan
y_pred_inverse = np.expm1(y_pred_log)

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_inverse))
print(f"Log Dönüşümlü Model RMSE: {rmse_log:.4f}") #rmse=23496.6207


#Adım_3: HİPERPARAMETRE OPTİMİZASYONU
from sklearn.model_selection import GridSearchCV, cross_validate
gbm_model = GradientBoostingRegressor(random_state=42)
gbm_model.get_params()
scoring_metrics = {
    'RMSE': 'neg_root_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'R2': 'r2'
}
cv_results = cross_validate(gbm_model, X, y, cv=10, scoring=scoring_metrics)
print(f"Ortalama RMSE: {-cv_results['test_RMSE'].mean():.4f}")
print(f"Ortalama MAE: {-cv_results['test_MAE'].mean():.4f}")
print(f"Ortalama R2 Skoru: {cv_results['test_R2'].mean():.4f}")

gbm_params = {
    "learning_rate": [0.01, 0.1],# learning_rate: Model ne kadar hızlı öğrensin?
    "max_depth": [3, 5],# max_depth: Ağaçlar ne kadar derin olsun?
    "n_estimators": [500, 1000, 3000], # n_estimators: Kaç tane ağaç dikilsin?
    "subsample": [1, 0.6, 0.5, 0.4] # Her ağaçta verinin kaçta kaçını kullansın?
}

#optimizasyon 5 katlı
gbm_cv = GridSearchCV(gbm_model,
                      gbm_params,
                      cv=5,
                      n_jobs=-1,
                      verbose=False).fit(X_train, y_train_log)

gbm_cv.best_params_

#modeli best params a göre kurma
gbm_final = GradientBoostingRegressor(**gbm_cv.best_params_, random_state=42).fit(X_train, y_train_log)
y_pred_final_log = gbm_final.predict(X_test)
y_pred_final_inverse = np.expm1(y_pred_final_log) # Tersine çevir
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final_inverse))
# final_rmse =23067.3344
#FİNAL RMSE : 22071.9883
print(f"\n FİNAL (Ayarlanmış) Model RMSE: {rmse_final:.4f}")

if rmse_final < 23496:
    print("SONUÇ: Tuning işlemi başarıyla hatayı düşürdü!")
else:
    print("SONUÇ: Tuning belirgin bir fark yaratmadı (Daha geniş aralık denenebilir).")



# Adım_4 DEĞİŞKEN ÖNEM DÜZEYİ

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False)[0:10])

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

plot_importance(gbm_final, X_train, num=15)


# SON AŞAMA TAHMİNLEME VE EXPORT İŞLEMİ
# modeli seçtik
model_to_use = gbm_final

X_test_final = test_df.drop(["Id", "SalePrice"], axis=1, errors="ignore")
X_test_final = X_test_final[X_train.columns]

#tahmin
y_pred_log = model_to_use.predict(X_test_final)
#log u geri al
y_pred_dollar = np.expm1(y_pred_log)
#Kaggle için df oluşturma
submission_df = pd.DataFrame({
    "Id": test["Id"].astype(int), # Test setinin orijinal ID'leri
    "SalePrice": y_pred_dollar
})

#head atalım
print(submission_df.head())

# CSV Dosyasını Kaydetme
file_name = "submisson/submission_final_kadir.csv"
submission_df.to_csv(file_name, index=False)

# Pickle formatında kaydetme
submission_df.to_pickle("tahminler.pkl")

# MODEL BAŞARI YÜZDESİ HESAPLAMA


# ======================================================
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# GERÇEK DEĞERLERİ KONTROL ET VE HAZIRLA
# Eğer y_val'in maksimum değeri 100'den büyükse, zaten Dolar'dır (Log değildir).
if y_test.max() > 100:
    y_gercek = y_test
else:
    y_gercek = np.expm1(y_test)

#  TAHMİNLERİ KONTROL ET VE HAZIRLA
# Model tahminlerini alalım
y_tahmin_raw = gbm_final.predict(X_test)

# Tahminlerin durumuna bakıyoruz (Log mu, Dolar mı?)
if y_tahmin_raw.max() > 100:
    y_tahmin = y_tahmin_raw
else:
    y_tahmin_clipped = np.clip(y_tahmin_raw, 0, 15)
    y_tahmin = np.expm1(y_tahmin_clipped)

#  METRİKLERİ HESAPLA
rmse = np.sqrt(mean_squared_error(y_gercek, y_tahmin))
mean_price = y_gercek.mean()

#  YÜZDELİK BAŞARI ORANLARI
hata_payi = (rmse / mean_price) * 100
model_başari_orani = 100 - hata_payi
r2 = r2_score(y_gercek, y_tahmin)

# SONUC
print(f"\n--- FİNAL ÖZET ---")
print(f"Ortalama Ev Fiyatı       : {mean_price:,.2f} $")
print(f"Modelin Ortalama Hatası  : {rmse:,.2f} $")
print(f"Hata Payı (Yüzde)        : %{hata_payi:.2f}")
print(f"----------------------------------")
print(f"MODEL BAŞARI ORANI       : %{model_başari_orani:.2f}")
print(f"R2        : %{r2*100:.2f}")
print(f"----------------------------------")
