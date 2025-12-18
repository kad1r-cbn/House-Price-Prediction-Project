import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sayfa Ayarları
st.set_page_config(page_title="Ev Fiyat Tahmin Modeli", layout="wide")

st.title("🏠 Ev Fiyat Tahmin Modeli ve Analizi")
st.markdown("Bu proje, ev fiyatlarını tahmin etmek için Gradient Boosting Regressor kullanır.")

# --- FONKSİYONLAR (CACHE MEKANİZMASI İLE) ---

@st.cache_data
def load_and_preprocess_data():
    # Veri Seti Okuma
    # NOT: GitHub'da 'veri_setleri' klasörünün ve içindeki csv'lerin olduğundan emin ol!
    try:
        test = pd.read_csv("test.csv")
        train = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error("CSV dosyaları bulunamadı. Lütfen 'veri_setleri/train.csv' ve 'test.csv' dosya yollarını kontrol et.")
        return None, None, None

    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 1. TİP DÖNÜŞÜMLERİ
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)

    # 2. AYKIRI DEĞER BASKILAMA (Fonksiyon içi)
    def replace_with_thresholds(dataframe, variable):
        q1 = 0.05
        q3 = 0.95
        quartile1 = dataframe[variable].quantile(q1)
        quartile3 = dataframe[variable].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        
        if dataframe[variable].dtype == 'int64':
            dataframe[variable] = dataframe[variable].astype(float)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in ["Id", "SalePrice"]]
    for col in num_cols:
        replace_with_thresholds(df, col)

    # 3. EKSİK DEĞER DOLDURMA
    none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                 "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                 "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]
    for col in none_cols:
        df[col] = df[col].fillna("No")

    zero_cols = ["GarageYrBlt", "GarageArea", "GarageCars",
                 "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]
    for col in zero_cols:
        df[col] = df[col].fillna(0)

    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    num_cols_all = [col for col in df.columns if df[col].dtypes != "O" and col != "SalePrice"]
    for col in num_cols_all:
         if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 4. RARE ENCODING
    def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()
        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
        for col in rare_columns:
            tmp = temp_df[col].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])
        return temp_df

    df = rare_encoder(df, 0.01)

    # 5. FEATURE ENGINEERING
    df["NEW_Total_House_Area"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    df["NEW_Total_Bath"] = df["FullBath"] + (df["HalfBath"] * 0.5) + df["BsmtFullBath"] + (df["BsmtHalfBath"] * 0.5)
    df["NEW_Total_Porch_SF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])
    df["NEW_House_Age"] = df["YrSold"] - df["YearBuilt"]
    df["NEW_Is_Renovated"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["NEW_Restoration_Age"] = df["YrSold"] - df["YearRemodAdd"]
    df["NEW_Total_Qual"] = df["OverallQual"] * df["OverallCond"]
    df["NEW_Has_Pool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    df["NEW_Has_Garage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)

    # 6. ENCODING
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
    labelencoder = LabelEncoder()
    for col in binary_cols:
        df[col] = labelencoder.fit_transform(df[col])

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2 and df[col].dtype == 'O']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    return df, train, test

# Veriyi Yükle
df, raw_train, raw_test = load_and_preprocess_data()

if df is not None:
    # --- MODELLEME ---
    train_df = df[df['SalePrice'].notnull()]
    test_df = df[df['SalePrice'].isnull()]

    y = train_df['SalePrice']
    X = train_df.drop(["Id", "SalePrice"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

    # Cache Resource: Model eğitimi ağır işlemdir, tekrar tekrar yapmasın.
    @st.cache_resource
    def train_model(X_train, y_train):
        # Log dönüşümü
        y_train_log = np.log1p(y_train)
        
        # Basitlik için gridsearch sonucundaki best parametreleri buraya elle giriyorum
        # Gerçek bir app'te GridSearch her seferinde çalıştırılmaz!
        # Senin kodundaki best_params mantığını simüle eden güçlü parametreler:
        params = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 500,
            "subsample": 0.6,
            "random_state": 42
        }
        
        gbm_final = GradientBoostingRegressor(**params)
        gbm_final.fit(X_train, y_train_log)
        return gbm_final

    st.subheader("⚙️ Model Eğitimi Durumu")
    with st.spinner('Model eğitiliyor, lütfen bekleyin...'):
        gbm_final = train_model(X_train, y_train)
    st.success("Model başarıyla eğitildi!")

    # --- TAHMİNLER VE METRİKLER ---
    y_pred_log = gbm_final.predict(X_test)
    y_pred_inverse = np.expm1(y_pred_log)
    
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_inverse))
    r2_final = r2_score(y_test, y_pred_inverse)

    # Metrikleri Göster
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE (Hata)", f"{rmse_final:,.2f} $")
    col2.metric("R2 Score", f"%{r2_final*100:.2f}")
    col3.metric("Test Verisi Sayısı", len(y_test))

    # --- GÖRSELLEŞTİRME ---
    st.subheader("📊 Değişken Önem Düzeyleri")
    
    feature_imp = pd.DataFrame({'Value': gbm_final.feature_importances_, 'Feature': X_train.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp, ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # --- KORELASYON ANALİZİ (Opsiyonel Buton) ---
    if st.checkbox("Korelasyon Matrisini Göster"):
        st.write("Sadece sayısal değişkenlerin korelasyonu:")
        # Orijinal veriden sayısal sütunları alıp gösterelim
        num_cols_raw = [col for col in raw_train.columns if raw_train[col].dtype != "O"]
        corr = raw_train[num_cols_raw].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="RdBu", ax=ax_corr)
        st.pyplot(fig_corr)

else:
    st.stop()

