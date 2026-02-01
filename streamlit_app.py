import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Load dataset (cache data)
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("coba-data.csv")   # ganti dengan dataset UKT kamu
    return df

df = load_data()

# -----------------------------------------
# 2. Preprocessing + Model (cache resource)
# -----------------------------------------
important_cols = [
    'penghasilan_ayah','penghasilan_ibu','daya_listrik',
    'jumlah_tanggungan','luas_tanah','jarak_pusat_kota',
    'bahan_tembok','bahan_lantai','id_pekerjaan_ayah',
    'id_pekerjaan_ibu','jenjang','prodi','sumber_listrik'
]
important_cols = [c for c in important_cols if c in df.columns]
X = df[important_cols].copy()

# Parsing luas_tanah (string → angka)
def parse_area(val):
    if isinstance(val, str):
        val = val.replace("m2", "").replace(" ", "")
        if "-" in val:
            parts = val.split("-")
            try:
                nums = [float(p) for p in parts if p]
                if len(nums) == 2:
                    return sum(nums)/2
            except:
                return np.nan
        try:
            return float(val)
        except:
            return np.nan
    return val

if "luas_tanah" in X.columns:
    X["luas_tanah"] = X["luas_tanah"].apply(parse_area)

X = X.apply(pd.to_numeric, errors="coerce")

@st.cache_resource
def build_model(X):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=min(3, X.shape[1]))

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_pca = pca.fit_transform(X_scaled)

        # Pakai 2 cluster saja
    final_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    final_kmeans.fit(X_pca)


    return imputer, scaler, pca, final_kmeans, X_pca

imputer, scaler, pca, final_kmeans, X_pca = build_model(X)

# -----------------------------------------
# 3. UI Judul & Deskripsi
# -----------------------------------------
st.markdown("<h1 style='text-align:center;color:#2E86C1;'>📊 Prediksi Kategori UKT Mahasiswa</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Aplikasi ini menggunakan clustering K-Means dengan EDA, PCA untuk memprediksi kategori UKT.</p>", unsafe_allow_html=True)


# -----------------------------------------
# 4. Form Input User
# -----------------------------------------
st.subheader("Masukkan Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    penghasilan_ayah = st.number_input("Penghasilan Ayah (Rp)", min_value=0.0, step=100000.0)
    daya_listrik = st.number_input("Daya Listrik (VA)", min_value=0, step=100)
    jumlah_tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, step=1)
    luas_tanah = st.number_input("Luas Tanah (m2)", min_value=0, step=1)
    bahan_tembok = st.selectbox("Bahan Tembok", ["Bata","Kayu","Beton","Lainnya"])
    bahan_lantai = st.selectbox("Bahan Lantai", ["Keramik","Kayu","Semen","Lainnya"])

with col2:
    penghasilan_ibu = st.number_input("Penghasilan Ibu (Rp)", min_value=0.0, step=100000.0)
    jarak_pusat_kota = st.number_input("Jarak ke Pusat Kota (km)", min_value=0, step=1)
    jenjang = st.selectbox("Jenjang", ["Diploma","Sarjana","Magister"])
    prodi = st.number_input("Prodi (ID)", min_value=0, step=1)
    id_pekerjaan_ayah = st.number_input("ID Pekerjaan Ayah", min_value=0, step=1)
    id_pekerjaan_ibu = st.number_input("ID Pekerjaan Ibu", min_value=0, step=1)
    sumber_listrik = st.selectbox("Sumber Listrik", ["PLN","Genset","Lainnya"])

# Mapping kategori ke angka
map_bahan_tembok = {"Bata":0,"Kayu":1,"Beton":2,"Lainnya":3}
map_bahan_lantai = {"Keramik":0,"Kayu":1,"Semen":2,"Lainnya":3}
map_jenjang = {"Diploma":1,"Sarjana":2,"Magister":3}
map_sumber_listrik = {"PLN":0,"Genset":1,"Lainnya":2}

user_input = {
    "penghasilan_ayah": penghasilan_ayah,
    "penghasilan_ibu": penghasilan_ibu,
    "daya_listrik": daya_listrik,
    "jumlah_tanggungan": jumlah_tanggungan,
    "luas_tanah": luas_tanah,
    "jarak_pusat_kota": jarak_pusat_kota,
    "bahan_tembok": map_bahan_tembok[bahan_tembok],
    "bahan_lantai": map_bahan_lantai[bahan_lantai],
    "id_pekerjaan_ayah": id_pekerjaan_ayah,
    "id_pekerjaan_ibu": id_pekerjaan_ibu,
    "jenjang": map_jenjang[jenjang],
    "prodi": prodi,
    "sumber_listrik": map_sumber_listrik[sumber_listrik]
}

# -----------------------------------------
# 5. Prediksi
# -----------------------------------------
if st.button("Prediksi UKT"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=X.columns)
    input_df = input_df.apply(pd.to_numeric, errors="coerce")

    if input_df.isnull().values.any():
        st.error("⚠️ Data input belum lengkap atau tidak valid.")
    else:
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        input_pca = pca.transform(input_scaled)

        cluster_pred = int(final_kmeans.predict(input_pca)[0])

        kategori_ukt_map = {
        0: "UKT Rendah",
        1: "UKT Tinggi"
        }

      
        st.success(f"Kategori UKT: {kategori_ukt_map.get(cluster_pred, 'Perlu interpretasi lanjutan')}")

      



