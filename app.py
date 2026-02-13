import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Gaji Peserta Vokasi", layout="centered")

# --- FUNGSI LOAD DATA & MODEL (Cache agar cepat) ---
@st.cache_resource
def load_assets():
    with open('model_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_gaji.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def get_label_encoders():
    # Load dataset untuk fitting ulang LabelEncoder agar konsisten dengan training
    df_raw = pd.read_csv('dataset_pelatihan_vokasi_dirty.csv')
    
    # Cleaning singkat (sama dengan logika training Anda)
    mapping_gender = {'Pria':'Laki-laki', 'L':'Laki-laki', 'laki-laki':'Laki-laki', 
                      'perempuan':'Perempuan', 'Wanita':'Perempuan', 'wanita':'Perempuan', "P":'Perempuan'}
    df_raw['Jenis_Kelamin'] = df_raw['Jenis_Kelamin'].replace(mapping_gender)
    df_raw['Jurusan'] = df_raw['Jurusan'].fillna(df_raw['Jurusan'].mode()[0])
    
    le_pendidikan = LabelEncoder().fit(df_raw['Pendidikan'])
    le_jurusan = LabelEncoder().fit(df_raw['Jurusan'])
    
    return le_pendidikan, le_jurusan, mapping_gender

# Load semua asset
try:
    loaded_model, loaded_scaler = load_assets()
    le_pendidikan, le_jurusan, mapping_gender = get_label_encoders()
except Exception as e:
    st.error(f"Error loading files: {e}. Pastikan file .pkl dan .csv ada di folder yang sama.")
    st.stop()

# --- UI STREAMLIT ---
st.title("💰 Prediksi Gaji Pertama")
st.write("Masukkan data peserta di bawah ini untuk melihat estimasi gaji pertama.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        usia = st.number_input("Usia", min_value=15, max_value=60, value=25)
        durasi = st.number_input("Durasi Pelatihan (Jam)", min_value=1, value=40)
        nilai = st.slider("Nilai Ujian", 0.0, 100.0, 85.0)
    
    with col2:
        pendidikan = st.selectbox("Pendidikan Terakhir", le_pendidikan.classes_)
        jurusan = st.selectbox("Jurusan", le_jurusan.classes_)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        status = st.selectbox("Status Bekerja", ["Belum Bekerja", "Sudah Bekerja"])

    submit = st.form_submit_button("Prediksi Sekarang")

# --- PROSES PREDIKSI ---
if submit:
    # 1. Siapkan DataFrame Input
    feature_cols_order = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                          'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Perempuan',
                          'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']
    
    processed_features = pd.DataFrame(0, index=[0], columns=feature_cols_order)

    # 2. Isi nilai numerik
    processed_features['Usia'] = usia
    processed_features['Durasi_Jam'] = durasi
    processed_features['Nilai_Ujian'] = nilai

    # 3. Apply Label Encoding
    processed_features['Pendidikan'] = le_pendidikan.transform([pendidikan])[0]
    processed_features['Jurusan'] = le_jurusan.transform([jurusan])[0]

    # 4. Apply One-Hot Encoding Manual
    if gender == 'Laki-laki':
        processed_features['Jenis_Kelamin_Laki-laki'] = 1
    else:
        processed_features['Jenis_Kelamin_Perempuan'] = 1

    if status == 'Belum Bekerja':
        processed_features['Status_Bekerja_Belum Bekerja'] = 1
    else:
        processed_features['Status_Bekerja_Sudah Bekerja'] = 1

    # 5. Scaling
    scaled_features = loaded_scaler.transform(processed_features)
    final_df = pd.DataFrame(scaled_features, columns=feature_cols_order)

    # 6. Predict
    prediction = loaded_model.predict(final_df)[0]

    # Tampilkan Hasil
    st.divider()
    st.subheader(f"Estimasi Gaji Pertama: ")
    st.header(f"Rp {prediction:.2f} Juta")
