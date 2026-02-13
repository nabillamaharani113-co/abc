import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import sys
import os

# Define paths to saved model and scaler
# IMPORTANT: Adjust these paths if your app.py is not in the same directory as your .pkl files
# For local deployment, you might place .pkl files in the same directory as app.py
# or specify an absolute path.
filename_model = 'model_random_forest.pkl'
filename_scaler = 'scaler_gaji.pkl'

# Helper function to safely display Streamlit messages or fall back to print
def display_ui_message(message, is_error=False):
    # Check if streamlit is loaded and an active session exists
    if 'streamlit' in sys.modules and hasattr(st, 'runtime') and st.runtime.exists():
        if is_error:
            st.error(message)
        else:
            st.success(message)
    else:
        # Fallback to print if not in Streamlit app
        if is_error:
            print(f"ERROR: {message}")
        else:
            print(f"SUCCESS: {message}")

def stop_application():
    if 'streamlit' in sys.modules and hasattr(st, 'runtime') and st.runtime.exists():
        st.stop()
    else:
        # If not running as a Streamlit app, exit the script
        sys.exit(1)

# Load the Random Forest model
try:
    with open(filename_model, 'rb') as file:
        loaded_model = pickle.load(file)
    display_ui_message(f"Random Forest model loaded successfully from {filename_model}")
except FileNotFoundError:
    display_ui_message(f"Error: Model file not found at {filename_model}. Please ensure the file exists.", is_error=True)
    stop_application()
except Exception as e:
    display_ui_message(f"Error loading model: {e}", is_error=True)
    stop_application()

# Load the StandardScaler object
try:
    with open(filename_scaler, 'rb') as file:
        loaded_scaler = pickle.load(file)
    display_ui_message(f"Scaler object loaded successfully from {filename_scaler}")
except FileNotFoundError:
    display_ui_message(f"Error: Scaler file not found at {filename_scaler}. Please ensure the file exists.", is_error=True)
    stop_application()
except Exception as e:
    display_ui_message(f"Error loading scaler: {e}", is_error=True)
    stop_application()

# --- Preprocessing components start ---
# Hardcoded unique values for categorical features (based on the training data)
unique_pendidikan = ['SMA', 'SMK', 'D3', 'S1']
unique_jurusan = ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif']
unique_jenis_kelamin = ['Laki-laki', 'Perempuan']
unique_status_bekerja = ['Sudah Bekerja', 'Belum Bekerja']

# Initialize and fit LabelEncoders with the hardcoded unique values
le_pendidikan = LabelEncoder()
le_pendidikan.fit(unique_pendidikan)

le_jurusan = LabelEncoder()
le_jurusan.fit(unique_jurusan)

label_encoders = {
    'Pendidikan': le_pendidikan,
    'Jurusan': le_jurusan
}

# Define the exact order of feature columns that the trained model expects
feature_cols_order = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                      'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Perempuan',
                      'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

display_ui_message("Persiapan preprocessing selesai (tanpa file CSV).")

def preprocess_input(user_input, label_encoders, loaded_scaler, feature_cols_order):
    input_df = pd.DataFrame([user_input])

    # Ensure consistent gender mapping (although unique_jenis_kelamin already clean)
    mapping_gender = {'Pria':'Laki-laki', 'L':'Laki-laki', 'laki-laki':'Laki-laki', 'perempuan':'Perempuan', 'Wanita':'Perempuan', 'wanita':'Perempuan', "P":'Perempuan'}
    input_df['Jenis_Kelamin'] = input_df['Jenis_Kelamin'].replace(mapping_gender)

    processed_features = pd.DataFrame(0, index=[0], columns=feature_cols_order)

    # Populate numerical features
    processed_features['Usia'] = input_df['Usia'].values[0]
    processed_features['Durasi_Jam'] = input_df['Durasi_Jam'].values[0]
    processed_features['Nilai_Ujian'] = input_df['Nilai_Ujian'].values[0]

    # Apply Label Encoding
    processed_features['Pendidikan'] = label_encoders['Pendidikan'].transform([input_df['Pendidikan'].values[0]])[0]
    processed_features['Jurusan'] = label_encoders['Jurusan'].transform([input_df['Jurusan'].values[0]])[0]

    # Apply One-Hot Encoding for Jenis_Kelamin
    if input_df['Jenis_Kelamin'].values[0] == 'Laki-laki':
        processed_features['Jenis_Kelamin_Laki-laki'] = 1
        processed_features['Jenis_Kelamin_Perempuan'] = 0
    else:
        processed_features['Jenis_Kelamin_Laki-laki'] = 0
        processed_features['Jenis_Kelamin_Perempuan'] = 1

    # Apply One-Hot Encoding for Status_Bekerja
    if input_df['Status_Bekerja'].values[0] == 'Belum Bekerja':
        processed_features['Status_Bekerja_Belum Bekerja'] = 1
        processed_features['Status_Bekerja_Sudah Bekerja'] = 0
    else:
        processed_features['Status_Bekerja_Belum Bekerja'] = 0
        processed_features['Status_Bekerja_Sudah Bekerja'] = 1

    # Ensure the order of columns is correct before scaling
    processed_features = processed_features[feature_cols_order]

    # Scale the features
    scaled_features = loaded_scaler.transform(processed_features)

    return scaled_features

display_ui_message("Fungsi `preprocess_input` telah didefinisikan.")

# --- UI elements start ---
st.title('Prediksi Gaji Pertama Peserta Pelatihan Vokasi')
st.write('Aplikasi ini memprediksi gaji pertama peserta pelatihan vokasi berdasarkan profil mereka.')

usia = st.number_input('Usia', min_value=18, max_value=60, value=30)
durasi_jam = st.number_input('Durasi Pelatihan (Jam)', min_value=20, max_value=100, value=60)
nilai_ujian = st.number_input('Nilai Ujian', min_value=50.0, max_value=100.0, value=75.0, step=0.1)

jenis_kelamin = st.radio('Jenis Kelamin', unique_jenis_kelamin)

pendidikan = st.selectbox('Pendidikan', unique_pendidikan)
jurusan = st.selectbox('Jurusan', unique_jurusan)
status_bekerja = st.radio('Status Bekerja', unique_status_bekerja)

predict_button = st.button('Prediksi Gaji')

# --- Prediction logic start ---
if predict_button:
    user_input = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    preprocessed_input_data = preprocess_input(
        user_input,
        label_encoders,
        loaded_scaler,
        feature_cols_order
    )

    preprocessed_df_for_prediction = pd.DataFrame(preprocessed_input_data, columns=feature_cols_order)

    predicted_salary = loaded_model.predict(preprocessed_df_for_prediction)
    st.success(f'Prediksi Gaji Pertama: {predicted_salary[0]:.2f} Juta Rupiah')
