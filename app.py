import streamlit as st # Import library Streamlit
import pandas as pd    # Untuk DataFrame
import numpy as np     # Untuk operasi numerik
import joblib          # Untuk memuat model dan scaler

# --- Memuat Semua Model dan Scaler yang Sudah Disimpan ---
# Pastikan path file benar. Jika file berada di sub-folder, sesuaikan path-nya.
try:
    # Memuat Scaler (hanya butuh satu scaler untuk semua model)
    scaler = joblib.load('scaler.joblib')
    st.success("Scaler berhasil dimuat!")

    # Memuat masing-masing model
    knn_model = joblib.load('knn_model.joblib')
    dtree_model = joblib.load('dtree_model.joblib')
    gnb_model = joblib.load('gnb_model.joblib')
    st.success("Semua model (KNN, Decision Tree, Gaussian Naive Bayes) berhasil dimuat!")

    # Menyimpan model dalam dictionary agar mudah diakses
    models = {
        "K-Nearest Neighbors": knn_model,
        "Decision Tree": dtree_model,
        "Gaussian Naive Bayes": gnb_model
    }

except FileNotFoundError:
    st.error("Error: Salah satu atau lebih file model/scaler tidak ditemukan.")
    st.info("Pastikan file `scaler.joblib`, `knn_model.joblib`, `dtree_model.joblib`, dan `gnb_model.joblib` berada di direktori yang sama dengan `app.py`.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")
    st.stop()

# --- Fungsi untuk Melakukan Pre-processing pada Input Pengguna ---
# Fungsi ini harus MEREPLIKASI LANGKAH-LANGKAH PRE-PROCESSING yang sama seperti saat melatih model.
# Ini termasuk penanganan outlier (capping) dan one-hot encoding jika diperlukan.

def preprocess_input(input_df):
    df_processed = input_df.copy()

    # 1. Label Encoding untuk 'Sex'
    df_processed['Sex'] = df_processed['Sex'].map({'Male': 0, 'Female': 1})

    # 2. One-Hot Encoding untuk 'Embarked'
    df_processed['Embarked_Q'] = (df_processed['Embarked'] == 'Q').astype(bool)
    df_processed['Embarked_S'] = (df_processed['Embarked'] == 'S').astype(bool)
    df_processed = df_processed.drop('Embarked', axis=1)

    # 3. Penanganan Outlier (Capping menggunakan IQR)
    # Gunakan batas outlier yang konsisten dengan saat training
    # Ini harusnya diambil dari nilai Q1, Q3, IQR dari data training yang sudah dianalisis.
    # Untuk demo, kita akan menggunakan nilai yang di-hardcode (dari hasil analisis sebelumnya).
    fare_upper_bound = 65.65 # Dari df['Fare'].quantile(0.75) + 1.5 * IQR_Fare
    age_upper_bound = 64.37  # Dari df['Age'].quantile(0.75) + 1.5 * IQR_Age
    age_lower_bound = 2.50   # Dari df['Age'].quantile(0.25) - 1.5 * IQR_Age

    df_processed['Fare'] = np.where(df_processed['Fare'] > fare_upper_bound, fare_upper_bound, df_processed['Fare'])
    df_processed['Age'] = np.where(df_processed['Age'] > age_upper_bound, age_upper_bound, df_processed['Age'])
    df_processed['Age'] = np.where(df_processed['Age'] < age_lower_bound, age_lower_bound, df_processed['Age'])

    # 4. Urutan Kolom (Sangat Penting!)
    # Pastikan urutan kolom sama dengan X.columns saat melatih model
    # X.columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    df_processed = df_processed[feature_cols]

    # 5. Scaling Fitur Numerik
    scaled_data = scaler.transform(df_processed)
    return scaled_data

# --- Tampilan Aplikasi Streamlit ---
st.set_page_config(page_title="Prediksi Survival Titanic (Multi-Model)", layout="centered")

st.title("🚢 Prediksi Survival Penumpang Titanic")
st.markdown("Pilih model Machine Learning dan masukkan data penumpang untuk memprediksi survival.")

st.subheader("Masukkan Data Penumpang:")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3], format_func=lambda x: f"{x} - {'First' if x==1 else ('Second' if x==2 else 'Third')}")
    sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Umur", min_value=0.5, max_value=80.0, value=25.0, step=0.5)
    sibsp = st.slider("Jumlah Saudara/Pasangan di Kapal (SibSp)", 0, 8, 0)

with col2:
    parch = st.slider("Jumlah Orang Tua/Anak di Kapal (Parch)", 0, 6, 0)
    fare = st.number_input("Harga Tiket (Fare)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
    embarked = st.selectbox("Pelabuhan Keberangkatan", ["S", "C", "Q"], format_func=lambda x: f"{x} - {'Southampton' if x=='S' else ('Cherbourg' if x=='C' else 'Queenstown')}")

st.markdown("---")
st.subheader("Pilih Model untuk Prediksi:")
selected_model_name = st.selectbox("Pilih Model", list(models.keys()))

# Tombol Prediksi
if st.button("Prediksi Survival"):
    # Ambil model yang dipilih pengguna
    selected_model = models[selected_model_name]

    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    st.write("Data Input Mentah:")
    st.dataframe(input_data)

    # Lakukan pre-processing pada data input
    processed_input = preprocess_input(input_data)

    # Prediksi
    prediction = selected_model.predict(processed_input)
    prediction_proba = selected_model.predict_proba(processed_input)

    st.subheader(f"Hasil Prediksi ({selected_model_name}):")
    if prediction[0] == 1:
        st.success(f"**Penumpang Diprediksi SELAMAT!** 🎉")
        st.balloons()
    else:
        st.error(f"**Penumpang Diprediksi TIDAK SELAMAT.** 😢")

    st.write(f"Probabilitas Selamat: **{prediction_proba[0][1]*100:.2f}%**")
    st.write(f"Probabilitas Tidak Selamat: **{prediction_proba[0][0]*100:.2f}%**")

    st.markdown("---")
    st.caption("Akurasi model ini tergantung pada data pelatihan dan algoritma yang digunakan.")