import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Load model, vectorizer, dan judul lama ===
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
judul_lama = joblib.load("judul_lama.pkl")

# Vectorisasi semua judul lama
judul_lama_vec = vectorizer.transform(judul_lama)

# === Konfigurasi halaman ===
st.set_page_config(page_title="Deteksi Plagiarisme Judul Skripsi", layout="centered")
st.title("🔍 Deteksi Plagiarisme Judul Skripsi")
st.write("Masukkan judul skripsi yang ingin diperiksa:")

# === Input dari pengguna ===
judul_baru = st.text_input("✏️ Masukkan Judul Skripsi:")

# === Tombol cek ===
if st.button("🔎 Cek Plagiarisme"):
    if judul_baru.strip() == "":
        st.warning("⚠️ Judul tidak boleh kosong.")
    else:
        # Vectorisasi judul input
        judul_baru_vec = vectorizer.transform([judul_baru])

        # Prediksi label dengan model
        hasil = model.predict(judul_baru_vec)[0]

        # Hitung kemiripan dengan semua judul lama
        kemiripan = cosine_similarity(judul_baru_vec, judul_lama_vec)[0]
        index_terdekat = np.argmax(kemiripan)
        judul_terdekat = judul_lama[index_terdekat]
        persen = round(kemiripan[index_terdekat] * 100, 2)

        if hasil == 1:
            st.error("❌ Judul terdeteksi plagiarisme.")
            st.markdown(f"🔁 Mirip dengan: **{judul_terdekat}**")
            st.markdown(f"📊 Kemiripan: **{persen}%**")
        else:
            st.success("✅ Judul tidak mengandung plagiarisme.")
            st.markdown(f"📊 Kemiripan tertinggi dengan judul lama: **{persen}%**")
