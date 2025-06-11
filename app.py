import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import io
import textwrap

# === Load model dan data ===
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
judul_lama = joblib.load("judul_lama.pkl")
judul_lama_vec = vectorizer.transform(judul_lama)

# === Konfigurasi Streamlit ===
st.set_page_config(page_title="Deteksi Plagiarisme Judul Skripsi", layout="centered")
st.title("🔍 Deteksi Plagiarisme Judul Skripsi")
st.markdown("Masukkan judul skripsi yang ingin diperiksa:")

# === Input pengguna ===
judul_baru = st.text_input("✏️ Masukkan Judul Skripsi:")

# === Fungsi PDF dengan header & wrap ===
def buat_pdf(judul_baru, hasil, top3_judul, top3_persen):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # Logo dan Header
    try:
        logo = ImageReader("Logo-UNUJA.png")
        c.drawImage(logo, 50, height - 100, width=60, height=60, mask='auto')
    except:
        pass

    c.setFont("Helvetica-Bold", 18)
    c.drawString(130, height - 65, "UNUJA")
    c.setFont("Helvetica", 12)
    c.drawString(130, height - 85, "Universitas Nurul Jadid")

    # Judul dokumen
    y = height - 140
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, "LAPORAN DETEKSI PLAGIARISME JUDUL SKRIPSI")
    y -= 40

    # Informasi Judul Dicek
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Judul yang Dicek:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(180, y, judul_baru[:80])
    y -= 25

    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Hasil Prediksi:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(180, y, "Plagiat" if hasil == 1 else "Tidak Plagiat")
    y -= 40

    # Bungkus teks panjang agar tabel tidak melebar
    wrapped_judul = []
    for j in top3_judul:
        lines = textwrap.wrap(j, 60)
        wrapped_judul.append("\n".join(lines))

    data = [["No", "Judul Lama", "Kemiripan (%)"]]
    for i in range(3):
        data.append([str(i + 1), wrapped_judul[i], f"{top3_persen[i]}%"])

    table = Table(data, colWidths=[30, 380, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f0f8ff")),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    table.wrapOn(c, width, height)
    table.drawOn(c, 50, y - (len(data) * 30))

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 30, "Developed by: Muhammad Babun Waseptian")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# === Logika utama aplikasi ===
if st.button("🔎 Cek Plagiarisme"):
    if judul_baru.strip() == "":
        st.warning("⚠️ Judul tidak boleh kosong.")
    else:
        judul_vec = vectorizer.transform([judul_baru])
        hasil = model.predict(judul_vec)[0]
        kemiripan = cosine_similarity(judul_vec, judul_lama_vec)[0]
        top3_idx = np.argsort(kemiripan)[-3:][::-1]
        top3_judul = [judul_lama[i] for i in top3_idx]
        top3_persen = [round(kemiripan[i]*100, 2) for i in top3_idx]

        # Hasil deteksi
        if hasil == 1:
            st.error("❌ Judul terdeteksi plagiarisme.")
        else:
            st.success("✅ Judul tidak mengandung plagiarisme.")

        # Tabel hasil di web
        st.markdown("### 📊 Tabel Kemiripan Judul")
        st.dataframe({
            "Judul Lama": top3_judul,
            "Kemiripan (%)": top3_persen
        }, use_container_width=True)

        # Detail dengan progress bar
        with st.expander("📄 Lihat Detail Kemiripan"):
            for i in range(3):
                st.markdown(f"**{i+1}. {top3_judul[i]}**")
                st.progress(int(top3_persen[i]))

        # Tombol Download PDF
        pdf_file = buat_pdf(judul_baru, hasil, top3_judul, top3_persen)
        st.download_button(
            label="📥 Download Hasil PDF",
            data=pdf_file,
            file_name="hasil_deteksi_plagiarisme.pdf",
            mime="application/pdf"
        )
