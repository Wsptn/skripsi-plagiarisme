import streamlit as st
import joblib
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import io
import textwrap

# === CSS Styling ===
def set_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #ffffff;
            color: #000000;
        }

        .stApp {
            background-color: #ffffff;
            padding: 2rem;
        }

        h1, h2, h3, h4, h5, h6,
        label, .stTextInput label,
        .stMarkdown, .stExpanderContent, summary {
            color: #000000 !important;
        }

        .stTextInput input {
            border: 2px solid #0073e6;
            border-radius: 8px;
            padding: 10px;
            font-size: 1rem;
            color: #ffffff;
        }

        .stButton button {
            background-color: #0073e6;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            border: none;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }

        .stButton button:hover {
            background-color: #005bb5;
        }

        .stSpinner {
            color: #000000 !important;
        }

        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }

        .logo-container img {
            width: 100px;
            margin-top: 10px;
        }

        div.stAlert > div {
            color: #000000 !important;
        }

        .stAlert[data-baseweb="notification"][role="alert"] {
            background-color: #d1e7dd !important;
            border-left: 6px solid #0f5132 !important;
        }

        .stAlert[data-baseweb="notification"].stError {
            background-color: #f8d7da !important;
            border-left: 6px solid #842029 !important;
        }

        /* Responsive for mobile */
        @media only screen and (max-width: 768px) {
            .stApp {
                padding: 1rem;
            }

            h1 {
                font-size: 1.4rem !important;
            }

            .stTextInput input,
            .stButton button {
                font-size: 0.95rem !important;
            }

            .logo-container img {
                width: 70px;
            }
        }

        /* Footer kanan bawah */
        .footer {
            position: fixed;
            bottom: 10px;
            right: 15px;
            font-size: 0.8rem;
            color: #666666;
            font-style: italic;
        }

        @media (max-width: 768px) {
            .footer {
                font-size: 0.7rem;
                text-align: center;
                left: 0;
                right: 0;
                bottom: 5px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

# === Setup Halaman
st.set_page_config(page_title="Deteksi Plagiarisme Judul Skripsi", layout="centered")
set_custom_css()

# === Logo
st.markdown("""
<div class="logo-container">
    <img src="https://raw.githubusercontent.com/Wsptn/skripsi-plagiarisme/main/Logo-UNUJA.png">
</div>
""", unsafe_allow_html=True)

st.title("🔍 Deteksi Plagiarisme Judul Skripsi")
st.markdown("Masukkan judul skripsi yang ingin diperiksa:")

# === Load model dan data
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
judul_lama = joblib.load("judul_lama.pkl")
judul_lama_vec = vectorizer.transform(judul_lama)

# === Input pengguna
judul_baru = st.text_input("✏️ Masukkan Judul Skripsi:")

# === Fungsi PDF
def buat_pdf(judul_baru, hasil, top_judul, top_persen):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    try:
        logo = ImageReader("Logo-UNUJA.png")
        c.drawImage(logo, 50, height - 100, width=60, height=60, mask='auto')
    except:
        pass

    c.setFont("Helvetica-Bold", 18)
    c.drawString(130, height - 65, "UNUJA")
    c.setFont("Helvetica", 12)
    c.drawString(130, height - 85, "Universitas Nurul Jadid")

    y = height - 140
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, "LAPORAN DETEKSI PLAGIARISME JUDUL SKRIPSI")
    y -= 40

    # Judul Dicek
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Judul yang Dicek:")
    c.setFont("Helvetica-Bold", 12)
    wrapped_input = textwrap.wrap(judul_baru, 80)
    for line in wrapped_input:
        c.drawString(180, y, line)
        y -= 15
    y -= 20

    # Hasil Prediksi (sejajar)
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Hasil Prediksi:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(180, y, "Plagiat" if hasil else "Tidak Plagiat")
    y -= 55

    # Tabel Kemiripan
    wrapped_judul = []
    for j in top_judul:
        lines = textwrap.wrap(j, 60)
        wrapped_judul.append("\n".join(lines))

    data = [["No", "Judul Lama", "Kemiripan (%)"]]
    for i in range(10):
        data.append([str(i + 1), wrapped_judul[i], f"{top_persen[i]}%"])

    table = Table(data, colWidths=[30, 370, 100])
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
    table.drawOn(c, 50, y - (len(data) * 28))

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 30, "Developed by: Muhammad Babun Waseptian")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# === Proses Cek
# ... (bagian atas tidak diubah)

# === Proses Cek
if st.button("🔎 Cek Plagiarisme"):
    if judul_baru.strip() == "":
        st.warning("⚠️ Judul tidak boleh kosong.")
    else:
        with st.spinner("⏳ Sedang menganalisis judul skripsi... Mohon tunggu sebentar"):
            time.sleep(1.5)

            judul_vec = vectorizer.transform([judul_baru])
            kemiripan = cosine_similarity(judul_vec, judul_lama_vec)[0]
            top_idx = np.argsort(kemiripan)[-10:][::-1]
            top_judul = [judul_lama[i] for i in top_idx]
            top_persen = [round(kemiripan[i]*100, 2) for i in top_idx]

            hasil_model = model.predict(judul_vec)[0]
            hasil = hasil_model == 1 or top_persen[0] >= 40

            if hasil:
                st.error("❌ Judul terdeteksi plagiarisme.")
            else:
                st.success("✅ Judul tidak mengandung plagiarisme.")

            st.markdown("### 📊 10 Judul Lama Paling Mirip")
            st.dataframe({
                "Judul Lama": top_judul,
                "Kemiripan (%)": top_persen
            }, use_container_width=True)

            with st.expander("📄 Detail Kemiripan"):
                for i in range(10):
                    st.markdown(f"**{i+1}. {top_judul[i]}**")
                    st.markdown(f"""
                        <div style="position: relative; background-color: #e0e0e0; border-radius: 10px; height: 24px; margin-bottom: 16px;">
                            <div style="width: {top_persen[i]}%; background-color: #0073e6; height: 100%; border-radius: 10px;"></div>
                            <div style="position: absolute; top: 0; right: 10px; height: 100%; display: flex; align-items: center; font-size: 13px; font-weight: bold; color: black;">
                                {top_persen[i]}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            pdf_file = buat_pdf(judul_baru, hasil, top_judul, top_persen)
            st.download_button(
                label="📥 Download Hasil PDF",
                data=pdf_file,
                file_name="hasil_deteksi_plagiarisme.pdf",
                mime="application/pdf"
            )

# === FOOTER
st.markdown("""
    <div class="footer">
        Project by: Muhammad Babun Waseptian
    </div>
""", unsafe_allow_html=True)
