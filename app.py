import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import os
import time

# -------------------- Konfigurasi Halaman --------------------
st.set_page_config(page_title="Lokatmala - Deteksi Motif Batik", layout="wide")

# -------------------- Fungsi CSS Kustom --------------------
def add_custom_css():
    css = """
    <style>
        /* Background utama aplikasi */
        .stApp {
            background: linear-gradient(to bottom, #990000, #4d0000);
            padding: 2rem;
            overflow-x: hidden !important;  /* cegah scroll horizontal */
        }

        /* Container isi utama */
        .block-container {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0px 0px 25px rgba(0, 0, 0, 0.4);
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Warna teks judul dan radio label */
        h1, h2, h3, .stRadio > label {
            color: #330000;
        }

        /* Tombol */
        .stButton > button {
            background-color: #990000;
            color: #ffffff;
            font-weight: bold;
            padding: 10px 16px;
            border-radius: 8px;
        }

        .stButton > button:hover {
            background-color: #cc0000;
        }

        /* Responsive untuk mobile */
        @media only screen and (max-width: 768px) {
            h1 { font-size: 1.8em !important; }
            h2 { font-size: 1.4em !important; }
            p, li, div { font-size: 1em !important; }
        }
        /* Copyright footer */
        .footer {
            position: relative;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding-top: 2rem;
            font-size: 0.9rem;
            color: #990000;
            margin-top: 3rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------- Fungsi Tambah Logo --------------------
def add_logo_base64(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = f.read()
    logo_base64 = base64.b64encode(logo_data).decode()

    logo_html = f"""
    <div style="position: absolute; top: 15px; left: 20px; z-index: 999;">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 60px;">
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)

# -------------------- Fungsi Banner Gambar Shadow --------------------
def add_shadow_banner(image_path):
    with open(image_path, "rb") as f:
        bg_data = f.read()
    bg_base64 = base64.b64encode(bg_data).decode()

    banner_html = f"""
    <style>
    .shadow-header {{
        position: relative;
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        border-radius: 16px;
        padding: 60px 30px 40px 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        color: white;
        text-align: center;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
    }}
    </style>

    <div class="shadow-header">
        <h1>CARITALOKA</h1>
        <p style="font-size: 1.1em;">Identifikasi Motif dan Filosofi Kain Batik Lokatmala</p>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

# -------------------- Panggilan Semua Komponen --------------------
add_custom_css()
add_logo_base64("logo lokatmala.png")     # ← pastikan nama & lokasi file sesuai
add_shadow_banner("bg3.png")       # ← pastikan file ini ada

# -------------------- Load Model Sekali (Cache) --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Lokatmala_saved")  # <-- Ganti ke folder SavedModel

model = load_model()

# Label kelas (urutan harus sama dengan model)
class_names = [
    "Candramawat", "Elang Jawa Situ Gunung", "Garuda Ngupuk", "Jantung Kole", "Leungli",
    "Makara", "Mandala Bagja", "Manuk Julang", "Masagi", "Mata Air Sukabumi",
    "Merak Kinanti", "Mozaik Kadudampit", "Nakamesta", "Pakwan", "Palawan",
    "Penyu Sukabumian", "Puyuh", "Rereng Gunung Parang", "Rereng Tjaiwangi", "Wijayakusumah"
]

# Dictionary filosofi (tidak berubah)
filosofi_dict = {
    # ... (isi filosofi sama seperti sebelumnya)
}

# -------------------- Layout Dua Kolom --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Gambar")

    input_method = st.radio("Pilih Metode:", ["Upload Gambar", "Ambil dari Kamera"])

    image = None
    if input_method == "Upload Gambar":
        uploaded_file = st.file_uploader("Unggah gambar batik (.png/ lokatmala.png)", type=["png", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input("Ambil gambar dari kamera")
        if camera_image:
            image = Image.open(camera_image)

    if image:
        st.image(image, caption="Gambar telah di identifikasi", use_container_width=True)

with col2:
    st.subheader("Hasil Prediksi")

    if image:
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        filosofi = filosofi_dict.get(predicted_class, "Filosofi tidak ditemukan.")

        st.markdown(f"<div style='font-size: 1.2em;'><strong>Motif Terdeteksi:</strong> {predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1em; color: gray;'><strong>Tingkat Keyakinan:</strong> {confidence:.2f}%</div>", unsafe_allow_html=True)

        if confidence < 70:
            st.warning(
                "⚠️ Keyakinan rendah. Silakan coba ulang dengan gambar yang lebih baik, dengan memperhatikan spesifikasi sebagai berikut:\n"
                "- Gunakan gambar batik yang fokus dan jelas.\n"
                "- Hindari kain terlipat atau kusut.\n"
                "- Jangan gunakan latar belakang yang ramai.\n"
                "- Ambil gambar dari jarak sedang, tidak terlalu dekat atau jauh.\n"
                "- Pastikan pencahayaan cukup dan merata."
            )
        st.markdown("<hr style='margin-top: 20px; margin-bottom: 10px;'>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: justify; font-size: 0.95em; line-height: 1.7;'><strong>Filosofi Motif:</strong><br>{filosofi}</div>", unsafe_allow_html=True)
    else:
        st.info("Silakan unggah atau ambil gambar terlebih dahulu.")

# -------------------- Copyright Footer --------------------
st.markdown(
    """
    <div class="footer">
        © 2025 Batik Lokatmala 
    </div>
    """,
    unsafe_allow_html=True
)
