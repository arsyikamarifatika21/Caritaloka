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
    return tf.keras.models.load_model("Lokatmala.h5")

model = load_model()


# Label kelas (urutan harus sama dengan model)
class_names = [
    "Candramawat", "Elang Jawa Situ Gunung", "Garuda Ngupuk", "Jantung Kole", "Leungli",
    "Makara", "Mandala Bagja", "Manuk Julang", "Masagi", "Mata Air Sukabumi",
    "Merak Kinanti", "Mozaik Kadudampit", "Nakamesta", "Pakwan", "Palawan",
    "Penyu Sukabumian", "Puyuh", "Rereng Gunung Parang", "Rereng Tjaiwangi", "Wijayakusumah"
]

# Dictionary filosofi
filosofi_dict = {
    "Candramawat": "Pola hias batik Cendramawat terinspirasi dari dongeng Sunda tentang Nini anteh, sebuah dongeng yang mengisahkan bercak hitam di permukaan bulan purnama. Dalam cerita ini, Nini anteh dikisahkan sebagai seorang nenek menenun kain sambil ditemani oleh seekor kucing Bernama Candramawat. Nama Candramawat sendiri diambil dari kata ‘Candra’ yang berarti ‘bulan’ dan ‘Mawat’ yang memiliki makna ‘keberuntungan’. Pola hias batik ini mengangkat nilai nilai filosofi tentang ketekunan, cinta kasih, dan harapan akan keberuntungan, menjadikannya symbol budaya yang kaya makna dari tradisi lisan Sunda. Selain itu, di daerah Cikananga, Kabupaten Sukabumi hingga kini masih terdapat konservasi macan dahan (Neofelis Diardi), spesies kucing besar atau nama lain dari kucing Candramawat.",
    "Elang Jawa Situ Gunung": "Pola hias batik Elang Jawa Situ Gunung terinspirasi dari burung Elang Jawa (Nisaetus artelsi), yang juga dikenal sebagai burung garuda. Pola hias batik ini juga menggali inspiras dari lambang negara Indonesia sebagai simbol kekuatan, perlindungan, dan kebijaksanaan. Populasi Elang Jawa, yang merupakan spesies endemik Indonesia, mash dapat ditemukan di kawasan konservasi Cimungkad, Situ Gunung, Kadudampit Kabupaten Sukabumi. Elang Jawa Situ Gunung tidak hanya merepresentasikan keindahan alam dan keanekaragaman hayati Indonesia, tetapi juga mengandung pesan mendalam tentang pelestarian alam serta kebanggaan terhadap kekayaan fauna endemik Indonesia.",
    "Garuda Ngupuk": "Pola hias batik Garuda Ngupuk terinspirasi dari konsep tata ruang pembangunan pusat pemerintangan yang ideal dalam budaya Sunda. Filosofi mendalam yang terkandung dalam pola hias batik ini mengajarkan bahwa setiap manusia perlu mrmiliki sumber kehidupan yang memadai, seperti keluasa ilmu pengetahuan, kemampuan beradaptasi secara dinamis dalam berbagai situasi, serta keteguhan hati untuk menghadapi segala tantangan. Dalam kepercayaan tradisional Masyarakat Sunda, lahan yang baik untuk pusat pemerintahan diibaratkan seperti “Garuda ngupuk, bahe ngaler-ngetan, deukeut pangguyangan badak putih”. Ungkapan ini menggambarkan bahwa Lokasi pusat kehidupan atau pusat pemerintahan harus strategis dan mendukung dari segala aspek, salah satunya dekat dengan sumber air yang melimpah. Garuda Ngupuk menjadi symbol perpaduan antara kearifan local, harapan akan kesejahteraan dan visi pembangunan yang berkelanjutan. ",
    "Jantung Kole": "Pola hias batik Jantung Kole terinspirasi dari bentuk tumbuhan pisang Kole (Musa Salaccensis), terutama bagian jantungnya yang berwarna merah keunguan. Pisang Kole adalah salah satu jenis pisang asli Jawa Barat yang kini tumbuh lar di lereng-lereng hutan atau di bawah naungan pohon, jenis pisang ini mash banyak dijumpai di dataran tinggi Taman Nasional Gunung Gede Pangrango, dengan ciri khas buahnya buah tegak, kecil berwarna keunguan dan memiliki tekstur kesat di lidah.Dalam legenda Pakujajar di Gunung Parang, tokoh Wangsa Suta mendapat perintah untuk mendirikan pemukiman di wilayah dengan cir-ciri khusus: permukaan tanah miring ke selatan, adanya pohon beringin kembar, tanaman paku berjajar dengan lima dahan, serta keberadaan pohon pisang kole atau pisang hutan, yang memiliki daun berwarna ungu. Kisah ini menambahkan makna historis dan filosofis pada pola hias batik Jantung Kole, yang merepresentasikan keunikan alam dan kekayaan vegetasi tanaman yang khas.",
    "Leungli": "Pola hias batik Leungli terinspirasi dari dongeng Sunda tentang Si Leungli, kisah penuh makna yang menceritakan Nyi Bungsu Rarang, seorang gadis malang yang menemukan persahabatan sejati dengan seekor ikan mas bernama Si Leungli. Dongeng ini mengajarkan bahwa alam akan memberikan kebaikan jika diperlakukan dengan baik. Ungkapan 'Melak cabé jadi cabé, melak onténg jadi bonténg', menggambarkan pesan moral bahwa kebaikan akan selalu berbuah kebaikan, begitu pula sebaliknya. Si Leungli juga menjadi simbol ekologis yang mengingatkan kita bahwa ikan hanya dapat hidup dan berkembang di lingkungan yang bersih, berair jernih, dan diperlakukan dengan baik. Di Sukabumi spesies ikan mas (Cyprinus carpio Linnaeus) yang tumbuh dengan baik, berkualitas dan menjadi aset yang hingga kini mash dibudidayakan di Balai Besar Perikanan Budidaya Air Tawar (BBPBAT) Sukabumi, dahulu bernama Landbouw School tahun 1920 pada zaman Belanda yang menjadi sekolah pertanian di Sukabumi.",
    "Makara": "Pola hias batik Makara terinspirasi dari makhluk mitologi dalam agama Hindu bernama Makara'. Makara memiliki bentuk unik berupa kombinasi belalai gajah, kepala singa, dan tubuh ikan, melambangkan perlindungan dan kekuatan. Selain dikenal sebagai simbol mitologi, Makara juga memiliki makna mendalam dalam budaya Sunda. Bagi masyarakat Sukabumi yang hidup berdampingan dengan laut, seperti di kawasan pantai Pelabuhan Ratu, Makara dipercaya dapat memahami dan melindungi manusia. Ketika seseorang merasa penat atau menghadapi kesulitan, berteriak di hadapan laut diyakini mampu memberikan rasa lega dan ketenangan hati. Makara menjadi simbol perlindungan, harmoni, dan hubungan erat manusia dengan elemen air, mengingatkan kita akan pentingnya menjaga keseimbangan alam sekaligus menghormati keajaiban yang ada di dalamnya.",
    "Mandala Bagja": "Pola hias batik Mandala Bagja terinspirasi dari konsep ‘Mandala’, yang berasal dari Bahasa Sanskerta dan bermakna ‘lingkaran yang utuh’. Kata ‘Bagja’ sendiri diambil dari Bahasa Sunda, yang berarti ‘kebahagiaan’. Pola hias batik ini menjadian Mandala berbentuk lingkaran sebagai objek utama dengan makna filosofis yang mendalam. Mandala Bagja merepresentasikan perjalanan dan perputaran kehidupan manusia yang harmonis, diwarnai dengan rasa kebahagiaan sebagai tujuan utama serta memberikan keyakinan bahwa perputaran hidup manusia pasti akan berakhir indah. Pola hias ini menjadi sebuah harapan dan keyakinan bagi Masyarakat Sukabumi dalam menjalani kehidupan. ",
    "Manuk Julang": "Pola hias batik Manuk Julang terinspirasi dari burung rangkong papan (Buceros Bicornis), satwa endemik asli Indonesia. Dalam bahasa Sunda, burung ini dikenal sebagai Manuk Julang. Spesies burung ini hingga kini mash dijaga di Pusat Penyelamatan Satwa Cikananga (PPSC) di Kecamatan Nyalingdung, Kabupaten Sukabumi. Manuk julang memiliki karakter pantang menyerah, terlihat dari kegigihannya saat terbang mencari sumber air atau makanan hingga berhasil mendapatkannya. Kegigihan burung ini menjadi inspirasi bagi bentuk ikat kepala 'julang ngapak', yang dikenakan oleh 'lengser' bijak yang menguasai ilmu pengetahuan dan berperan sebagai penasihat raja. Selain itu, symbol 'julang ngapak' (Burung julang dengan sayap yang terbentang) juga ditemukan pada atap rumah tradisional di Jawa Barat, melambangkan kekuatan dan perlindungan. Pola hias batik ini merepresentasikan nilai-nilai kegigihan, kebijaksanaan, dan harmoni yang erat kaitannya dengan budaya Sunda.",
    "Masagi": "Pola hias batik Masagi terinspirasi dari filosofi kehidupan masyrakat di Jawa Barat tentang manusia paripurna Sunda yang seimbang, teguh, kokoh dalam berpikir, berucap dan berperilaku. Dalam Bahasa Sunda, ‘Masagi’ berarti ‘persegi’, yang melambangkan kesetaraan sisi dan menjadi symbol dari keseimbangan. Motif pada pola hias batik ini mencakup berbagai objek, seperti kendi (Monumenalun-alun Kota Sukabumi), unsur air, sayap manuk julang, biji/fuli pala, dan bunga Wijayakusuma. Setiap objek dalam pola hias batik Masagi memiliki makna yang erat kaitannya dengan harapan agar Masyarakat Sukabumi menjadi insan paripurna.",
    "Mata Air Sukabumi": "Pola hias batik Mata Air Sukabumi terinspirasi dari kondisi geografis Sukabumi yang kaya akan sumber mata air. Air sebagai sumber kehidupan memiliki peran yang sangat penting bagi masyarakat Sunda, termasuk di Kota Sukabumi. Hal ini tercermin dari banyaknya nama tempat yang diawali dengan Tji atau Ci, yang dalam bahasa Sunda berarti cai atau air. Air tidak hanya mewakili unsur alam yang baik, tetapi juga menjadi simbol harapan, cita-cita, dan doa yang terus mengalir, melambangkan keberlanjutan dan kehidupan manusia.",
    "Merak Kinanti": "Pola hias batik Merak Kinanti terinspirasi dari fauna endemic Pulau Jawa merak hijau (Pavo Muticus), salah satu hewan eksotis Indonesia. Kata ‘Kinanti’ diambil dari istilah dalam pupuh Sunda yang bermakna ‘kelak yang dinanti-nanti’ atau ‘yang di tunggu-tunggu’ Pada pola hias batik ini, merak digambarkan dalam posisi menunggu dengan sayap yang tidak terkepak, melambangkan kesabaran dan keyakinan bahwa keindahan akan datang pada waktunya sekaligus menjadi cerminan individu yang mampu menempatkan diri disegala situasi dan kondisi. Merak Kinanti tidak hanya merepresentasikan keindahan alam, tetapi juga menyampaikan pesan filosofis tentang harapan dan ketenangan dalam menghadapi perjalanan kehidupan.",
    "Mozaik Kadudampit": "Pola hias batik Mozaik Kadudampit terinspirasi oleh bunga Kadudampit (Rhododendron Wilhelminae). Bunga ini merupakan jenis flora endemik Indonesia yang hanya tumbuh subur di ketinggian 1.350 mdpl, dapat ditemukan di Gunung Gede Pangrango serta Gunung Salak. Penamaan Rhododendron Wilhelminae sendiri berhubungan dengan kunjungan Ratu Wilhelmina ke Situ Gunung Kadudampit, Kabupaten Sukabumi. Mozaik Kadudampit memiliki latar belakang yang menampilkan pola mozaik atau pattern perupa garis-garis statis yang memberikan kesan harmonis pada desain. Makna mozaik pada pola hias batik merupakan metafora dari kumpulan informasi yang pengrajin dapat sehingga menghasilkan visual pola hias dengan latar belakang garis yang saling berhubungan. Melalui sola hias batik ini, diharapkan bunga Kadudampit dapat terus dilestarikan dan tumbuh dengar baik dengan subur, hal tersebut melambangkan ketahanan alam yang harus dijaga.",
    "Nakamesta": "Pola hias batik Nekamesta merupakan merepresentasikan konsep multiverse dalam kaitannya lengan alam semesta yang terdiri atas tiga dimensi kehidupan: masa kini, masa lalu, dan masa depan. Pola hias batik ini menggambarkan pentingnya penguasaan diri dalam Buana Handap (dunia bawah), Buana Panca Tengah (dunia tengah), dan Buana Nyuncung (dunia atas), serta keberanian untuk menghadapi dan menyelaraskan ketiga dimensi tersebut demi menciptakan keseimbangan dan keharmonisan alam.",
    "Pakwan": "Pola hias batik Pakwan terinspirasi dari pakis. Tanaman ini banyak ditemukan di Sukabumi, menandakan bahwa daerah tersebut sejak dahulu merupakan kawasan hutan hujan tropis (rainforest). Bentuk pohon pakis dengan daunnya yang bergelung melingkar ke dalam menyimpan makna filosofis tentang perjalanan hidup manusia. Pakis melambangkan proses ntrospeksi, dimana seseorang diajak untuk terlebih dahulu mengenal jati dirinya sebelum berinteraksi secara seimbang dengan sesama, alam, dan Sang Pencipta. Bentuknya yang melingkar ke dalam mencerminkan pentingnya evaluasi diri sebelum menilai atau memberikan solusi kepada orang lain. Selain itu, pertumbuhan pohon pakis yang terus menjulang ke atas tetapi daunnya semakin merunduk melambangkan sikap rendah hati manusia yang tidak melupakan asal-usulnya, serta kesadaran untuk selalu menghargai sesama dalam perjalanan menuju tujuan hidupnya. Dalam pengetahuan tradisional Sunda, pakis memiliki peranan penting sebagai penjaga kesehatan tanaman lain dalam sebuah ekosistem.",
    "Palawan": "Pola hias batik Palawan terinspirasi oleh buah pala (Myristica fragrans Houtt) terutama pada bagian biji pala. Pala dalam bahasa Sanskerta memiliki kaitan erat dengan kata pahlawan. Phala-wan' berarti orang yang menghasilkan buah 'phala' berkualitas untuk bangsa, negara, dan agama melalui keberanian dan pengorbanannya dalam membela kebenaran. Seperti halnya pohon pala yang menjulang tinggi dan memiliki umur panjang, pahlawan juga memberikan manfaat besar bagi umat manusia dan selalu dikenang. Palawan memiliki motif berupa salur yang terinspirasi dari bagian fuli/biji pala yang membentuk huruf 's' yang melambangkan kekayaan dan kesinambungan atau stabilitas yang tidak pernah putus. Pohon pala tumbuh dengan baik dan berkualitas di Sukabumi, sehingga kehadirannya harus tetap dijaga.Palawan memiliki makna yang berhubungan dengan sosok pahlawan menjadi inspirasi dan teladan bagi masyarakat, yang diharapkan hadir dalam setiap diri individu untuk memberikan manfaat bagi orang banyak, serta berkontribusi membawa Sukabumi menuju masa depan yang lebih baik.",
    "Penyu Sukabumian": "Pola hias batik Penyu Sukabumian terinspirasi dari hewan penyu hijau (Chelonia Mydas), yang keberadaannya dilestarikan melalui penangkaran di Taman Pesisir Pantai Pangumbahan, Ujung Genteng, Kabupaten Sukabumi. Pantai ini merupakan satu-satunya lokasi di Sukabumi yang menjadi tempat pendaratan alami bagi penyu hijau. Penyu Sukabumian menjadi salah satu simbol penting dalam pelestarian ekosistem laut setempat.",
    "Puyuh": "Pola hias batik Puyuh terinspirasi toponimi salah satu Kecamatan di Kota Sukabumi, yaitu Kecamatan Gunung Puyuh. Terdapat cerita rakyat yang berkembang di masyarakat Sukabumi. Kisah ini bermula dari sebuah perbukitan yang tidak berpenghuni. Menurut cerita, jika terdengar suara gemuruh dari gunung, masyarakat percaya itu pertanda alam sedang tidak baik-baik saja. Namun ternyata gemuruh yang terdengar tidak membawa kejadian apa pun, melainkan berasal dari suara burung yang ramai dan riuh. Ternyata suara itu berasal dari burung puyuh (Coturnix). Sejak saat itu, daerah tersebut dikenal dengan nama Gunung Puyuh. Objek lain dalam pola hias batik Puyuh merupakan daun Jamuju (Podocarpus Imbricatus) erupakan spesies pohon besar memiliki bentuk daun sempit pada ranting pendek menyirip dan eperti sisik pada ujung ranting. Selain itu, daun jamuju dapat dimafaatkan untuk pengobatan tradisional.",
    "Rereng Gunung Parang": "Pola hias batik Rereng Gunung Parang terinspirasi dari toponomi Sukabumi di masa lampau, yaitu Gunung Parang. Batik ini diciptakan sebagai upaya untuk menjaga ingatan kolektif dan merawat warisan cerita rakyat legenda Pakujajar di Gunung Parang. Cerita rakyat Pakujajar di Gunung Parang menimpan pesan moral yang menjadi bagian dari identitas Masyarakat Sukabumi. Tokoh Utama Bernama Wangsa Suta digambarkan dengan karakter yang kuat, keteguhan, optimisme, dan semangat. Pembuktian yang dimiliki Wangsa Suta menjadikannya sosok legendaris dalam kisah ini, yaitu sosok yang pertama kali membuka lahan atau ‘Ngababakan’ hutan belantara, sehingga menjadi cikal bakal peradaban baru yang kini dikenal dengan nama Sukabumi",
    "Rereng Tjaiwangi": "Pola hias batik Rereng Tjaiwangi terinspirasi dari dongeng rakyat tentang Nyimas Tjaiwangi, seorang wanita yang dikenal sangat mencintai alam. Dikisahkan bahwa pada masa lalu, seorang raksasa dari Gunung Arca datang untuk merusak lingkungan. Dengan keberanian, Nyimas Tjaiwangi behasil mengalahan raksasa tersebut menggunakan sehelai tenunan serat daun suji. Tubuh raksasa itu terbungkus oleh tenunan daun suji, lalu jatuh ke dalam kolam air panas. Akibatnya, tubuhnya melepuh dan akhirnya menghilang, menyelamatkan alam dari ancamannya. Kisah Nyimas Tjaiwangi tidak hanya menggambarkan sikap keberanian dan kecintaannya terhadap alam, tetapi juga menjadi symbol penting dalam upaya pelestarian lingkungan. Pola hias batik Rereng Tjaiwangi mengabdikan nilai-nilai ini melalui pola yang mencerminkan harmoni antara manusia dan alam. Hal tersebut mengingatkan pentingnya menjaga keseimbangan ekosistem untuk generasi mendatang.",
    "Wijayakusumah": "Pola hias batik Wijayakusumah terinspirasi oleh bunga Wijayakusuma, (Epiphyllum Anguliger), salah satu jenis tanaman kaktus yang dapat tumbuh subur di daerah dengan iklim sedang hingga tropis, termasuk Indonesia. Bunga Wijayakusuma melambangkan cahaya yang menerangi dalam kegelapan, serta menjadi teladan kebaikan, kejujuran, kewibawaan, dan simbol kemenangan. Kata 'Wijaya' yang berasal dari kata 'Widya' berarti pengetahuan, yang menggambarkan bagaimana manusia bisa menjadi penerang di dunia melalui ilmu pengetahuan. Melalui pola hias ini, diharapkan masyarkat Sukabumi dapat menjadi individu yang mumpuni di berbagai bidang ilmu, sebagai bekal untuk menjalani kehidupan dengan bijaksana, baik di masa kini maupun masa depan.",
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