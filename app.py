import streamlit as st
import pandas as pd
import torch
import re
import math
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# penyimpanan sementara 
CACHE_DIR = Path("./data_cache")
CACHE_FILE = CACHE_DIR / "session_data.json"

# fungsi
def save_session_to_file():
    """Simpan session state ke file JSON"""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        
        data_to_save = {
            'df_uploaded': st.session_state['df_uploaded'].to_dict() if st.session_state['df_uploaded'] is not None else None,
            'df_clean': st.session_state['df_clean'].to_dict() if st.session_state['df_clean'] is not None else None,
            'preprocessing_done': st.session_state['preprocessing_done']
        }
        
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False

def load_session_from_file():
    """Load session state dari file JSON"""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            st.session_state['df_uploaded'] = pd.DataFrame(data['df_uploaded']) if data['df_uploaded'] else None
            st.session_state['df_clean'] = pd.DataFrame(data['df_clean']) if data['df_clean'] else None
            st.session_state['preprocessing_done'] = data['preprocessing_done']
            
            return True
        return False
    except Exception as e:
        return False

def reset_all_data():
    """Reset semua data dan hapus cache"""
    st.session_state['df_uploaded'] = None
    st.session_state['df_clean'] = None
    st.session_state['preprocessing_done'] = False
    
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    
    st.success("✅ Semua data telah direset!")
    st.rerun()

# state inisiasi
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['df_uploaded'] = None
    st.session_state['df_clean'] = None
    st.session_state['preprocessing_done'] = False
    
    # load from file
    if load_session_from_file():
        st.toast("📂 Data sebelumnya berhasil dimuat!", icon="✅")

# --- SLANG DICTIONARY ---
SLANG_DICT = {
    "110": "layanan darurat",
    "1312": " ACAB (Semua Polisi Bajingan)",
    "1312 forever": "selalu kritik polisi",
    "acab": "all cops are bastards (semua polisi bajingan)",
    "adu": "mengadu",
    "aduan": "pengaduan",
    "aja": "saja",
    "anjay": "anjing (ekspresi kaget/kesal)",
    "anjir": "anjing kaget",
    "anjir polisi": "anjing polisi (kesal)",
    "arogansi": "arogansi",
    "baper": "bawa perasaan",
    "berbelit": "rumit",
    "bgt": "banget",
    "bhayangkara": "polri",
    "biaya": "biaya",
    "bisa2nya": "bisa bisanya",
    "bjir": "anjing kaget",
    "bkn": "bukan",
    "blm": "belum",
    "blum": "belum",
    "bucin": "budak cinta",
    "call 110": "layanan darurat 110",
    "call center 110": "call center 110",
    "cari tambahan": "cari tambahan",
    "cari tambahan polisi": "cari tambahan polisi",
    "cepat": "cepat",
    "cepat tanggap": "cepat tanggap",
    "cepat tanggap polri": "cepat tanggap Polri",
    "cepet": "cepat",
    "cepet tanggap": "cepat tanggap",
    "cokelat": "polisi",
    "coklat": "polisi",
    "cringe": "malu-maluin",
    "dah": "sudah",
    "damai": "suap",
    "damai di tempat": "suap/damai",
    "damkar": "damkar",
    "dg": "dengan",
    "dr" : "dari",
    "dgn": "dengan",
    "dipersulit": "dipersulit",
    "diurus": "diurus",
    "dong": "dong",
    "donk": "dong",
    "duit": "uang",
    "duit haram": "uang haram (dari pungli)",
    "duitnya": "uangnya",
    "dumas": "pengaduan masyarakat",
    "dumas polri": "pengaduan masyarakat Polri",
    "dumas presisi": "dumas presisi",
    "humas" : "hubungan masyarakat",
    "emang": "memang",
    "emg": "memang",
    "forever1312": "selalu kritik polisi",
    "fr": "beneran",
    "frustrasi": "frustrasi",
    "ga": "tidak",
    "ga gercep": "tidak gerak cepat",
    "ga responsif": "tidak responsif",
    "ga tanggap": "tidak tanggap",
    "gabut": "gak ada kerjaan",
    "gak": "tidak",
    "gas": "gas",
    "gaskeun": "ayo lanjut",
    "gerak": "bergerak",
    "gerak cepat": "gerak cepat",
    "geraknya cepet": "gerak cepat",
    "gercep": "gerak cepat",
    "gercep kalau viral": "gerak cepat kalau viral",
    "gercep polri": "gerak cepat Polri",
    "gk": "tidak",
    "gue": "saya",
    "gw": "saya",
    "halo dek": "polisi",
    "hubungi 110": "hubungi 110",
    "humanis polri": "humanis Polri",
    "isul": "polisi",
    "jb": "join bareng",
    "jd": "jadi",
    "julid": "iri nyinyir",
    "kalo": "kalau",
    "kayak": "seperti",
    "keadilan": "keadilan",
    "kecewa": "kecewa",
    "kecewa berat": "kecewa berat",
    "kek": "seperti",
    "kenapa": "kenapa",
    "kepercayaan polri": "kepercayaan Polri",
    "kepercayaan publik": "kepercayaan publik",
    "kepo": "ingin tahu",
    "kinerja": "kinerja",
    "kinerja polri": "kinerja Polri",
    "kl": "kalau",
    "klo": "kalau",
    "knp": "kenapa",
    "kyk": "kayak",
    "lah": "lah",
    "lambat": "lambat",
    "lambat banget": "lambat banget",
    "lambat respon": "lambat respons",
    "lambat tanggap": "lambat tanggap",
    "lambat tanggap polri": "lambat tanggap Polri",
    "lambatnya": "lambatnya",
    "lapor": "lapor",
    "laporan": "laporan",
    "laporan polisi": "laporan polisi",
    "lebay": "berlebihan",
    "lg": "lagi",
    "lit": "seru",
    "lo": "kamu",
    "lu": "kamu",
    "luar biasa": "luar biasa",
    "mager": "malas gerak",
    "mantap": "mantap",
    "mbah condro": "polisi",
    "moots": "mutuals",
    "muak": "muak",
    "muak polri": "muak Polri",
    "nder": "teman panggilan",
    "ngab": "anak bangsat",
    "ngga": "tidak",
    "nggak": "tidak",
    "nggak gercep": "tidak gerak cepat",
    "nggak responsif": "tidak responsif",
    "nggak tanggap": "tidak tanggap",
    "nih": "ini",
    "no cap": "nggak bohong",
    "no viral no justice": "tidak viral tidak ada keadilan",
    "nunggu viral": "baru gerak kalau viral",
    "nyari": "mencari",
    "oknum": "oknum polisi",
    "oknum aparat": "oknum aparat",
    "oknum polisi": "oknum polisi",
    "operasi jebakan": "operasi jebakan tilang",
    "pake": "pakai",
    "pakpol": "bapak polisi",
    "palak": "palak",
    "palak polisi": "palak polisi",
    "parcok": "polisi",
    "pd": "pada",
    "percuma lapor polisi": "percuma lapor polisi",
    "pig": "polisi",
    "pilih kasus": "pilih kasus",
    "halo deck" : "polisi",
    "polisi anjing": "polisi anjing",
    "polisi bisnis": "polisi bisnis (narkoba/pungli)",
    "polisi brengsek": "polisi brengsek",
    "polisi cepat": "polisi cepat",
    "polisi ga responsif": "polisi tidak responsif",
    "polisi gendut": "polisi gendut (sindiran malas/serakah)",
    "polisi hadir": "polisi hadir",
    "polisi lambat": "polisi lambat",
    "polisi oon": "polisi bodoh",
    "polisi tidur": "polisi tidur",
    "polri cepat": "polri cepat",
    "polri ga tanggap": "polri tidak tanggap",
    "polri gercep": "polri gerak cepat",
    "polri hadir": "polri hadir",
    "polri humanis": "polri humanis",
    "polri lambat": "polri lambat",
    "polri lambat tanggap": "polri lambat tanggap",
    "polri presisi": "polri presisi",
    "polri usut": "polri usut",
    "presisi": "presisi",
    "presisi polri": "presisi Polri",
    "pulis": "polisi",
    "pungli": "pungutan liar",
    "pungli tilang": "pungutan liar tilang",
    "reformasi polisi": "reformasi polisi",
    "reformasi polri": "reformasi polri",
    "respon cepat": "respons cepat",
    "respon polri": "respons Polri",
    "responnya cepet": "responsnya cepat",
    "respons darurat": "respons darurat",
    "responsif": "responsif",
    "responsivitas": "responsivitas",
    "ribet": "rumit",
    "salam tempel": "suap",
    "salty": "kesal/sensi",
    "salut": "salut",
    "samsat": "kantor samsat",
    "santuy": "santai aja",
    "sbg": "sebagai",
    "sdh": "sudah",
    "sm" : "sama",
    "seragam": "polisi",
    "sigap": "sigap",
    "silup": "polisi",
    "sim": "surat izin mengemudi",
    "slay": "keren banget",
    "sy": "saya",
    "tanggap": "tanggap",
    "tanggap bencana": "tanggap bencana",
    "tanggapan polri": "tanggapan Polri",
    "tdk": "tidak",
    "tebang pilih": "tebang pilih",
    "terima kasih": "terima kasih",
    "tilang": "tindakan langsung",
    "tilang elektronik": "tilang elektronik",
    "tilang ilegal": "tilang ilegal",
    "titip": "titip denda/suap",
    "titip denda": "suap/tilang damai",
    "tp": "tapi",
    "th" : "tahun",
    "tuntut keadilan" : "tuntut keadilan",
    "udah": "sudah",
    "urusan": "urusan",
    "usut tuntas": "usut tuntas",
    "utk": "untuk",
    "vibe": "suasana",
    "viral": "viral",
    "viral baru gerak": "viral baru gerak",
    "viral dulu": "baru gerak kalau viral",
    "wereng": "polisi",
    "wereng coklat": "polisi",
    "wkwk": "tertawa",
    "wkwkwk": "tertawa",
    "yg": "yang",
    "mo" : "mau",
    "santuyy" : "santai",
    "santuy" : "santai",
    "anj" : "anjing",
    "anjg" : "anjing",
    "abis" : "habis",
    "trs" : "terus",
    "trus" : "terus",
    "ortu" : "orang tua",
    "gx" : "gak",
    "yan" : "yang",
    "dsb" : "dan sebagainya",
    "kabid" : "kepala bidang",
    "kombes" : "komisaris besar polisi",
    "polda" : "polisi daerah",
    "ni" : "ini",
    "nyerang" : "menyerang",
    "udh" : "sudah",
    "ges" : "temen temen",
    "asep" : "asap",
    "ngarah" : "mengarah",
    "jg" : "juga",
}

# FUNGSI LOAD MODEL 
@st.cache_resource
def load_model():
    model_path = "./model_indobert"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Folder model tidak ditemukan di: {model_path}")
        st.info("Pastikan folder 'model_indobert' ada di direktori yang sama dengan script ini.")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# fungsi prediksi
def predict_sentiment(text, tokenizer, model):
    if not text or str(text).strip() == '':
        return 1  
    
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, 
                   padding="max_length", max_length=128)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()
 
label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

# fungsi pre-processing
def clean_noise(text):
    """Hapus link, mention, hashtag, dan karakter spesial"""
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def replace_slang(text):
    """Replace kata slang dengan kata baku"""
    words = str(text).split()
    return ' '.join([SLANG_DICT.get(word.lower(), word) for word in words])

# FUNGSI PAGINATION
def render_dataframe_with_pagination(df, key_prefix):
    if df is None or len(df) == 0:
        st.warning("Tidak ada data untuk ditampilkan.")
        return
    
    rows_per_page = 10
    total_rows = len(df)
    total_pages = math.ceil(total_rows / rows_per_page)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav2:
        current_page = st.number_input(
            f"Halaman (Total: {total_pages})", 
            min_value=1, 
            max_value=max(1, total_pages), 
            value=1, 
            step=1, 
            key=f"{key_prefix}_page"
        )
    
    start_idx = (current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    st.caption(f"Menampilkan baris {start_idx+1} sampai {end_idx} dari {total_rows} data.")

#FUNGSI BERSIHKAN NAMA KOLOM
def clean_column_names(df):
    """Bersihkan nama kolom dari karakter aneh dan duplikasi"""
    valid_cols = []
    for col in df.columns:
        col_str = str(col)
        if not col_str[0].isdigit() and 'Unnamed' not in col_str:
            valid_cols.append(col)
    return df[valid_cols]

# home
def show_home():
    st.title("Selamat Datang di Dashboard Analisis Sentimen")
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0; color: #31333F;'>
        <h3 style='color: #31333F; margin-bottom: 10px;'>📊 Platform Analisis Sentimen</h3>
        <p style='font-size: 16px; line-height: 1.6;'>
        Platform ini dirancang untuk menganalisis sentimen masyarakat di Media Sosial X menggunakan 
        <strong>IndoBERT (Indonesian BERT)</strong>. Dengan teknologi AI Natural Language Processing, 
        kami mengolah dan mengklasifikasikan opini publik terkait sentimen <strong>Kinerja Polri</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Alur Penggunaan:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**1️⃣ Import Data**\n\nUpload file CSV/Excel berisi **data mentah tanpa label** untuk diprediksi.")
    with col2:
        st.success("**2️⃣ Proses Data**\n\nLakukan preprocessing: Case Folding, Noise Removal, Slang Normalization, dan Tokenizing.")
    with col3:
        st.warning("**3️⃣ Klasifikasi**\n\n**Evaluasi**: Upload dataset yang sudah dilabel manual")
    
    st.divider()
    
    st.markdown("### 📈 Spesifikasi Teknis")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("🤖 Model", "IndoBERT Base")
    with col_b:
        st.metric("📚 Dataset", "3 Kelas Sentimen")
    with col_c:
        st.metric("⚙️ Metode", "Pre-trained Model")
    with col_d:
        st.metric("🎯 Output", "Negatif/Netral/Positif")
    
    st.divider()
    
    st.markdown("""
    💡 Cara Penggunaan:
    - **Import Data**: Upload data mentah (raw tweets/comments) tanpa label
    - **Evaluasi Model**: Upload file yang sudah dipreprocess dan dilabel manual (0=Negatif, 1=Netral, 2=Positif)
    - **Prediksi Data Baru**: Menggunakan data dari Import Data yang sudah di-preprocessing, **untuk data yang belum terlabel**
    - Model akan otomatis mendeteksi sentimen berdasarkan konteks kalimat
    - Data tersimpan otomatis (sementara)
    """)
    
    st.divider()
    if st.button("🗑️ Reset All Data", type="secondary", use_container_width=True):
        if st.session_state['df_uploaded'] is not None or st.session_state['df_clean'] is not None:
            reset_all_data()
        else:
            st.info("Tidak ada data untuk direset.")

# import data page
def show_import():
    st.title("📂 Import Data")
    
    st.markdown("### Upload File Excel atau CSV (Data Mentah)")
    st.info("Upload dataset mentah (raw tweets/comments) untuk diprediksi.")
    
    uploaded_file = st.file_uploader("Drag and drop file here", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            possible_cols = ['full_text', 'text', 'Text', 'tweet', 'Tweet', 'content']
            text_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
        
            df = df.dropna(subset=[text_col])
        
            df = df[df[text_col].astype(str).str.strip() != '']
        
            df = df.reset_index(drop=True)



            if len(df) == 0:
                st.error("❌ File kosong! Upload file yang berisi data.")
            else:
                df = clean_column_names(df)
                
                st.session_state['df_uploaded'] = df
                st.session_state['preprocessing_done'] = False
                
                # Save to file
                save_session_to_file()
                
                st.success(f"✅ Data berhasil diupload! Total: **{len(df)}** baris")
                st.info(f"📋 Kolom yang tersedia: `{'`, `'.join(df.columns)}`")
                
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {e}")
            st.info("Pastikan format file benar dan tidak corrupt.")
    
    if st.session_state['df_uploaded'] is not None:
        st.markdown("---")
        st.subheader("Preview Data:")
        render_dataframe_with_pagination(st.session_state['df_uploaded'], "import_raw")

# proses data page
def show_process():
    st.title("Proses Data (Preprocessing)")
    st.markdown("Tahapan pembersihan dan normalisasi data sebelum klasifikasi sentimen.")
    
    if st.session_state['df_uploaded'] is None:
        st.error("**Belum ada data!** Silakan import data terlebih dahulu di menu **Import Data**.")
        return

    df = st.session_state['df_uploaded'].copy()
    
    cols = df.columns.tolist()
    possible_text_cols = ['full_text', 'text', 'Text', 'tweet', 'Tweet', 'content', 'Content']
    text_col = None
    
    for col in possible_text_cols:
        if col in cols:
            text_col = col
            break
    
    if text_col is None:
        text_col = cols[0]
        st.warning(f"Kolom teks tidak terdeteksi otomatis. Menggunakan kolom pertama: **{text_col}**")
    else:
        st.success(f"✅ Menggunakan kolom teks: **{text_col}**")
    
    with st.expander("ℹ️ Penjelasan Tahapan Preprocessing"):
        st.markdown("""
        1. **Case Folding**: Mengubah semua huruf menjadi huruf kecil (lowercase)
        2. **Noise Removal**: Menghapus URL, mention (@username), hashtag, RT, dan karakter spesial
        3. **Slang Word Normalization**: Mengubah kata-kata tidak baku menjadi baku (contoh: 'gak' → 'tidak')
        4. **Tokenizing**: Memecah kalimat menjadi kata-kata individual
        """)
    
    if st.button("🚀 Mulai Pre-processing", type="primary", use_container_width=True):
        with st.spinner("⏳ Sedang memproses data..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📝 Tahap 1/4: Case Folding...")
            df['Case Folding'] = df[text_col].astype(str).str.lower()
            progress_bar.progress(25)
            
            status_text.text("🧹 Tahap 2/4: Noise Removal...")
            df['Noise Removal'] = df['Case Folding'].apply(clean_noise)
            progress_bar.progress(50)
            
            status_text.text("📖 Tahap 3/4: Slang Word Normalization...")
            df['Slang Removal'] = df['Noise Removal'].apply(replace_slang)
            progress_bar.progress(75)
            
            status_text.text("✂️ Tahap 4/4: Tokenizing...")
            df['Tokenizing'] = df['Slang Removal'].apply(lambda x: x.split() if x.strip() else [])
            progress_bar.progress(100)

            df = df.dropna(subset=['Slang Removal'])
            df = df[df['Slang Removal'].astype(str).str.strip() != '']
            df = df.reset_index(drop=True)
            
            st.session_state['df_clean'] = df
            st.session_state['preprocessing_done'] = True
            
            # Save to file
            save_session_to_file()
            
            status_text.empty()
            progress_bar.empty()
            
        st.success("✅ **Preprocessing Selesai!** Silakan lihat hasil di tab di bawah.")

    if st.session_state['preprocessing_done']:
        st.markdown("---")
        st.subheader("📊 Hasil Preprocessing")
        
        tab_cf, tab_nr, tab_sw, tab_tok, tab_over = st.tabs([
            "1️⃣ Case Folding", 
            "2️⃣ Noise Removal", 
            "3️⃣ Slang Normalization", 
            "4️⃣ Tokenizing", 
            "📋 Overview"
        ])
        
        df_res = st.session_state['df_clean']
        
        with tab_cf:
            st.markdown("Hasil Case Folding")
            st.caption("Mengubah semua teks menjadi huruf kecil (lowercase)")
            if text_col in df_res.columns and 'Case Folding' in df_res.columns:
                render_dataframe_with_pagination(df_res[[text_col, 'Case Folding']], "tab_cf")
            
        with tab_nr:
            st.markdown("Hasil Noise Removal")
            st.caption("Menghapus URL, mention, hashtag, RT, dan karakter spesial")
            if 'Case Folding' in df_res.columns and 'Noise Removal' in df_res.columns:
                render_dataframe_with_pagination(df_res[['Case Folding', 'Noise Removal']], "tab_nr")
            
        with tab_sw:
            st.markdown("Hasil Slang Word Normalization")
            st.caption("Mengubah kata tidak baku menjadi kata baku")
            if 'Noise Removal' in df_res.columns and 'Slang Removal' in df_res.columns:
                render_dataframe_with_pagination(df_res[['Noise Removal', 'Slang Removal']], "tab_sw")
            
        with tab_tok:
            st.markdown("Hasil Tokenizing")
            st.caption("Memecah kalimat menjadi kata-kata individual")
            if 'Slang Removal' in df_res.columns and 'Tokenizing' in df_res.columns:
                df_display = df_res[['Slang Removal', 'Tokenizing']].copy()
                df_display['Tokenizing'] = df_display['Tokenizing'].apply(str)
                render_dataframe_with_pagination(df_display, "tab_tok")
            
        with tab_over:
            st.markdown("Complete Preprocessing Overview")
            
            desired_columns = ['username', 'created_at', 'full_text', 'Case Folding', 'Noise Removal', 'Slang Removal', 'Tokenizing']
            final_columns = [col for col in desired_columns if col in df_res.columns]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Data", len(df_res))
            
            if 'Slang Removal' in df_res.columns:
                avg_len = df_res['Slang Removal'].astype(str).apply(len).mean()
                col2.metric("📏 Avg Text Length", f"{avg_len:.1f} chars")
            
            if 'Tokenizing' in df_res.columns:
                avg_words = df_res['Tokenizing'].apply(len).mean()
                col3.metric("🔤 Avg Words", f"{avg_words:.1f} words")
            
            st.markdown("---")
            st.dataframe(df_res[final_columns], use_container_width=True, height=400)
            
            csv = df_res[final_columns].to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil Preprocessing (CSV)",
                csv,
                "data_preprocessing.csv",
                "text/csv",
                use_container_width=True
            )

# klasifikasi page
def show_classification():
    st.title("Klasifikasi Data (IndoBERT)")
    
    tab_eval, tab_pred = st.tabs(["📊 Evaluasi Model", "Prediksi Data Baru"])
    
    # evaluasi model tab 1
    with tab_eval:
        st.header("📊 Evaluasi Model Performance")
        st.caption("Upload file yang sudah terlabel untuk evaluasi model dengan split 80% train, 10% validation, 10% test")
        
        st.markdown("---")
        st.markdown(" 📤 Upload Labeled Dataset")
        st.info("Upload file CSV yang sudah terlabel dengan kolom `label` (0=Negatif, 1=Netral, 2=Positif)")
        
        uploaded_labeled = st.file_uploader("Upload dataset terlabel", type=["csv"], key="labeled_upload")
        
        if uploaded_labeled:
            try:
                df_labeled = pd.read_csv(uploaded_labeled, encoding='utf-8')
                
                # Normalisasi kolom
                if 'Label' in df_labeled.columns:
                    df_labeled.rename(columns={'Label': 'label'}, inplace=True)
                
                if 'label' not in df_labeled.columns:
                    st.error("❌ Kolom 'label' tidak ditemukan dalam file!")
                else:
                    # Filter data berlabel
                    df_labeled_clean = df_labeled[df_labeled['label'].notna()].copy()
                    df_labeled_clean = df_labeled_clean[df_labeled_clean['label'].isin([0, 1, 2])].copy()
                    
                    if len(df_labeled_clean) == 0:
                        st.error("❌ Tidak ada data berlabel yang valid!")
                    else:
                        st.success(f"✅ File berhasil diupload! Total data berlabel: **{len(df_labeled_clean)}** baris")
                        
                        # Deteksi kolom teks
                        text_cols = ['full_text', 'text', 'Text', 'tweet', 'Tweet', 'content', 'Content']
                        text_col_labeled = None
                        for col in text_cols:
                            if col in df_labeled_clean.columns:
                                text_col_labeled = col
                                break
                        
                        if text_col_labeled is None:
                            text_col_labeled = df_labeled_clean.columns[0]
                            st.warning(f"⚠️ Menggunakan kolom pertama sebagai teks: **{text_col_labeled}**")
                        
                        # Info distribusi label
                        label_counts = df_labeled_clean['label'].value_counts().sort_index()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📊 Total", len(df_labeled_clean))
                        col2.metric("😞 Negatif (0)", label_counts.get(0, 0))
                        col3.metric("😐 Netral (1)", label_counts.get(1, 0))
                        col4.metric("😊 Positif (2)", label_counts.get(2, 0))
                        
                        st.markdown("---")
                        
                        # Evaluasi Model 
                        st.markdown("🎯 Evaluasi Model")
                        st.info("💡Evaluasi dengan split 80% train, 10% validation, 10% test")
                        
                        # Deteksi kolom teks hasil preprocessing
                        preprocessing_cols = ['text_slang_removal','Slang Removal', 'slang_removal', 'Tokenizing', 'tokenizing']
                        text_col_preprocessed = None
                        for col in preprocessing_cols:
                            if col in df_labeled_clean.columns:
                                text_col_preprocessed = col
                                break
                        
                        if text_col_preprocessed is None:
                            st.warning("Kolom hasil preprocessing tidak ditemukan. Menggunakan kolom teks original.")
                            text_col_preprocessed = text_col_labeled
                        else:
                            st.success(f"")
                        
                        if st.button("🚀 Mulai Evaluasi Model", type="primary", use_container_width=True):
                            tokenizer, model = load_model()
                            
                            if not tokenizer or not model:
                                st.error("❌ Model tidak dapat dimuat!")
                            else:
                                with st.spinner("⏳ Sedang melakukan evaluasi model..."):
                                    
                                    train_ratio = 0.8
                                    val_ratio = 0.1
                                    test_ratio = 0.1
                                    
                                    X = df_labeled_clean[text_col_preprocessed].values
                                    y = df_labeled_clean['label'].values
                                    
                                    # Split train dan temp (val+test)
                                    X_train, X_temp, y_train, y_temp = train_test_split(
                                        X, y, train_size=train_ratio, random_state=42, stratify=y
                                    )
                                    
                                    # Split temp menjadi val dan test (50-50 dari sisa 20%)
                                    X_val, X_test, y_val, y_test = train_test_split(
                                        X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp
                                    )
                                    
                                    st.info(f"✅ Data split: Train={len(X_train)} ({len(X_train)/len(X)*100:.1f}%), Val={len(X_val)} ({len(X_val)/len(X)*100:.1f}%), Test={len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
                                    
                                    # Prediksi pada test set
                                    st.markdown("🔄 Melakukan Prediksi pada Test Set...")
                                    preds = []
                                    
                                    progress_bar = st.progress(0)
                                    progress_text = st.empty()
                                    
                                    for i, text in enumerate(X_test):
                                        pred = predict_sentiment(text, tokenizer, model)
                                        preds.append(pred)
                                        progress = (i + 1) / len(X_test)
                                        progress_bar.progress(progress)
                                        progress_text.text(f"Processing: {i+1}/{len(X_test)}")
                                    
                                    progress_bar.empty()
                                    progress_text.empty()
                                    
                                    # Hitung metrics
                                    acc = accuracy_score(y_test, preds)
                                    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                                    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                                    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                                    
                                    # Simpan data labeled ke session state untuk digabung nanti
                                    if st.session_state['df_clean'] is not None:
                                        df_existing = st.session_state['df_clean'].copy()
                                        
                                        # Merge labeled data ke df_clean berdasarkan kolom text
                                        if text_col_labeled in df_existing.columns:
                                            
                                            df_labeled_clean['is_labeled'] = True
                                            
                                            # Merge berdasarkan text column
                                            df_merged = df_existing.merge(
                                                df_labeled_clean[[text_col_labeled, 'label', 'is_labeled']],
                                                on=text_col_labeled,
                                                how='left',
                                                suffixes=('', '_labeled')
                                            )
                                            
                                            st.session_state['df_clean'] = df_merged
                                            save_session_to_file()
                                            
                                            st.info(f"✅ Data labeled berhasil digabungkan dengan data existing ({len(df_merged[df_merged['is_labeled'] == True])} data berlabel)")
                                    
                                    st.success("✅ Evaluasi Selesai!")
                                    
                                    # Display Metrics
                                    st.markdown("---")
                                    st.markdown("📈 Hasil Evaluasi Model")
                                    
                                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                    col_m1.metric("🎯 Accuracy", f"{acc:.2%}")
                                    col_m2.metric("📊 Precision", f"{prec:.2%}")
                                    col_m3.metric("🔍 Recall", f"{rec:.2%}")
                                    col_m4.metric("⚖️ F1-Score", f"{f1:.2%}")
                                    
                                    # Confusion Matrix
                                    st.markdown("---")
                                    st.markdown("📈 Confusion Matrix")
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    cm = confusion_matrix(y_test, preds)
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                               xticklabels=['Negatif', 'Netral', 'Positif'],
                                               yticklabels=['Negatif', 'Netral', 'Positif'],
                                               ax=ax)
                                    ax.set_ylabel('Actual Label')
                                    ax.set_xlabel('Predicted Label')
                                    ax.set_title('Confusion Matrix')
                                    st.pyplot(fig)
                                    
                                    # Classification Report
                                    st.markdown("---")
                                    st.markdown("📋 Classification Report")
                                    report = classification_report(
                                        y_test, preds,
                                        target_names=['Negatif', 'Netral', 'Positif'],
                                        output_dict=True
                                    )
                                    df_report = pd.DataFrame(report).transpose()
                                    st.dataframe(df_report.style.format("{:.2f}"), use_container_width=True)
                                    
                                    # Distribusi Prediksi
                                    st.markdown("---")
                                    st.markdown("🥧 Distribusi Prediksi pada Test Set")
                                    
                                    col_viz1, col_viz2 = st.columns(2)
                                    
                                    with col_viz1:
                                        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                                        pred_counts = pd.Series(preds).value_counts().sort_index()
                                        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
                                        ax_pie.pie(pred_counts, labels=['Negatif', 'Netral', 'Positif'],
                                                   autopct='%1.1f%%', colors=colors, startangle=90)
                                        ax_pie.set_title('Distribusi Prediksi Model')
                                        st.pyplot(fig_pie)
                                    
                                    with col_viz2:
                                        fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                                        pred_counts.plot.bar(ax=ax_bar, color=colors)
                                        ax_bar.set_xlabel('Sentimen')
                                        ax_bar.set_ylabel('Jumlah')
                                        ax_bar.set_title('Bar Chart Prediksi')
                                        ax_bar.set_xticklabels(['Negatif', 'Netral', 'Positif'], rotation=0)
                                        for i, v in enumerate(pred_counts.values):
                                            ax_bar.text(i, v + 0.5, str(v), ha='center', va='bottom')
                                        st.pyplot(fig_bar)
                        
            except Exception as e:
                st.error(f"❌ Error saat membaca file: {e}")
    
    # page 2 data belum terlabel
    with tab_pred:
        st.header("Prediksi Data Baru")
        st.caption("Prediksi sentimen untuk data yang **belum terlabel**")
        
        if not st.session_state['preprocessing_done']:
            st.warning("⚠️ Belum ada data yang diproses!")
            st.info("Silakan lakukan langkah berikut:\n1. Upload data mentah di menu **Import Data**\n2. Lakukan **Preprocessing** di menu **Proses Data**\n3. Kembali ke sini untuk melakukan prediksi")
            return
        
        # Ambil data yang sudah dipreprocess
        df_all = st.session_state['df_clean'].copy()
        
        # Filter hanya data yang belum terlabel (tidak punya kolom 'label' atau label = NaN)
        if 'label' in df_all.columns:
            df_to_predict = df_all[df_all['label'].isna()].copy()
            df_labeled_existing = df_all[df_all['label'].notna()].copy()
            
            st.info(f"📊 Data berlabel (dari evaluasi): **{len(df_labeled_existing)}** baris")
            st.info(f"🔮 Data yang akan diprediksi: **{len(df_to_predict)}** baris")
            
            if len(df_to_predict) == 0:
                st.success("✅ Semua data sudah terlabel! Tidak ada data baru untuk diprediksi.")
                st.info("Silakan upload data baru di menu **Import Data** untuk prediksi.")
                return
        else:
            # Jika tidak ada kolom label sama sekali, semua data dianggap unlabeled
            df_to_predict = df_all.copy()
            st.info(f"📊 Total data yang akan diprediksi: **{len(df_to_predict)}** baris")
        
        # Tombol Prediksi
        if st.button("🚀 Mulai Prediksi Data Baru", type="primary", use_container_width=True):
            tokenizer, model = load_model()
            
            if not tokenizer or not model:
                st.error("❌ Model tidak dapat dimuat!")
                return
            
            with st.spinner("⏳ Sedang memprediksi sentimen..."):
                preds_new = []
                texts_new = df_to_predict['Slang Removal'].tolist()
                
                prog = st.progress(0)
                prog_text = st.empty()
                
                for i, txt in enumerate(texts_new):
                    pred = predict_sentiment(txt, tokenizer, model)
                    preds_new.append(pred)
                    progress = (i + 1) / len(texts_new)
                    prog.progress(progress)
                    prog_text.text(f"Processing: {i+1}/{len(texts_new)}")
                
                prog.empty()
                prog_text.empty()
                
                df_to_predict['Prediksi Label'] = preds_new
                df_to_predict['Prediksi Sentimen'] = df_to_predict['Prediksi Label'].map(label_map)
                
                if 'label' in df_all.columns and len(df_labeled_existing) > 0:
                    df_labeled_existing['Prediksi Label'] = df_labeled_existing['label']
                    df_labeled_existing['Prediksi Sentimen'] = df_labeled_existing['Prediksi Label'].map(label_map)
                    
                    df_combined = pd.concat([df_labeled_existing, df_to_predict], ignore_index=True)
                    st.session_state['df_clean'] = df_combined
                else:
                    st.session_state['df_clean'] = df_to_predict
                
                # Save to file
                save_session_to_file()
                
            st.success("✅ Prediksi Selesai!")
            
            # Statistik Hasil (hanya untuk data yang baru diprediksi)
            st.markdown("---")
            st.markdown("📊 Statistik Hasil Prediksi (Data Baru)")
            
            sentiment_counts = df_to_predict['Prediksi Sentimen'].value_counts()
            total = len(df_to_predict)
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            col_stat1.metric("📊 Total Data", total)
            col_stat2.metric("😊 Positif", f"{sentiment_counts.get('Positif', 0)} ({sentiment_counts.get('Positif', 0)/total*100:.1f}%)")
            col_stat3.metric("😐 Netral", f"{sentiment_counts.get('Netral', 0)} ({sentiment_counts.get('Netral', 0)/total*100:.1f}%)")
            col_stat4.metric("😞 Negatif", f"{sentiment_counts.get('Negatif', 0)} ({sentiment_counts.get('Negatif', 0)/total*100:.1f}%)")
            
            # Visualisasi
            st.markdown("---")
            st.markdown("📈 Visualisasi Hasil")
            
            col_viz1, col_viz2 = st.columns(2)
            
            # Word Cloud
            with col_viz1:
                st.markdown("☁️ Word Cloud")
                text_all = " ".join(df_to_predict['Slang Removal'].dropna().astype(str))
                if text_all.strip():
                    try:
                        wc = WordCloud(width=800, height=400, background_color='white',
                                     colormap='viridis').generate(text_all)
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.warning(f"⚠️ Tidak dapat membuat word cloud: {e}")
                else:
                    st.warning("⚠️ Tidak ada teks untuk membuat Word Cloud.")
            
            # Pie Chart
            with col_viz2:
                st.markdown("🥧 Distribusi Sentimen")
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
                sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax_pie, colors=colors,
                                         startangle=90, textprops={'fontsize': 12})
                ax_pie.set_ylabel('')
                ax_pie.set_title('Distribusi Sentimen Hasil Prediksi', fontsize=14, fontweight='bold')
                st.pyplot(fig_pie)
            
            # Bar Chart
            st.markdown("📊 Bar Chart Sentimen")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            sentiment_counts.plot.bar(ax=ax_bar, color=['#ff6b6b', '#4ecdc4', '#95e1d3'])
            ax_bar.set_xlabel('Sentimen', fontsize=12)
            ax_bar.set_ylabel('Jumlah', fontsize=12)
            ax_bar.set_title('Distribusi Sentimen (Bar Chart)', fontsize=14, fontweight='bold')
            ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=0)
            for i, v in enumerate(sentiment_counts.values):
                ax_bar.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=11)
            st.pyplot(fig_bar)
            
            # Tabel Hasil
            st.markdown("---")
            st.markdown("📋 Tabel Hasil Prediksi (Data Baru)")
            
            # Deteksi kolom original
            cols = df_to_predict.columns.tolist()
            text_col_orig = None
            for col in ['full_text', 'text', 'Text', 'tweet']:
                if col in cols:
                    text_col_orig = col
                    break
            
            if text_col_orig:
                display_cols = [text_col_orig, 'Slang Removal', 'Prediksi Sentimen']
            else:
                display_cols = ['Slang Removal', 'Prediksi Sentimen']
            
            # Filter berdasarkan sentimen
            filter_option = st.selectbox(
                "🔍 Filter berdasarkan sentimen:",
                ["Semua", "Positif", "Netral", "Negatif"],
                key="filter_pred"
            )
            
            if filter_option != "Semua":
                df_display = df_to_predict[df_to_predict['Prediksi Sentimen'] == filter_option]
            else:
                df_display = df_to_predict
            
            st.dataframe(df_display[display_cols], use_container_width=True, height=400)
            
            # Statistik Gabungan (jika ada data labeled)
            if 'label' in df_all.columns and len(df_labeled_existing) > 0:
                st.markdown("---")
                st.header("📊 Statistik & Visualisasi Total ")
                st.info("Bagian ini menampilkan analisis gabungan dari Data Latih (Labeled) + Data Baru (Prediksi).")

                
                df_final = st.session_state['df_clean'].copy()
                df_final = df_final.dropna(how='all')  
                df_final = df_final.dropna(subset=['Slang Removal']) 
                df_final = df_final[df_final['Slang Removal'].astype(str).str.strip() != '']
                df_final = df_final.reset_index(drop=True)
                
                sentiment_counts_total = df_final['Prediksi Sentimen'].value_counts()
                total_final = len(df_final) 
                
                # 1. METRICS ANGKA
                col_total1, col_total2, col_total3, col_total4 = st.columns(4)
                col_total1.metric("📊 Total Semua Data", total_final)
                col_total2.metric("😊 Positif", f"{sentiment_counts_total.get('Positif', 0)} ({sentiment_counts_total.get('Positif', 0)/total_final*100:.1f}%)")
                col_total3.metric("😐 Netral", f"{sentiment_counts_total.get('Netral', 0)} ({sentiment_counts_total.get('Netral', 0)/total_final*100:.1f}%)")
                col_total4.metric("😞 Negatif", f"{sentiment_counts_total.get('Negatif', 0)} ({sentiment_counts_total.get('Negatif', 0)/total_final*100:.1f}%)")
                
                # 2. VISUALISASI CHART GABUNGAN
                st.markdown("📈 Visualisasi Data Gabungan")
                col_viz_tot1, col_viz_tot2 = st.columns(2)
                
                # Word Cloud Total
                with col_viz_tot1:
                    st.markdown("☁️ Word Cloud (Total)")
                    text_total = " ".join(df_final['Slang Removal'].dropna().astype(str))
                    if text_total.strip():
                        try:
                            wc_total = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_total)
                            fig_wc_total, ax_wc_total = plt.subplots(figsize=(10, 5))
                            ax_wc_total.imshow(wc_total, interpolation='bilinear')
                            ax_wc_total.axis("off")
                            st.pyplot(fig_wc_total)
                        except:
                            st.warning("Gagal membuat Word Cloud total.")
                    else:
                        st.warning("Data teks kosong.")
                
                # Pie Chart Total
                with col_viz_tot2:
                    st.markdown("🥧 Distribusi Sentimen (Total)")
                    fig_pie_total, ax_pie_total = plt.subplots(figsize=(8, 8))
                    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
                    
                    if not sentiment_counts_total.empty:
                        sentiment_counts_total.plot.pie(autopct='%1.1f%%', ax=ax_pie_total, colors=colors, startangle=90)
                        ax_pie_total.set_ylabel('')
                        st.pyplot(fig_pie_total)
                    else:
                        st.warning("Data sentimen kosong.")
                
                # Bar Chart Total (Full Width)
                st.markdown("📊 Bar Chart Sentimen (Total)")
                fig_bar_total, ax_bar_total = plt.subplots(figsize=(12, 5))
                
                if not sentiment_counts_total.empty:
                    sentiment_counts_total.plot.bar(ax=ax_bar_total, color=colors)
                    ax_bar_total.set_xlabel('Sentimen')
                    ax_bar_total.set_ylabel('Jumlah Data')
                    ax_bar_total.set_xticklabels(ax_bar_total.get_xticklabels(), rotation=0)
                    
                    # Tambah label angka di atas batang
                    for i, v in enumerate(sentiment_counts_total.values):
                        ax_bar_total.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig_bar_total)
            
            # Download Button
            st.markdown("---")
            st.markdown("📥 Download Hasil")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                
                df_download = st.session_state['df_clean']
                csv = df_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Semua Data (CSV)",
                    csv,
                    "hasil_sentimen_lengkap.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_dl2:
                # Download hanya data yang baru diprediksi
                if text_col_orig:
                    df_summary = df_to_predict[[text_col_orig, 'Prediksi Sentimen']]
                else:
                    df_summary = df_to_predict[['Slang Removal', 'Prediksi Sentimen']]
                
                csv_summary = df_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Data Baru Saja (CSV)",
                    csv_summary,
                    "hasil_prediksi_baru.csv",
                    "text/csv",
                    use_container_width=True
                )

# ================= NAVIGASI UTAMA =================
def main():
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                background-color: #2c3e50;
            }
            section[data-testid="stSidebar"] > div {
                padding-top: 2rem;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stButton>button {
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 10px;'>
                <h1 style='color: white; margin: 0;'>Dashboard</h1>
                <h2 style='color: white; margin: 5px 0;'>Analisis Sentimen</h2>
                <p style='color: #bdc3c7; margin: 0;'></p>
                <p style='color: #95a5a6; font-size: 12px; margin: 5px 0;'>Powered by IndoBERT</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        selected = option_menu(
            menu_title=None,
            options=["Beranda", "Import Data", "Proses Data", "Klasifikasi Data"],
            icons=["house-fill", "cloud-upload-fill", "gear-fill", "bar-chart-fill"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "5px", 
                    "padding": "10px 15px",
                    "color": "white",
                    "border-radius": "5px",
                    "--hover-color": "#34495e"
                },
                "nav-link-selected": {
                    "background-color": "#1abc9c",
                    "font-weight": "bold"
                },
            }
        )
        
        st.markdown("---")
        
        st.markdown("""
            <div style='text-align: center; color: #95a5a6; font-size: 12px; padding: 20px 10px;'>
                <p>📊 Dashboard Sentiment Analysis </p>
                <p>🤖 IndoBERT </p>
                <p>© 2026 Bintang Wahyu Ramadhan </p>
            </div>
        """, unsafe_allow_html=True)
        
    if selected == "Beranda":
        show_home()
    elif selected == "Import Data":
        show_import()
    elif selected == "Proses Data":
        show_process()
    elif selected == "Klasifikasi Data":
        show_classification()

if __name__ == '__main__':
    main()