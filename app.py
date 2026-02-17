import streamlit as st
import pandas as pd
import joblib
import time
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter  # Pasta grafiği sayımı için gerekli
from sklearn.metrics import f1_score

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="SOC - Siber Güvenlik Merkezi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Stil (Karanlık Mod ve Tablo Fontu)
st.markdown("""
<style>
    .stMetric {background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; color: white; padding: 10px;}
    div.stButton > button {width: 100%; background-color: #FF4B4B; color: white;}
    .attack-list {font-family: monospace; color: #ff6b6b; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_all_models():
    try:
        # Binary Model Dosyaları
        binary_model = joblib.load("xgboost_model_regularized.pkl")
        binary_scaler = joblib.load("xgboost_scaler.pkl")
        binary_cols = joblib.load("xgboost_columns.pkl")
        try: binary_num_cols = joblib.load("xgboost_num_cols.pkl")
        except: binary_num_cols = []
        
        # Type (Çoklu Sınıflandırma) Model Dosyaları
        # NOT: Eğer bu dosyalar yoksa try-except bloğu None döndürür.
        type_model = joblib.load("xgboost_type_model.pkl")
        type_scaler = joblib.load("xgboost_type_scaler.pkl")
        type_le = joblib.load("xgboost_type_le.pkl")
        type_cols = joblib.load("xgboost_type_columns.pkl") 
        return binary_model, binary_scaler, binary_cols, binary_num_cols, type_model, type_scaler, type_cols, type_le
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None, None, None, None, None, None, None, None

# Modelleri yükle
(bin_model, bin_scaler, bin_cols, bin_num, 
 typ_model, typ_scaler, typ_cols, typ_le) = load_all_models()

# SHAP Açıklayıcı
@st.cache_resource
def build_shap_explainer(_model):
    return shap.TreeExplainer(_model)

if "shap_explainer" not in st.session_state:
    st.session_state.shap_explainer = build_shap_explainer(bin_model)

# --- CSV OKUYUCU FONKSİYONU ---
def multi_csv_reader(files, chunk_size):
    for file in files:
        sample = pd.read_csv(file, nrows=1, header=None)
        file.seek(0)
        has_header = "srcip" in str(sample.iloc[0].values).lower()
        header_opt = 0 if has_header else None

        reader = pd.read_csv(
            file,
            chunksize=chunk_size,
            header=header_opt,
            encoding="utf-8-sig"
        )

        for chunk in reader:
            yield chunk

# --- VERİ İŞLEME FONKSİYONLARI ---
def clean_and_prepare(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    mapping = {
        "smeansz": "smean", "dmeansz": "dmean", "res_bdy_len": "response_body_len",
        "sintpkt": "sinpkt", "dintpkt": "dinpkt", "label": "label",
        "spkts": "spkts", "dpkts": "dpkts", "dur": "dur"
    }
    df = df.rename(columns=mapping)
    
    # Rate (Hız) sütunu hesaplama
    if 'rate' not in df.columns:
        if {'spkts', 'dpkts', 'dur'}.issubset(df.columns):
            s = pd.to_numeric(df['spkts'], errors='coerce').fillna(0)
            d = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)
            du = pd.to_numeric(df['dur'], errors='coerce').fillna(0)
            df['rate'] = (s + d) / (du.replace(0, 0.000001))
        else:
            df['rate'] = 0         
    return df

def get_model_input(df, scaler, target_cols, num_cols_ref=None):
    drop_list = ['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'attack_cat', 'label', 'id', 'tahmin', 'olasilik']
    X = df.drop(columns=[c for c in df.columns if c in drop_list], errors='ignore')
    X = pd.get_dummies(X)
    X = X.loc[:, ~X.columns.duplicated()]
    
    if num_cols_ref: 
        for col in num_cols_ref:
            if col not in X.columns: X[col] = 0
        exist = [c for c in num_cols_ref if c in X.columns]
        X[exist] = scaler.transform(X[exist])
    elif scaler:
        X = X.reindex(columns=target_cols, fill_value=0)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        return X

    return X.reindex(columns=target_cols, fill_value=0)

# --- ARAYÜZ BAŞLIYOR ---
st.title("🛡️ Tehdit Algılama Sistemi")
st.sidebar.title("Kontrol Paneli")

uploaded_files = st.sidebar.file_uploader(
    "Ağ Trafiği Dosyaları (CSV)",
    type=["csv"],
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("İşleme Hızı (Paket/Döngü)", 100, 2000, 500)
threshold = st.sidebar.slider("Alarm Eşiği", 0.0, 1.0, 0.5)

if bin_model is None:
    st.error("Model dosyaları (pkl) bulunamadı! Lütfen proje klasörünü kontrol et.")
    st.stop()

if uploaded_files and st.sidebar.button("ANALİZİ BAŞLAT"):
    
    # 1. BÖLÜM: CANLI TABLO
    st.subheader("📡 Canlı Paket Akışı")
    table_spot = st.empty() 
    
    st.markdown("---") 

    # 2. BÖLÜM: METRİKLER
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_box1 = m1.empty()
    metric_box2 = m2.empty()
    metric_box3 = m3.empty()
    metric_box4 = m4.empty()
    metric_box5 = m5.empty()

    # 3. BÖLÜM: GRAFİKLER (Zaman + Pasta)
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.subheader("📈 Zaman Çizelgesi")
        line_chart_spot = st.empty()
        
    with c_chart2:
        st.subheader("Saldırı Türü Dağılımı")
        pie_chart_spot = st.empty()

    st.markdown("---")
    
    # 4. BÖLÜM: ANALİZ VE LOG (SHAP + Liste)
    c_shap, c_list = st.columns([1, 1])
    
    with c_shap:
        st.subheader("Yapay Zeka Analizi (SHAP)")
        shap_spot = st.empty()
        
    with c_list:
        st.subheader("Tehdit Listesi")
        attack_list_spot = st.empty()
    
    # --- DEĞİŞKENLER ---
    total_packets = 0
    total_attacks = 0

    # --- F1 için etiket havuzları ---
    bin_true_labels = []
    bin_pred_labels = []

    multi_true_labels = []
    multi_pred_labels = []

    detected_attacks_log = []
    history_data = []
    
    # Saldırı kayıtlarını tutmak için listeler
    all_attack_frames = []      # İndirilebilir CSV için saldırı paketlerini saklar
    type_counter = Counter()    # Pasta grafiği için türleri sayar
    
    explainer = build_shap_explainer(bin_model)
    orijinal_isimler = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", 
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", 
        "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", 
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", 
        "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", 
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", 
        "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", 
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
    ]

    reader = multi_csv_reader(uploaded_files, chunk_size)
    
    # --- CANLI DÖNGÜ ---
    for chunk in reader:
        
        # Sütun İsimlendirme
        if 'srcip' not in chunk.columns:
            if len(chunk.columns) == len(orijinal_isimler):
                chunk.columns = orijinal_isimler
            else:
                chunk.columns = orijinal_isimler[:len(chunk.columns)]

        # ID Atama
        if 'id' not in chunk.columns:
            chunk['id'] = range(total_packets + 1, total_packets + len(chunk) + 1)

        # Veri Temizleme ve Hazırlık
        df_clean = clean_and_prepare(chunk.copy())
        
        # 1. Aşama: Binary Tahmin
        X_bin = get_model_input(df_clean.copy(), bin_scaler, bin_cols, bin_num)
        probs = bin_model.predict_proba(X_bin)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        # 2. Aşama: Saldırı Türü Tahmini
        attack_indices = np.where(preds == 1)[0]
        attack_types = ["-"] * len(preds)
        
        if len(attack_indices) > 0:
            # Sadece saldırı olanları filtrele
            X_typ_raw = df_clean.iloc[attack_indices].copy()
            
            # Eğer Type modeli yüklü ise tahmin yap, değilse "Generic Attack" de
            if typ_model is not None:
                X_typ = get_model_input(X_typ_raw, typ_scaler, typ_cols, None)
                type_preds = typ_model.predict(X_typ)
                type_labels = typ_le.inverse_transform(type_preds)
            else:
                type_labels = ["Generic Attack"] * len(attack_indices)
            
            # --- Multiclass F1 için veri toplama ---
            if 'attack_cat' in df_clean.columns and typ_model is not None:
                true_types = df_clean.iloc[attack_indices]['attack_cat'].values
                pred_types = type_labels

                multi_true_labels.extend(true_types)
                multi_pred_labels.extend(pred_types)

            # Sonuçları işle
            for i, idx in enumerate(attack_indices):
                turu = type_labels[i]
                attack_types[idx] = turu
                detected_attacks_log.insert(0, f"#{chunk['id'].iloc[idx]} ➔ {turu}")
            
            # Sayaç Güncelle (Pasta Grafiği için)
            type_counter.update(type_labels)

        # Sonuçları DataFrame'e yaz
        chunk['DURUM'] = ["SALDIRI" if p==1 else "NORMAL" for p in preds]
        chunk['TUR'] = attack_types
        chunk['SKOR'] = probs
        
        # Sadece saldırı olan paketleri sakla (Download için)
        if len(attack_indices) > 0:
            attacks_only = chunk.iloc[attack_indices].copy()
            all_attack_frames.append(attacks_only)

        # --- GÖRSEL GÜNCELLEMELER ---
        
        # 1. Tabloyu Güncelle
        show_cols = [c for c in ['id', 'srcip', 'proto', 'dur', 'rate', 'DURUM', 'TUR', 'SKOR'] if c in chunk.columns]
        
        def highlight_row(row):
            if row.DURUM == 'SALDIRI':
                return ['background-color: #5a1a1a; color: #ffcccc; font-weight: bold'] * len(row)
            else:
                return ['background-color: #1e2e1e; color: #ccffcc'] * len(row)

        styled_table = chunk[show_cols].tail(50).style.apply(highlight_row, axis=1).format({'SKOR': "{:.4f}", 'rate': "{:.2f}"})
        table_spot.dataframe(styled_table, use_container_width=True)

        # İstatistik Hesapla
        saldiri_sayisi = sum(preds)
        total_packets += len(chunk)
        total_attacks += saldiri_sayisi
        
        f1_text = "N/A"

        if 'label' in df_clean.columns:
            y_true = df_clean['label'].astype(int).values
            y_pred = preds.astype(int)

            bin_true_labels.extend(y_true)
            bin_pred_labels.extend(y_pred)

        # 2. Line Grafiği Güncelle
        history_data.append(saldiri_sayisi)

        fig_line = px.area(
            y=history_data,
            title="Zaman İçindeki Saldırı Yoğunluğu",
            labels={'y': 'Saldırı Sayısı', 'index': 'Zaman'}
        )
        fig_line.update_traces(line_color='#FF4B4B')
        fig_line.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        line_chart_spot.plotly_chart(fig_line, use_container_width=True , key=f"line_{total_packets}")

        # 3. Pasta Grafiği Güncelle
        if type_counter:
            df_pie = pd.DataFrame.from_dict(
                type_counter,
                orient='index',
                columns=['sayac']
            ).reset_index()

            df_pie.rename(columns={'index': 'Saldırı Türü'}, inplace=True)

            fig_pie = px.pie(
                df_pie,
                values='sayac',
                names='Saldırı Türü',
                title='Tespit Edilen Saldırı Dağılımı',
                hole=0.4
            )

            fig_pie.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            pie_chart_spot.plotly_chart(fig_pie, use_container_width=True , key=f"pie_{total_packets}")
        else:
            pie_chart_spot.info("Henüz saldırı türü tespit edilmedi.")
     
        # 4. SHAP Güncelle (Sadece risk varsa)
        if sum(preds) > 0:
            try:
                risk_idx = np.argmax(probs)
                fig_shap, ax = plt.subplots()
                shap_values = explainer.shap_values(X_bin.iloc[[risk_idx]])
                shap.plots.bar(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_bin.iloc[risk_idx], feature_names=bin_cols), show=False)
                shap_spot.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                shap_spot.warning(f"SHAP hatası: {e}")
        else:
            shap_spot.info("✅ Şu an aktif tehdit yok")


        # 5. Listeyi Güncelle
        if detected_attacks_log:
            attack_list_spot.code("\n".join(detected_attacks_log[:15]), language="text")
        else:
            attack_list_spot.success("Sistem Temiz")
        
        # --- F1 Hesaplamaları ---
        bin_f1_text = "N/A"
        multi_f1_text = "N/A"

        if len(bin_true_labels) > 0:
            bin_f1 = f1_score(bin_true_labels, bin_pred_labels, average="binary")
            bin_f1_text = f"{bin_f1:.4f}"

        if len(multi_true_labels) > 0:
            multi_f1 = f1_score(
                multi_true_labels,
                multi_pred_labels,
                average="weighted"
            )
            multi_f1_text = f"{multi_f1:.4f}"
        # Akışı yavaşlat
        time.sleep(0.01)

    # --- DÖNGÜ SONRASI İŞLEMLER --- 
    # METRİKLERİ GÖSTER
    metric_box1.metric("📦 Toplam Paket", total_packets)
    metric_box2.metric("🚨 Saldırı Tespit", total_attacks) 
    metric_box3.metric("🎯 Binary F1-Score", bin_f1_text)
    metric_box4.metric("🧬 Multiclass F1-Score", multi_f1_text)
    risk = (total_attacks/total_packets)*100 if total_packets > 0 else 0
    metric_box5.metric("⚠️ Risk Oranı", f"%{risk:.2f}")

    # İndirme Butonu 
    if all_attack_frames:
        st.subheader("📥 Raporlama")
        final_attack_df = pd.concat(all_attack_frames)
        csv = final_attack_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="🚨 Tespit Edilen Saldırıları İndir (CSV)",
            data=csv,
            file_name='tespit_edilen_saldirilar.csv',
            mime='text/csv',
        )
    else:
        st.info("İndirilecek saldırı kaydı bulunamadı (Sistem temiz).")