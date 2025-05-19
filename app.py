import streamlit as st
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
import joblib
import os

st.set_page_config(page_title="BTC Fiyat Yönü Tahmini", layout="wide")

# --- GÜNCELLENMİŞ Yollar ---
MODEL_YOLU = "randomforest_rfe_model.pkl" 
OLCEKLEYICI_YOLU = "scaler.pkl"
OZNITELIKLER_YOLU = "rfe_selected_features.pkl" 
OLCEKLEYICI_SUTUNLARI_YOLU = "features_columns_for_scaler.pkl"


GECMIS_VERI_YOLU = "btc_verisi.csv"
GEREKLI_GECMIS_GUN_SAYISI = 40 

@st.cache_resource
def model_varliklarini_yukle():
    yollar = {
        "Model": MODEL_YOLU,
        "Ölçekleyici": OLCEKLEYICI_YOLU,
        "RFE Özellikleri": OZNITELIKLER_YOLU, # Açıklama güncellendi
        "Ölçekleyici Sütunları": OLCEKLEYICI_SUTUNLARI_YOLU
    }
    yuklenen_varliklar = {}
    hepsi_mevcut = True
    for ad, yol in yollar.items():
        if not os.path.exists(yol):
            st.error(f"{ad} dosyası bulunamadı: {yol}")
            hepsi_mevcut = False
    if not hepsi_mevcut:
        return None

    try:
        yuklenen_varliklar["model"] = joblib.load(MODEL_YOLU)
        yuklenen_varliklar["olcekleyici"] = joblib.load(OLCEKLEYICI_YOLU)
        yuklenen_varliklar["rfe_oznitelik_listesi"] = joblib.load(OZNITELIKLER_YOLU) # Anahtar güncellendi
        yuklenen_varliklar["olcekleyici_sutun_listesi"] = joblib.load(OLCEKLEYICI_SUTUNLARI_YOLU)
        return yuklenen_varliklar
    except Exception as e:
        st.error(f"Model varlıkları yüklenirken hata: {e}")
        return None

@st.cache_data
def gecmis_veriyi_yukle(dosya_yolu):
    if not os.path.exists(dosya_yolu):
        st.error(f"Geçmiş veri dosyası bulunamadı: {dosya_yolu}")
        return None
    try:
        df = pd.read_csv(dosya_yolu, usecols=['Open', 'Close', 'Volume'])
        if len(df) < GEREKLI_GECMIS_GUN_SAYISI:
            st.warning(f"Geçmiş veri dosyasında yetersiz satır var. En az {GEREKLI_GECMIS_GUN_SAYISI} gün gerekli, {len(df)} satır bulundu.")
        return df
    except Exception as e:
        st.error(f"Geçmiş veri yüklenirken hata oluştu: {e}")
        return None

def teknik_indikatorleri_hesapla(giris_df):
    df = giris_df.copy()
    if 'Close' not in df.columns:
        st.error("DataFrame indikatör hesaplaması için 'Close' sütununu içermelidir.")
        return pd.DataFrame()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if df['Close'].isnull().any():
        st.error("Close sütunu, sayısal olmayan değerler içeriyor veya dönüştürme sonrası NaN oldu.")
        return pd.DataFrame()

    if len(df) < 14: st.warning("RSI için yetersiz veri."); df["RSI"] = np.nan
    else: df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    if len(df) < 26: st.warning("MACD için yetersiz veri."); df["MACD"] = np.nan; df["Signal"] = np.nan
    else:
        macd_indicator = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd_indicator.macd()
        df["Signal"] = macd_indicator.macd_signal()

    if len(df) < 20: st.warning("SMA20 için yetersiz veri."); df["SMA20"] = np.nan
    else: df["SMA20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()

    if len(df) < 20: st.warning("EMA20 için yetersiz veri."); df["EMA20"] = np.nan
    else: df["EMA20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()

    return df

varliklar = model_varliklarini_yukle()
tam_gecmis_veri = gecmis_veriyi_yukle(GECMIS_VERI_YOLU)

st.title("Bitcoin (BTC/USD) Fiyat Yönü Tahmini")

if varliklar is None or tam_gecmis_veri is None:
    st.error("Uygulama başlatılamadı. Lütfen yukarıdaki hata mesajlarını kontrol edin ve gerekli dosyaların mevcut olduğundan emin olun.")
    if tam_gecmis_veri is not None and len(tam_gecmis_veri) < GEREKLI_GECMIS_GUN_SAYISI:
         st.error(f"Geçmiş veri dosyasında {len(tam_gecmis_veri)} satır var ancak indikatör hesaplamaları için en az {GEREKLI_GECMIS_GUN_SAYISI} gün verisi önerilir.")
    st.stop()

model = varliklar["model"]
olcekleyici = varliklar["olcekleyici"]
# --- Değişken adı güncellendi ---
RFE_SECILMIS_OZNITELIKLER = varliklar["rfe_oznitelik_listesi"]
# --- Değişken adı güncellendi sonu ---
OLCEKLEYICI_GIRIS_SUTUNLARI = varliklar["olcekleyici_sutun_listesi"]

st.sidebar.header("Güncel Günün Verilerini Girin")
st.sidebar.markdown("Lütfen tahmin yapmak istediğiniz günün Açılış (Open), Kapanış (Close) ve Hacim (Volume) değerlerini girin.")

varsayilan_acilis = tam_gecmis_veri['Open'].iloc[-1] if not tam_gecmis_veri.empty else 40000.0
varsayilan_kapanis = tam_gecmis_veri['Close'].iloc[-1] if not tam_gecmis_veri.empty else 40500.0
varsayilan_hacim = tam_gecmis_veri['Volume'].iloc[-1] if not tam_gecmis_veri.empty else 50000000000.0

guncel_acilis = st.sidebar.number_input("Bugünkü Açılış (Open)", value=float(varsayilan_acilis), step=100.0, format="%.2f")
guncel_kapanis = st.sidebar.number_input("Bugünkü Kapanış (Close)", value=float(varsayilan_kapanis), step=100.0, format="%.2f")
guncel_hacim = st.sidebar.number_input("Bugünkü Hacim (Volume)", value=float(varsayilan_hacim), step=1000000.0, format="%.0f")

if st.sidebar.button("Tahmin Et", type="primary"):
    if len(tam_gecmis_veri) < GEREKLI_GECMIS_GUN_SAYISI -1 :
         st.error(f"Tahmin için yetersiz geçmiş veri. En az {GEREKLI_GECMIS_GUN_SAYISI-1} geçmiş gün verisi gereklidir. Mevcut: {len(tam_gecmis_veri)}")
    else:
        try:
            son_gecmis = tam_gecmis_veri[['Open', 'Close', 'Volume']].iloc[-(GEREKLI_GECMIS_GUN_SAYISI - 1):].copy()

            guncel_veri_sozlugu = {
                'Open': [guncel_acilis],
                'Close': [guncel_kapanis],
                'Volume': [guncel_hacim]
            }
            guncel_df_satiri = pd.DataFrame(guncel_veri_sozlugu)

            birlestirilmis_df = pd.concat([son_gecmis, guncel_df_satiri], ignore_index=True)
            if len(birlestirilmis_df) < GEREKLI_GECMIS_GUN_SAYISI:
                st.error(f"Birleştirilmiş veri {len(birlestirilmis_df)} satır içeriyor, ancak {GEREKLI_GECMIS_GUN_SAYISI} gerekli. Bu bir hata olmalı.")
                st.stop()

            indikatorlu_df = teknik_indikatorleri_hesapla(birlestirilmis_df)

            if indikatorlu_df.empty:
                st.error("Teknik indikatörler hesaplanamadı. Lütfen girdi verilerini kontrol edin.")
                st.stop()

            tahmin_icin_guncel_oznitelikler = indikatorlu_df.iloc[-1:].copy()

            eksik_olcekleyici_sutunlari = [sutun for sutun in OLCEKLEYICI_GIRIS_SUTUNLARI if sutun not in tahmin_icin_guncel_oznitelikler.columns]
            if eksik_olcekleyici_sutunlari:
                st.error(f"İndikatör hesaplaması sonrası ölçekleyici için beklenen sütunlar eksik: {', '.join(eksik_olcekleyici_sutunlari)}. "
                         f"Mevcut sütunlar: {tahmin_icin_guncel_oznitelikler.columns.tolist()}")
                st.stop()

            olceklenecek_df = tahmin_icin_guncel_oznitelikler[OLCEKLEYICI_GIRIS_SUTUNLARI]

            if olceklenecek_df.isnull().values.any():
                 st.error("Hesaplanan son gün verilerinde (indikatörler dahil) NaN değerler bulundu. Bu genellikle yetersiz geçmiş veriden veya hesaplama hatasından kaynaklanır. Lütfen verileri kontrol edin.")
                 st.write("Hesaplanan son satır (indikatörler dahil, ölçekleme öncesi, NaN içeren):")
                 st.dataframe(olceklenecek_df)
                 st.info(f"Not: İndikatör hesaplamaları için en az {GEREKLI_GECMIS_GUN_SAYISI} gün verisi (geçmiş + güncel) gereklidir. "
                         f"Kullanılan birleştirilmiş veri {len(birlestirilmis_df)} satırdan oluşuyor.")
                 st.stop()

            olceklenmis_oznitelikler_dizisi = olcekleyici.transform(olceklenecek_df)
            olceklenmis_oznitelikler_df = pd.DataFrame(olceklenmis_oznitelikler_dizisi, columns=OLCEKLEYICI_GIRIS_SUTUNLARI, index=olceklenecek_df.index)

            # --- Model için doğru özellikleri seçme güncellendi ---
            eksik_model_sutunlari = [sutun for sutun in RFE_SECILMIS_OZNITELIKLER if sutun not in olceklenmis_oznitelikler_df.columns]
            if eksik_model_sutunlari:
                st.error(f"Ölçeklenmiş veride model için beklenen özellikler eksik: {', '.join(eksik_model_sutunlari)}")
                st.stop()

            model_icin_son_oznitelikler = olceklenmis_oznitelikler_df[RFE_SECILMIS_OZNITELIKLER]
            # --- Model için özellik seçimi sonu ---


            tahmin = model.predict(model_icin_son_oznitelikler)
            model.predict_proba(model_icin_son_oznitelikler) # Olasılıkları almak isterseniz

            st.subheader("Tahmin Sonucu")
            sutun1, sutun2 = st.columns([1,3])
            with sutun1:
                if tahmin[0] == 1:
                    st.image("https://emojigraph.org/media/apple/chart-increasing_1f4c8.png", width=100)
                else:
                    st.image("https://emojigraph.org/media/apple/chart-decreasing_1f4c9.png", width=100)
            with sutun2:
                if tahmin[0] == 1:
                    st.success("📈 **YÜKSELİŞ** bekleniyor.")
                    st.markdown("Tahminimiz, yarınki Bitcoin kapanış fiyatının bugünkü kapanış fiyatından **daha yüksek** olacağı yönündedir.")
                else:
                    st.error("📉 **DÜŞÜŞ** bekleniyor.")
                    st.markdown("Tahminimiz, yarınki Bitcoin kapanış fiyatının bugünkü kapanış fiyatından **daha düşük** olacağı yönündedir.")
                    if hasattr(model, 'predict_proba'):
                     olasiliklar = model.predict_proba(model_icin_son_oznitelikler)
                     st.write(f"Model Güveni (Yükseliş olasılığı): {olasiliklar[0][1]:.2%}")


            with st.expander("Detaylar: Modele Giren Veriler"):
                st.markdown("##### Ham Girilen Veriler (Bugün):")
                st.dataframe(guncel_df_satiri.style.format({"Open": "{:.2f}", "Close": "{:.2f}", "Volume": "{:.0f}"}))
                st.markdown(f"##### {GEREKLI_GECMIS_GUN_SAYISI} Günlük Veri Üzerinden Hesaplanan İndikatörler (Bugün İçin, Ölçekleme Öncesi):")
                st.dataframe(olceklenecek_df.style.format("{:.2f}"))
                st.markdown("##### Ölçeklenmiş ve Model İçin Seçilmiş Özellikler:")
                st.dataframe(model_icin_son_oznitelikler.style.format("{:.4f}"))
                # --- Caption güncellendi ---
                st.caption(f"Modelin kullandığı özellikler: ` {', '.join(RFE_SECILMIS_OZNITELIKLER)} `")
                # --- Caption sonu ---

        except ValueError as ve:
            st.error(f"Veri hatası: {ve}")
        except Exception as e:
            st.error(f"Tahmin sırasında beklenmedik bir hata oluştu: {e}")
            st.error("Lütfen girdiğiniz değerleri ve geçmiş veri dosyasının bütünlüğünü kontrol edin.")
            import traceback
            st.text(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.caption("Bu uygulama demo amaçlıdır ve yatırım tavsiyesi değildir.")

if tam_gecmis_veri is not None and not tam_gecmis_veri.empty:
    st.markdown("---")
    st.subheader(f"Geçmiş Veri Örneği (Son 5 Gün - Toplam {len(tam_gecmis_veri)} gün mevcut)")
    gosterilecek_df = tam_gecmis_veri[['Open', 'Close', 'Volume']].tail().reset_index(drop=True)
    st.dataframe(gosterilecek_df.style.format({"Open": "{:.2f}", "Close": "{:.2f}", "Volume": "{:.0f}"}))
else:
    st.markdown("---")
    st.warning("Geçmiş veri yüklenemedi veya boş.")