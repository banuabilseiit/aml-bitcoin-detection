import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import json

# ---- Настройки страницы ----
st.set_page_config(
    page_title="AML Monitor | RegTech",
    page_icon="🔍",
    layout="wide"
)

# ---- Стили ----
st.markdown("""
<style>
    /* Фон приложения */
    .stApp { background-color: #0a0e1a; }

    /* Все тексты белые и яркие */
    html, body, [class*="css"], p, span, label, div {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Заголовки */
    h1 { color: #00d4ff !important; font-size: 2.2em !important; font-weight: 800 !important; }
    h2 { color: #ffffff !important; font-weight: 700 !important; }
    h3 { color: #00d4ff !important; font-weight: 700 !important; }
    h4, h5 { color: #ffffff !important; }

    /* Слайдеры — подписи */
    .stSlider label { color: #ffffff !important; font-size: 1em !important; font-weight: 600 !important; }
    .stSlider .st-ae { color: #00d4ff !important; }

    /* Кнопка */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0066ff) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1em !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px !important;
        letter-spacing: 1px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0066ff, #00d4ff) !important;
        transform: scale(1.02);
    }

    /* Метрики */
    [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8em !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 1em !important; font-weight: 600 !important; }
    [data-testid="stMetricDelta"] { color: #00ff88 !important; font-weight: 600 !important; }

    /* Карточки результата */
    .risk-high {
        background: linear-gradient(135deg, #4a0000, #8b0000);
        border: 2px solid #ff3333;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,50,50,0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #003300, #006600);
        border: 2px solid #00ff88;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,255,100,0.3);
    }

    /* Панели параметров */
    .param-box {
        background: linear-gradient(135deg, #111827, #1e2a3a);
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 10px;
    }

    /* Разделитель */
    hr { border-color: #00d4ff !important; opacity: 0.3 !important; }

    /* Info/Warning/Success блоки */
    .stAlert { background-color: #111827 !important; border-left: 4px solid #00d4ff !important; }
    .stAlert p { color: #ffffff !important; font-weight: 500 !important; }

    /* Caption */
    .stCaption { color: #aaaaaa !important; }

    /* Прогресс бар */
    .stProgress > div > div { background: linear-gradient(90deg, #00ff88, #00d4ff, #ff3333) !important; }
</style>
""", unsafe_allow_html=True)

# ---- Загрузка модели ----
@st.cache_resource
def load_model():
    try:
        return joblib.load('aml_xgboost_model.pkl')
    except:
        return None

# ---- Загрузка диапазонов ----
@st.cache_data
def load_ranges():
    try:
        with open('feature_ranges.json') as f:
            return json.load(f)
    except:
        return None

model  = load_model()
ranges = load_ranges()

def get_range(col):
    if ranges and col in ranges:
        return ranges[col]['min'], ranges[col]['max'], ranges[col]['default']
    return -10.0, 10.0, 0.0

# ---- Шапка ----
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1>🔍 AML Transaction Monitor</h1>
    <p style='color:#00d4ff; font-size:1.1em; font-weight:600; letter-spacing:2px;'>
        REGTECH · BITCOIN · FINANCIAL SUPERVISION
    </p>
</div>
""", unsafe_allow_html=True)

# Статус строка
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<p style='text-align:center; color:#00ff88; font-weight:700;'>✅ XGBoost Model</p>", unsafe_allow_html=True)
with c2:
    st.markdown("<p style='text-align:center; color:#00d4ff; font-weight:700;'>📊 ROC-AUC 0.9977</p>", unsafe_allow_html=True)
with c3:
    status = "✅ Ranges Loaded" if ranges else "⚠️ Default Ranges"
    color  = "#00ff88" if ranges else "#ffaa00"
    st.markdown(f"<p style='text-align:center; color:{color}; font-weight:700;'>{status}</p>", unsafe_allow_html=True)

st.divider()

# ---- Основной интерфейс ----
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### ⚙️ Параметры транзакции")
    st.markdown("<p style='color:#aaaaaa;'>Настройте значения и нажмите Оценить</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#00d4ff; font-weight:700; font-size:1em; text-transform:uppercase; letter-spacing:1px;'>📌 Локальные параметры</p>", unsafe_allow_html=True)

    mn, mx, dv = get_range('local_feat_5')
    feat_5  = st.slider("local_feat_5 — основной параметр",  mn, mx, dv)

    mn, mx, dv = get_range('local_feat_53')
    feat_53 = st.slider("local_feat_53 — параметр суммы",    mn, mx, dv)

    mn, mx, dv = get_range('local_feat_59')
    feat_59 = st.slider("local_feat_59 — параметр комиссии", mn, mx, dv)

    mn, mx, dv = get_range('local_feat_2')
    feat_2  = st.slider("local_feat_2 — входы транзакции",   mn, mx, dv)

    mn, mx, dv = get_range('local_feat_55')
    feat_55 = st.slider("local_feat_55 — параметр выходов",  mn, mx, dv)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#00d4ff; font-weight:700; font-size:1em; text-transform:uppercase; letter-spacing:1px;'>📊 Агрегированные параметры</p>", unsafe_allow_html=True)

    mn, mx, dv = get_range('agg_feat_32')
    agg_32  = st.slider("agg_feat_32 — агрегированный 1",    mn, mx, dv)

    mn, mx, dv = get_range('agg_feat_49')
    agg_49  = st.slider("agg_feat_49 — агрегированный 2",    mn, mx, dv)

    st.markdown("<br>", unsafe_allow_html=True)
    evaluate_btn = st.button("🔎 ОЦЕНИТЬ ТРАНЗАКЦИЮ", use_container_width=True, type="primary")

with right_col:
    st.markdown("### 📊 Результат оценки")
    st.markdown("<br>", unsafe_allow_html=True)

    if evaluate_btn:
        if model is None:
            st.error("❌ Модель не найдена. Проверьте файл aml_xgboost_model.pkl")
        else:
            features = np.zeros(165)
            features[4]   = feat_5
            features[52]  = feat_53
            features[58]  = feat_59
            features[1]   = feat_2
            features[54]  = feat_55
            features[124] = agg_32
            features[140] = agg_49

            proba    = model.predict_proba(features.reshape(1, -1))[0][1]
            pred     = model.predict(features.reshape(1, -1))[0]
            risk_pct = proba * 100

            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <p style="color:#ff9999; font-size:1em; font-weight:700; letter-spacing:2px;">СТАТУС ТРАНЗАКЦИИ</p>
                    <h2 style="color:#ff3333; font-size:2em;">🚨 ПОДОЗРИТЕЛЬНАЯ</h2>
                    <h1 style="color:#ffffff; font-size:3.5em; font-weight:900;">{risk_pct:.1f}%</h1>
                    <p style="color:#ffaaaa; font-weight:600;">Риск-скор транзакции</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("⚠️ Транзакция направлена на проверку комплаенс-офицеру")
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <p style="color:#99ffbb; font-size:1em; font-weight:700; letter-spacing:2px;">СТАТУС ТРАНЗАКЦИИ</p>
                    <h2 style="color:#00ff88; font-size:2em;">✅ ЛЕГИТИМНАЯ</h2>
                    <h1 style="color:#ffffff; font-size:3.5em; font-weight:900;">{risk_pct:.1f}%</h1>
                    <p style="color:#aaffcc; font-weight:600;">Риск-скор транзакции</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("✅ Транзакция прошла автоматический AML-скрининг")

            # Уровень риска
            st.markdown("<br>", unsafe_allow_html=True)
            color_label = "🔴 ВЫСОКИЙ" if risk_pct > 50 else ("🟡 СРЕДНИЙ" if risk_pct > 30 else "🟢 НИЗКИЙ")
            st.markdown(f"<p style='color:#ffffff; font-weight:700; font-size:1.1em;'>Уровень риска: {color_label} — {risk_pct:.1f}%</p>", unsafe_allow_html=True)
            st.progress(float(proba))

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={'projection': 'polar'})
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0a0e1a')

            theta = np.linspace(0, np.pi, 100)
            ax.plot(theta, [1]*100, color='#1e2a3a', linewidth=18, solid_capstyle='round')

            risk_theta  = np.linspace(0, np.pi * proba, 100)
            gauge_color = '#ff3333' if proba > 0.5 else ('#ffaa00' if proba > 0.3 else '#00ff88')
            ax.plot(risk_theta, [1]*100, color=gauge_color, linewidth=18, solid_capstyle='round')

            needle_theta = np.pi * proba
            ax.annotate('', xy=(needle_theta, 0.92), xytext=(needle_theta, 0.05),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

            ax.set_ylim(0, 1.3)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            ax.set_xticks([0, np.pi/2, np.pi])
            ax.set_xticklabels(['0%', '50%', '100%'], color='#aaaaaa', fontsize=9)
            ax.set_yticks([])
            ax.spines['polar'].set_visible(False)
            ax.set_title(f'Risk Score: {risk_pct:.1f}%', color='white', pad=15, fontsize=13, fontweight='bold')

            st.pyplot(fig)
            plt.close()

    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #111827, #1e2a3a); border: 1px solid #00d4ff; border-radius:12px; padding:24px;'>
            <p style='color:#00d4ff; font-weight:700; font-size:1.1em;'>Как пользоваться:</p>
            <p style='color:#ffffff;'>1. Настройте слайдеры с параметрами транзакции</p>
            <p style='color:#ffffff;'>2. Нажмите кнопку <b>ОЦЕНИТЬ ТРАНЗАКЦИЮ</b></p>
            <p style='color:#ffffff;'>3. Модель выдаст риск-скор и решение</p>
            <br>
            <p style='color:#00d4ff; font-weight:700;'>Интерпретация риск-скора:</p>
            <p style='color:#00ff88; font-weight:600;'>🟢 Менее 30% — легитимная транзакция</p>
            <p style='color:#ffaa00; font-weight:600;'>🟡 30–50% — требует внимания</p>
            <p style='color:#ff3333; font-weight:600;'>🔴 Более 50% — подозрительная, направляется на проверку</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ---- Метрики модели ----
st.markdown("<h3 style='color:#00d4ff; text-align:center;'>📈 Характеристики модели</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("ROC-AUC", "0.9977", "↑ vs baseline 0.97")
with m2:
    st.metric("F1-score", "0.96", "↑ vs baseline 0.72")
with m3:
    st.metric("Recall", "93%", "845 из 909 поймано")
with m4:
    st.metric("Ложные тревоги", "8", "из 8,404 легитимных")

st.divider()
st.markdown("<p style='text-align:center; color:#555555;'>🏛️ RegTech | AML-мониторинг цифровых активов | Elliptic Bitcoin Dataset | FATF Recommendation 16</p>", unsafe_allow_html=True)
