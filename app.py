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
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .risk-high {
        background: linear-gradient(135deg, #3d1515, #5c1f1f);
        border: 1px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #153d1e, #1f5c2e);
        border: 1px solid #2ecc71;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---- Загрузка модели ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load('aml_xgboost_model.pkl')
        return model
    except:
        return None

# ---- Загрузка диапазонов из JSON ----
@st.cache_data
def load_ranges():
    try:
        with open('feature_ranges.json') as f:
            return json.load(f)
    except:
        return None

model  = load_model()
ranges = load_ranges()

# ---- Функция получения диапазона ----
def get_range(col):
    if ranges and col in ranges:
        return ranges[col]['min'], ranges[col]['max'], ranges[col]['default']
    return -10.0, 10.0, 0.0

# ---- Шапка ----
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🔍 AML Transaction Monitor")
    st.markdown("##### RegTech | Bitcoin Transaction Risk Scoring System")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Elliptic Bitcoin Dataset**  \nXGBoost · ROC-AUC 0.998")

if ranges:
    st.success("✅ Диапазоны признаков загружены из датасета")
else:
    st.warning("⚠️ feature_ranges.json не найден — используются значения по умолчанию")

st.divider()

# ---- Основной интерфейс ----
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### ⚙️ Параметры транзакции")
    st.markdown("Настройте параметры и нажмите **Оценить**")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Локальные параметры транзакции**")

    mn, mx, dv = get_range('local_feat_5')
    feat_5  = st.slider("local_feat_5  — основной параметр", mn, mx, dv)

    mn, mx, dv = get_range('local_feat_53')
    feat_53 = st.slider("local_feat_53 — параметр суммы",    mn, mx, dv)

    mn, mx, dv = get_range('local_feat_59')
    feat_59 = st.slider("local_feat_59 — параметр комиссии", mn, mx, dv)

    mn, mx, dv = get_range('local_feat_2')
    feat_2  = st.slider("local_feat_2  — входы транзакции",  mn, mx, dv)

    mn, mx, dv = get_range('local_feat_55')
    feat_55 = st.slider("local_feat_55 — параметр выходов",  mn, mx, dv)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Агрегированные параметры**")

    mn, mx, dv = get_range('agg_feat_32')
    agg_32  = st.slider("agg_feat_32 — агрегированный 1",    mn, mx, dv)

    mn, mx, dv = get_range('agg_feat_49')
    agg_49  = st.slider("agg_feat_49 — агрегированный 2",    mn, mx, dv)

    st.markdown("<br>", unsafe_allow_html=True)
    evaluate_btn = st.button("🔎 Оценить транзакцию", use_container_width=True, type="primary")

with right_col:
    st.markdown("### 📊 Результат оценки")
    st.markdown("<br>", unsafe_allow_html=True)

    if evaluate_btn:
        if model is None:
            st.error("❌ Модель не найдена. Убедитесь что файл `aml_xgboost_model.pkl` находится в той же папке.")
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
                    <h2>🚨 ПОДОЗРИТЕЛЬНАЯ</h2>
                    <h1 style="color:#e74c3c; font-size:3em">{risk_pct:.1f}%</h1>
                    <p style="color:#aaa">Риск-скор транзакции</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("⚠️ Транзакция направлена на проверку комплаенс-офицеру")
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>✅ ЛЕГИТИМНАЯ</h2>
                    <h1 style="color:#2ecc71; font-size:3em">{risk_pct:.1f}%</h1>
                    <p style="color:#aaa">Риск-скор транзакции</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("✅ Транзакция прошла автоматический AML-скрининг")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Уровень риска:**")
            color = "🔴" if risk_pct > 50 else ("🟡" if risk_pct > 30 else "🟢")
            st.markdown(f"{color} {risk_pct:.1f}%")
            st.progress(float(proba))

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw={'projection': 'polar'})
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')

            theta = np.linspace(0, np.pi, 100)
            ax.plot(theta, [1]*100, color='#2d3250', linewidth=15, solid_capstyle='round')

            risk_theta  = np.linspace(0, np.pi * proba, 100)
            gauge_color = '#e74c3c' if proba > 0.5 else ('#f39c12' if proba > 0.3 else '#2ecc71')
            ax.plot(risk_theta, [1]*100, color=gauge_color, linewidth=15, solid_capstyle='round')

            needle_theta = np.pi * proba
            ax.annotate('', xy=(needle_theta, 0.95), xytext=(needle_theta, 0.0),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))

            ax.set_ylim(0, 1.3)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['polar'].set_visible(False)
            ax.set_title(f'Risk Score: {risk_pct:.1f}%', color='white', pad=10, fontsize=12)
            st.pyplot(fig)
            plt.close()

    else:
        st.info("👈 Настройте параметры транзакции и нажмите **Оценить транзакцию**")
        st.markdown("""
        **Как пользоваться:**
        1. Настройте слайдеры с параметрами транзакции
        2. Нажмите кнопку **Оценить транзакцию**
        3. Модель выдаст риск-скор и решение

        **Интерпретация:**
        - 🟢 < 30% — легитимная транзакция
        - 🟡 30–50% — требует внимания
        - 🔴 > 50% — подозрительная, отправляется на проверку
        """)

st.divider()

# ---- Информация о модели ----
st.markdown("### 📈 Характеристики модели")
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
st.caption("🏛️ RegTech | AML-мониторинг цифровых активов | Elliptic Bitcoin Dataset | FATF Recommendation 16")
