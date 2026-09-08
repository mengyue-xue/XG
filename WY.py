import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# st.set_page_config 必须放在所有streamlit调用最前面
st.set_page_config(page_title="重症结局风险预测模型", layout="wide")

# ===================== 全局配置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = WORK_DIR
os.makedirs(WEB_DIR, exist_ok=True)

feature_cols = [
    'Cr',
    'AGE',
    'CRRT',
    'BUN',
    'vein-Total daily dose',
    'BMI',
    'PLT',
    'CrCL',
    'TP',
    'TBIL'
]

THRESHOLD = 0.636

# ===================== 加载模型 =====================
@st.cache_resource
def load_model():
    model_path = os.path.join(WEB_DIR, "XGBoost_model.pkl")
    model = joblib.load(model_path)
    return model

model = load_model()

# ===================== 网页正文 =====================
st.title("重症结局风险预测模型")

st.markdown("""
本模型基于XGBoost算法，用于预测重症患者结局发生风险。
输入特征共10项: Cr、AGE、CRRT、vein‑Total daily dose、BUN、BMI、PLT、CrCL、TP、TBIL.
""")
st.divider()

# ===================== 输入表单 =====================
st.subheader("患者特征输入")
with st.form("pred_form"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        Cr = st.number_input("Cr", min_value=0.0, max_value=300.0, value=60.0, step=0.1)
        AGE = st.number_input("AGE", min_value=18, max_value=110, value=60)
    with col2:
        CRRT = st.selectbox("CRRT", options=["No", "Yes"])
        BUN = st.number_input("BUN", min_value=0.0, max_value=150.0, value=10.0, step=0.1)
    with col3:
        vein_Total_daily_dose = st.number_input("vein‑Total daily dose", min_value=0.0, max_value=5000.0, value=300.0, step=1.0)
        BMI = st.number_input("BMI", min_value=12.0, max_value=50.0, value=24.0, step=0.1)
    with col4:
        PLT = st.number_input("PLT", min_value=10, max_value=600, value=200, step=1)
        CrCL = st.number_input("CrCL", min_value=0.0, max_value=200.0, value=60.0, step=0.1)
    with col5:
        TP = st.number_input("TP", min_value=30.0, max_value=90.0, value=65.0, step=0.1)
        TBIL = st.number_input("TBIL", min_value=0.0, max_value=200.0, value=12.0, step=0.1)

    submit_btn = st.form_submit_button("Predict 预测")

if submit_btn:
    crrt_val = 1 if CRRT == "Yes" else 0
    input_values = [
        Cr,
        AGE,
        crrt_val,
        BUN,
        vein_Total_daily_dose,
        BMI,
        PLT,
        CrCL,
        TP,
        TBIL
    ]
    input_df = pd.DataFrame([input_values], columns=feature_cols)

    pred_proba = model.predict_proba(input_df)[0]
    prob_pos = pred_proba[1]
    prob_neg = pred_proba[0]
    pred_class = 1 if prob_pos > THRESHOLD else 0

    st.divider()
    st.subheader("📊预测结果")
    st.write(f"结局发生概率(Yes): **{prob_pos:.2%}**")
    st.write(f"结局不发生概率(No): **{prob_neg:.2%}**")
    st.write(f"模型最优阈值: {THRESHOLD}")

    if pred_class == 1:
        st.error("最终判定: **YES**")
        tip = f"患者结局发生风险较高，当前预测概率 {prob_pos:.1%}，建议密切监测，实施个体化干预。"
    else:
        st.success("最终判定: **NO**")
        tip = f"患者结局发生风险较低，当前预测概率 {prob_pos:.1%}，仍需常规临床随访观察。"
    st.info(tip)

    st.markdown("---")
    st.subheader("SHAP Waterfall Plot‑XGBoost")
    shap_explainer = shap.TreeExplainer(model)
    # ⭐关键：获取完整Explanation对象（包含values/base_values/data），不能只取shap_values数组
    exp = shap_explainer(input_df)
    plt.figure(figsize=(12,9))
    shap.plots.waterfall(exp[0], max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf(), dpi=300)
    plt.close()

st.divider()
st.markdown("""
> **Disclaimer**: This tool is for research demonstration only and does not replace clinical judgment. Clinical decisions should be comprehensively evaluated according to the patient's actual condition.
""")
