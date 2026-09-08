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

st.set_page_config(page_title="Compliance Outcome Predictor", layout="wide")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = WORK_DIR
os.makedirs(WEB_DIR, exist_ok=True)

# =========【关键！此处必须和训练XGBoost时feature_names完全一模一样，复制你训练时的特征列表】=========
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

@st.cache_resource
def load_model():
    model_path = os.path.join(WEB_DIR, "XGBoost_model.pkl")
    model = joblib.load(model_path)
    return model

model = load_model()

# ========== 左右分栏：左侧输入纵向排列，右侧结果 ==========
col_left, col_right = st.columns([0.40, 0.60])

with col_left:
    st.markdown("### Please enter the patient's details")
    with st.form("pred_form"):
        Cr = st.number_input("Cr", min_value=0.0, max_value=300.0, value=60.0, step=0.1)
        AGE = st.number_input("AGE", min_value=18, max_value=110, value=60)
        CRRT = st.selectbox("CRRT", options=["No", "Yes"])
        BUN = st.number_input("BUN", min_value=0.0, max_value=150.0, value=10.0, step=0.1)
        vein_Total_daily_dose = st.number_input("vein-Total daily dose", min_value=0.0, max_value=5000.0, value=300.0, step=1.0)
        BMI = st.number_input("BMI", min_value=12.0, max_value=50.0, value=24.0, step=0.1)
        PLT = st.number_input("PLT", min_value=10, max_value=600, value=200, step=1)
        CrCL = st.number_input("CrCL", min_value=0.0, max_value=200.0, value=60.0, step=0.1)
        TP = st.number_input("TP", min_value=30.0, max_value=90.0, value=65.0, step=0.1)
        TBIL = st.number_input("TBIL", min_value=0.0, max_value=200.0, value=12.0, step=0.1)

        submit_btn = st.form_submit_button("Predict")

with col_right:
    st.markdown("# Compliance Outcome Predictor")
    st.markdown("")

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
        # 构建DataFrame，强制对齐模型特征顺序
        input_df = pd.DataFrame([input_values], columns=feature_cols).reindex(columns=model.get_booster().feature_names)

        pred_proba = model.predict_proba(input_df)[0]
        prob_yes = pred_proba[0]  # Yes = Compliant
        prob_no  = pred_proba[1]  # No = Non‑compliant
        pred_is_no = 1 if prob_no > THRESHOLD else 0

        # ---------------- 1. Predicted Result ----------------
        st.markdown("### 🧪 Predicted Result")
        if pred_is_no == 1:
            st.markdown(f"Final outcome: **NO (Non‑compliant)**")
            st.markdown(f"Probability of non‑compliance = {prob_no:.2%}")
        else:
            st.markdown(f"Final outcome: **YES (Compliant)**")
            st.markdown(f"Probability of compliance = {prob_yes:.2%}")
        st.markdown(f"Model threshold: {THRESHOLD}")

        shap_explainer = shap.TreeExplainer(model)
        exp = shap_explainer(input_df, output=0)
        base_val = exp[0].base_value
        sum_shap = np.sum(exp[0].values)
        f_x = base_val + sum_shap

        # ---------------- 2. SHAP汇总加和结果（阈值下方，无单特征） ----------------
        st.markdown("**SHAP summary (logit scale):**")
        st.markdown(f"$E[f(X)]$ (Base value) = {base_val:.4f} · Sum of SHAP values = {sum_shap:.4f} · $f(x)$ = {f_x:.4f}")

        # ---------------- 3. SHAP Waterfall Plot 保持原图样式 ----------------
        st.markdown("### 🔍 SHAP Waterfall Plot")
        plt.figure(figsize=(10, 6), dpi=120)
        shap.plots.waterfall(exp[0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        # ---------------- 4. 图下解释文字 ----------------
        st.markdown("""
> *Interpretation: SHAP values are on the logit scale.
> Positive values increase the log‑odds of compliance (YES);
> negative values decrease the log‑odds of compliance (YES).*
""")

    else:
        st.markdown("Please fill in patient information and click Predict button.")

st.divider()
st.markdown("""
> **Disclaimer**: This tool is for research demonstration only and does not replace clinical judgment.
""")
