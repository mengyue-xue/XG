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
本模型基于XGBoost算法，用于预测重症患者达标风险。
> 标签定义：**Yes = 达标，No = 不达标**
输入特征共10项: Cr、AGE、CRRT、BUN、vein‑Total daily dose、BMI、PLT、CrCL、TP、TBIL.
""")
st.divider()

# 页面均分两列：左输入，右结果
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("患者特征输入")
    with st.form("pred_form"):
        Cr = st.number_input("Cr", min_value=0.0, max_value=300.0, value=60.0, step=0.1)
        AGE = st.number_input("AGE", min_value=18, max_value=110, value=60)
        CRRT = st.selectbox("CRRT", options=["No", "Yes"])
        BUN = st.number_input("BUN", min_value=0.0, max_value=150.0, value=10.0, step=0.1)
        vein_Total_daily_dose = st.number_input("vein‑Total daily dose", min_value=0.0, max_value=5000.0, value=300.0, step=1.0)
        BMI = st.number_input("BMI", min_value=12.0, max_value=50.0, value=24.0, step=0.1)
        PLT = st.number_input("PLT", min_value=10, max_value=600, value=200, step=1)
        CrCL = st.number_input("CrCL", min_value=0.0, max_value=200.0, value=60.0, step=0.1)
        TP = st.number_input("TP", min_value=30.0, max_value=90.0, value=65.0, step=0.1)
        TBIL = st.number_input("TBIL", min_value=0.0, max_value=200.0, value=12.0, step=0.1)

        submit_btn = st.form_submit_button("Predict 预测")

with col_right:
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
        input_df = input_df.reindex(columns=model.get_booster().feature_names)

        pred_proba = model.predict_proba(input_df)[0]
        prob_yes = pred_proba[0]
        prob_no  = pred_proba[1]

        pred_is_no = 1 if prob_no > THRESHOLD else 0

        st.subheader("📊预测结果")
        st.write(f"**达标(Yes)概率**: {prob_yes:.2%}")
        st.write(f"**不达标(No)概率**: {prob_no:.2%}")
        st.write(f"模型最优阈值: {THRESHOLD}")

        if pred_is_no == 1:
            st.error("最终判定: **NO（不达标）**")
            tip = f"患者不达标风险较高，当前不达标概率 {prob_no:.1%}，建议密切监测，实施个体化干预。"
        else:
            st.success("最终判定: **YES（达标）**")
            tip = f"患者达标可能性高，当前达标概率 {prob_yes:.1%}，仍需常规临床随访观察。"
        st.info(tip)

        # =========修复shap调用，移除output=0参数=========
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer(input_df)
        # 取类别0（Yes‑达标）的shap值
        exp0 = shap.Explanation(
            values=shap_values.values[:,:,0],
            base_values=shap_values.base_values[:,0],
            data=shap_values.data,
            feature_names=shap_values.feature_names
        )

        base_val = exp0.base_values[0]
        sum_shap = np.sum(exp0.values[0])
        f_x = base_val + sum_shap

        st.markdown("**SHAP summary (logit scale):**")
        st.markdown(f"$E[f(X)]$ (Base value) = {base_val:.4f} · Sum of SHAP values = {sum_shap:.4f} · $f(x)$ = {f_x:.4f}")

        st.markdown("---")
        st.subheader("SHAP Waterfall Plot‑XGBoost（解释：Yes‑达标）")
        plt.figure(figsize=(12,9))
        shap.plots.waterfall(exp0[0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), dpi=300)
        plt.close()

st.divider()
st.markdown("""
> **说明**: SHAP瀑布图用于解释【Yes‑达标】；红色条代表该特征**提升达标概率**，蓝色条代表该特征**降低达标概率**。E[f(X)]为模型基线期望输出，f(x)为该患者样本最终模型输出。
>
> **Disclaimer**: This tool is for research demonstration only and does not replace clinical judgment. Clinical decisions should be comprehensively evaluated according to the patient's actual condition.
""")
