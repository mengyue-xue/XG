import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from lime.lime_tabular import LimeTabularExplainer
import shap

# ===================== 全局配置：相对路径 =====================
plt.rcParams['font.family'] = 'Times New Roman'
RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = WORK_DIR
os.makedirs(WEB_DIR, exist_ok=True)

# 特征列表，务必与模型训练完全一致
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

# ===================== 加载模型 + LIME背景数据集 val_intersect_lasso_boruta.csv =====================
@st.cache_resource
def load_model_and_lime_bg():
    model_path = os.path.join(WEB_DIR, "XGBoost_model.pkl")
    model = joblib.load(model_path)

    lime_bg_path = os.path.join(WEB_DIR, "val_intersect_lasso_boruta.csv")
    lime_bg_df = pd.read_csv(lime_bg_path)

    if "CRRT" in lime_bg_df.columns:
        if pd.api.types.is_object_dtype(lime_bg_df["CRRT"]):
            lime_bg_df["CRRT"] = (
                lime_bg_df["CRRT"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
            )
    return model, lime_bg_df

model, lime_bg_data = load_model_and_lime_bg()

# ===================== 网页基础设置 =====================
st.set_page_config(page_title="重症结局风险预测模型", layout="wide")
st.title("重症患者结局风险预测模型")

st.markdown("""
本模型基于XGBoost算法，用于预测重症患者结局发生风险。
输入特征共10项：Cr、AGE、CRRT、vein‑Total daily dose、BUN、BMI、PLT、CrCL、TP、TBIL。
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
        vein_Total_daily_dose = st.number_input("vein‑Total daily dose", min_value=0.0, max_value=5000.0, value=1000.0, step=1.0)
        BMI = st.number_input("BMI", min_value=12.0, max_value=50.0, value=24.0, step=0.1)
    with col4:
        PLT = st.number_input("PLT", min_value=10, max_value=600, value=200)
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
    st.write(f"模型最优阈值：{THRESHOLD}")

    if pred_class == 1:
        st.error("最终判定：**YES**")
        tip = f"患者结局发生风险较高，当前预测概率 {prob_pos:.1%}，建议密切监测，实施个体化干预。"
    else:
        st.success("最终判定：**NO**")
        tip = f"患者结局发生风险较低，当前预测概率 {prob_pos:.1%}，仍需常规临床随访观察。"
    st.info(tip)

    st.markdown("---")
    # SHAP Force Plot：改用 st.pyplot，去掉BytesIO规避DOM报错
    st.subheader("SHAP Force Plot Explanation")
    shap_explainer = shap.TreeExplainer(model)
    shap_vals = shap_explainer.shap_values(input_df)
    fig, ax = plt.subplots(figsize=(14, 4))
    shap.force_plot(
        shap_explainer.expected_value,
        shap_vals[0],
        input_df,
        matplotlib=True,
        ax=ax
    )
    st.pyplot(fig, dpi=300)
    plt.close(fig)

    st.markdown("---")
    # LIME解释器，使用 val_intersect_lasso_boruta.csv作为背景数据
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=lime_bg_data.values,
        feature_names=lime_bg_data.columns.tolist(),
        class_names=["No Outcome", "Outcome"],
        mode="classification",
        random_state=RANDOM_SEED
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0].values,
        predict_fn=model.predict_proba
    )
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)

st.divider()
st.markdown("""
> **免责说明**：本工具仅为科研模型演示，不能替代临床医师判断。临床决策请结合患者实际病情综合评估。
""")
