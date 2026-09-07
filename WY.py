import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from lime.lime_tabular import LimeTabularExplainer

# ===================== 全局配置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
RANDOM_SEED = 111
np.random.seed(RANDOM_SEED)

WORK_DIR = r"F:/BLA2"
WEB_DIR = os.path.join(WORK_DIR, "7.网页用")
os.makedirs(WEB_DIR, exist_ok=True)

# 你的特征顺序，必须和训练时完全一致
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

# 你的 XGBoost 最优阈值
THRESHOLD = 0.636

# ===================== 加载模型和测试数据 =====================
@st.cache_resource
def load_model_and_data():
    model_path = os.path.join(WEB_DIR, "XGBoost_model.pkl")
    model = joblib.load(model_path)

    # X_test.csv 用于 LIME 背景数据
    X_test_path = os.path.join(WEB_DIR, "X_test.csv")
    X_test = pd.read_csv(X_test_path)

    # 确保 CRRT 是数值型 0/1
    if "CRRT" in X_test.columns:
        if pd.api.types.is_string_dtype(X_test["CRRT"]) or X_test["CRRT"].dtype == object:
            X_test["CRRT"] = (
                X_test["CRRT"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
            )

    return model, X_test

model, X_test = load_model_and_data()

# ===================== 网页界面 =====================
st.set_page_config(page_title="结局风险预测模型", layout="wide")

st.title("重症患者结局风险预测模型")

st.markdown(
    """
    本模型基于 XGBoost 算法构建，用于预测患者发生结局事件的风险。  
    模型输入包含 10 项特征：Cr、AGE、CRRT、BUN、vein-Total daily dose、BMI、PLT、CrCL、TP、TBIL。
    """
)

st.divider()

# ===================== 左侧输入表单 =====================
st.subheader("患者特征输入")

with st.form("prediction_form"):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Cr = st.number_input("Cr", min_value=0.0, max_value=300.0, value=60.0, step=0.1)
        AGE = st.number_input("AGE", min_value=18, max_value=110, value=60)

    with col2:
        CRRT = st.selectbox("CRRT", options=["No", "Yes"])
        BUN = st.number_input("BUN", min_value=0.0, max_value=150.0, value=10.0, step=0.1)

    with col3:
        vein_Total_daily_dose = st.number_input(
            "vein-Total daily dose",
            min_value=0.0,
            max_value=5000.0,
            value=1000.0,
            step=1.0
        )
        BMI = st.number_input("BMI", min_value=12.0, max_value=50.0, value=24.0, step=0.1)

    with col4:
        PLT = st.number_input("PLT", min_value=10, max_value=600, value=200)
        CrCL = st.number_input("CrCL", min_value=0.0, max_value=200.0, value=60.0, step=0.1)

    with col5:
        TP = st.number_input("TP", min_value=30.0, max_value=90.0, value=65.0, step=0.1)
        TBIL = st.number_input("TBIL", min_value=0.0, max_value=200.0, value=12.0, step=0.1)

    submitted = st.form_submit_button("Predict 预测")

# ===================== 预测逻辑 =====================
if submitted:
    # CRRT: Yes -> 1, No -> 0
    crrt_val = 1 if CRRT == "Yes" else 0

    # 组装特征，顺序必须和 feature_cols 一致
    feature_values = [
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

    input_df = pd.DataFrame([feature_values], columns=feature_cols)

    # XGBoost 预测
    pred_prob = model.predict_proba(input_df)[0]
    prob_pos = pred_prob[1]
    prob_neg = pred_prob[0]

    # 使用训练集最优阈值 0.636
    predicted_class = 1 if prob_pos > THRESHOLD else 0

    st.divider()

    # ===================== 预测结果展示 =====================
    st.subheader("📊 预测结果")

    st.write(f"预测发生结局概率 (DV=Yes)：**{prob_pos:.2%}**")
    st.write(f"预测不发生结局概率 (DV=No)：**{prob_neg:.2%}**")
    st.write(f"训练集最优阈值：**{THRESHOLD}**")

    if predicted_class == 1:
        st.error("最终预测输出：**YES**")
        advice = (
            f"模型提示该患者发生结局事件风险较高，预测风险概率为 {prob_pos:.1%}。"
            "建议临床综合评估患者病情，密切监测相关指标，并制定个体化干预方案。"
        )
    else:
        st.success("最终预测输出：**NO**")
        advice = (
            f"模型提示该患者发生结局事件风险较低，预测风险概率为 {prob_pos:.1%}。"
            "仍需坚持常规临床监测，并根据患者情况继续维持合理治疗方案。"
        )

    st.info(advice)

    st.markdown("---")

    # ===================== SHAP Force Plot =====================
    st.subheader("SHAP Force Plot Explanation")

    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(14, 4))

    shap.force_plot(
        explainer_shap.expected_value,
        shap_values[0],
        input_df,
        matplotlib=True,
        ax=ax
    )

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    st.image(buf, caption="SHAP Force Plot Explanation")
    plt.close(fig)

    st.markdown("---")

    # ===================== LIME Explanation =====================
    st.subheader("LIME Explanation")

    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=["No Outcome", "Outcome"],
        mode="classification",
        random_state=RANDOM_SEED
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0, :].values,
        predict_fn=model.predict_proba
    )

    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)

# ===================== 底部说明 =====================
st.divider()

st.markdown(
    """
    **说明：** 本预测模型仅用于科研探索和辅助分析，预测结果不能替代临床医生的专业判断。  
    实际诊疗决策应结合患者病情、临床表现、检查结果和医生经验综合确定。
    """
)
