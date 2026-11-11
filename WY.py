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

st.set_page_config(page_title="重症患者达标风险预测模型", layout="wide")

# ===================== 全局配置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = WORK_DIR

# 新版 6 变量（LASSO ∩ Boruta 交集，无泄漏版）
FEATURE_COLS = ['CRRT', 'Hb', 'AST', 'BMI', 'High', 'ALP']

# XGBoost 训练集 Youden 指数最优阈值（2.训练集构建模型/train_optimal_thresholds.csv）
THRESHOLD = 0.8658

# ===================== 加载模型 =====================
@st.cache_resource
def load_model():
    model_path = os.path.join(WEB_DIR, "XGBoost_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ===================== 网页正文 =====================
st.title("重症患者达标风险预测模型")

st.markdown("""
本模型基于 **XGBoost** 算法（6 个协变量：CRRT、Hb、AST、BMI、High（身高）、ALP），
用于预测重症患者**达标**风险。
> 标签定义：**Yes = 达标（类别 1），No = 不达标（类别 0）**
> 判定规则：达标概率 ≥ 阈值 {th:.4f} 判为"达标"，否则判为"不达标"。
""")

st.divider()

# ===================== 输入表单 =====================
st.subheader("患者特征输入")
with st.form("pred_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        CRRT = st.selectbox("CRRT（是否行连续性肾脏替代治疗）", options=["No", "Yes"])
        Hb = st.number_input("Hb 血红蛋白 (g/L)", min_value=1.0, max_value=200.0, value=80.0, step=0.1)
    with col2:
        AST = st.number_input("AST 谷草转氨酶 (U/L)", min_value=1.0, max_value=500.0, value=40.0, step=0.1)
        BMI = st.number_input("BMI 体质指数 (kg/m²)", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    with col3:
        High = st.number_input("High 身高 (cm)", min_value=130.0, max_value=210.0, value=168.0, step=0.5)
        ALP = st.number_input("ALP 碱性磷酸酶 (U/L)", min_value=10.0, max_value=1000.0, value=120.0, step=1.0)

    submit_btn = st.form_submit_button("Predict 预测")

if submit_btn:
    input_df = pd.DataFrame([{
        "CRRT": 1 if CRRT == "Yes" else 0,
        "Hb": Hb,
        "AST": AST,
        "BMI": BMI,
        "High": High,
        "ALP": ALP,
    }])[FEATURE_COLS]

    proba = model.predict_proba(input_df)[0]
    prob_no  = float(proba[0])   # 类别 0 = No  不达标
    prob_yes = float(proba[1])   # 类别 1 = Yes 达标

    # 阈值逻辑：达标概率 >= THRESHOLD 判为达标
    pred_is_yes = 1 if prob_yes >= THRESHOLD else 0

    st.divider()
    st.subheader("📊 预测结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("达标 (Yes) 概率", f"{prob_yes:.2%}")
    c2.metric("不达标 (No) 概率", f"{prob_no:.2%}")
    c3.metric("模型最优阈值", f"{THRESHOLD:.4f}")

    if pred_is_yes == 1:
        st.success(f"最终判定: **YES（达标）** —— 患者达标可能性高（{prob_yes:.1%}），仍需常规临床随访观察。")
    else:
        st.error(f"最终判定: **NO（不达标）** —— 患者不达标风险较高（不达标概率 {prob_no:.1%}），建议密切监测，实施个体化干预。")

    st.markdown("---")
    st.subheader("SHAP Waterfall Plot — XGBoost（解释：Yes-达标）")

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_df)

        # 统一到"类别 1 = 达标"的 SHAP 值
        if isinstance(sv, list):
            values_yes = sv[1]
            base = explainer.expected_value[1]
        else:
            values_yes = sv
            base = explainer.expected_value
        values_yes = np.asarray(values_yes).reshape(-1)
        base = float(np.asarray(base).reshape(-1)[0])

        explanation = shap.Explanation(
            values=values_yes,
            base_values=base,
            data=input_df.values[0],
            feature_names=FEATURE_COLS,
        )

        fig = plt.figure(figsize=(11, 5.5))
        shap.waterfall_plot(explanation, max_display=6, show=False)
        plt.tight_layout()
        st.pyplot(fig, dpi=200)
        plt.close(fig)

    except Exception as e:
        st.warning(f"瀑布图绘制失败，改用力图（force plot）：{e}")
        try:
            fig = plt.figure(figsize=(14, 4))
            shap.force_plot(
                base,
                values_yes,
                input_df,
                matplotlib=True,
                show=False,
            )
            st.pyplot(fig, dpi=200)
            plt.close(fig)
        except Exception as e2:
            st.error(f"SHAP 绘图失败: {e2}")
            st.info("请尝试: pip install shap==0.44.0")

st.divider()
st.markdown("""
> **说明**: SHAP 瀑布图解释该患者"达标（Yes）"的预测：红色条代表该特征**推高达标概率**，蓝色条代表该特征**降低达标概率**。E[f(X)] 为模型基线（验证集平均）输出，f(x) 为该患者的最终模型输出。
>
> **Disclaimer**: This tool is for research demonstration only and does not replace clinical judgment. Clinical decisions should be comprehensively evaluated according to the patient's actual condition.
""")
