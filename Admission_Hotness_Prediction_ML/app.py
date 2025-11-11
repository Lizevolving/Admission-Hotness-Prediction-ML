# -*- coding: utf-8 -*-
"""
Streamlit 应用 — 高校招生热度预测模型演示
要求：与训练脚本保持一致：MODEL_PATH, COLUMNS_PATH, DATA_PATH, APP_TITLE 在 config.py 中定义
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config import APP_TITLE, MODEL_PATH, COLUMNS_PATH, DATA_PATH

st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")

# -----------------------
# 加载函数（使用缓存，提高速度）
# -----------------------
@st.cache_data
def load_model_and_columns(model_path=MODEL_PATH, cols_path=COLUMNS_PATH):
    """
    加载模型和特征列。如果加载失败，返回 None。
    """
    try:
        model = joblib.load(model_path)
        cols = joblib.load(cols_path)
        return model, cols
    except Exception as e:
        return None, None

@st.cache_data
def load_data(path=DATA_PATH):
    """
    加载历史数据。如果加载失败，返回 None。
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        return None

def load_metrics(model_path=MODEL_PATH):
    """
    加载模型指标文件（.metrics.json）。如果不存在或加载失败，返回 None。
    """
    metrics_path = Path(model_path).with_suffix('.metrics.json')
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

# -----------------------
# 输入处理：特征工程（与训练脚本相同）
# -----------------------
def feature_engineering_input(user_input: dict, feature_columns: list):
    """
    将用户输入转为模型需要的特征格式。
    步骤：对数值列取对数（log1p），对分类列做 one-hot 编码（drop_first=True），然后对齐特征列（缺失填 0）。
    """
    df_in = pd.DataFrame([user_input])
    # 对数变换（如果列存在）
    for c in ['plan_quota', 'apply_num', 'min_score_rank']:
        if c in df_in.columns:
            df_in[f'log_{c}'] = np.log1p(df_in[c].astype(float))

    # One-hot 编码（对分类列）
    categorical = [c for c in ['province', 'school_tier', 'category'] if c in df_in.columns]
    if categorical:
        df_enc = pd.get_dummies(df_in, columns=categorical, drop_first=True)
    else:
        df_enc = df_in.copy()

    # 删除原始数值列（与训练一致）
    for drop_col in ['school_name', 'major_name', 'plan_quota', 'apply_num', 'min_score_rank']:
        if drop_col in df_enc.columns:
            df_enc = df_enc.drop(columns=[drop_col])

    # 对齐特征列（缺失填 0）
    aligned = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        aligned.loc[0, col] = df_enc[col].iloc[0] if col in df_enc.columns else 0
    aligned = aligned.fillna(0)
    return aligned.astype(float)

# -----------------------
# 辅助函数：用分数估算排名（基于历史数据）
# -----------------------
def estimate_rank_from_score(df, score):
    """
    用历史数据估算排名：找分数最近的 10 个样本，取排名中位数。
    如果数据缺少相关列，返回 None。
    """
    if 'min_score' not in df.columns or 'min_score_rank' not in df.columns:
        return None
    df_score = df[['min_score', 'min_score_rank']].copy().dropna()
    if df_score.empty:
        return None
    df_score['abs_diff'] = (df_score['min_score'] - score).abs()
    k = min(10, len(df_score))
    nearest = df_score.nsmallest(k, 'abs_diff')
    est_rank = int(nearest['min_score_rank'].median())
    return est_rank

# -----------------------
# 辅助函数：推荐相似样本（按热度差距）
# -----------------------
def recommend_similar(df, school_tier, category, target_hotness, top_k=5):
    """
    从历史数据中找相似样本：相同学校层次和科类，按热度差距排序，取前 5 个。
    如果没有热度列，就返回前 5 个匹配样本。
    """
    filt = df.copy()
    if 'school_tier' in df.columns:
        filt = filt[filt['school_tier'] == school_tier]
    if 'category' in df.columns:
        filt = filt[filt['category'] == category]
    if filt.empty:
        return pd.DataFrame()
    if 'hotness_index' in filt.columns:
        filt['hotness_diff'] = (filt['hotness_index'] - target_hotness).abs()
        return filt.nsmallest(top_k, 'hotness_diff')[['school_name','major_name','hotness_index','plan_quota','apply_num']].reset_index(drop=True)
    else:
        return filt.head(top_k)[['school_name','major_name','plan_quota','apply_num']].reset_index(drop=True)

# -----------------------
# 主函数：应用界面
# -----------------------
def main():
    st.title("🎓 高校招生热度预测模型演示")
    st.write("这个应用展示如何用简单输入测试模型。输入参数，运行预测，查看结果。重点：了解模型流程和特征处理。")

    # 加载模型、数据和指标
    model, feature_columns = load_model_and_columns()
    df = load_data()
    metrics = load_metrics()

    if model is None or feature_columns is None or df is None:
        st.error("模型或数据加载失败。请检查 config.py 中的路径设置，并确保模型已训练并保存。")
        st.stop()

    # 侧栏：输入参数
    st.sidebar.header("输入参数")
    input_mode = st.sidebar.radio("选择输入方式", ("直接输入最低排名（推荐）", "用分数估算排名（如果有分数数据）"))

    year = st.sidebar.selectbox("年份", options=sorted(df['year'].unique()) if 'year' in df.columns else [2025], index=0)
    province = st.sidebar.selectbox("省份", options=sorted(df['province'].unique()) if 'province' in df.columns else ["北京"])
    school_tier = st.sidebar.selectbox("学校层次", options=sorted(df['school_tier'].unique()) if 'school_tier' in df.columns else ["普通本科"])
    category = st.sidebar.selectbox("科类", options=sorted(df['category'].unique()) if 'category' in df.columns else ["工学"])

    plan_quota = st.sidebar.number_input("计划招生人数 (plan_quota)", min_value=1, value=100)
    apply_num = st.sidebar.number_input("报考人数 (apply_num)", min_value=1, value=1000)

    if input_mode == "直接输入最低排名（推荐）":
        min_score_rank = st.sidebar.number_input("最低录取分排名 (min_score_rank)", min_value=1, value=50000)
    else:
        score = st.sidebar.number_input("最低录取分数 (min_score) - 用于估算排名", min_value=0, max_value=750, value=550)
        est_rank = estimate_rank_from_score(df, score)
        if est_rank is None:
            st.sidebar.warning("无法用历史数据估算排名。请直接输入排名。")
            min_score_rank = st.sidebar.number_input("最低录取分排名 (min_score_rank)", min_value=1, value=50000)
        else:
            st.sidebar.info(f"基于历史数据估算的排名: {est_rank}")
            min_score_rank = est_rank

    # 预测按钮
    if st.sidebar.button("🔮 运行预测"):
        user_input = {
            'year': year,
            'province': province,
            'school_tier': school_tier,
            'category': category,
            'plan_quota': plan_quota,
            'apply_num': apply_num,
            'min_score_rank': min_score_rank
        }
        try:
            X_aligned = feature_engineering_input(user_input, feature_columns)
            pred = float(model.predict(X_aligned)[0])

            # 显示结果
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.metric("预测热度指数", f"{pred:.2f}")
            with c2:
                st.metric("报考竞争比", f"{apply_num / max(plan_quota,1):.1f}:1")
            with c3:
                difficulty = "高" if min_score_rank < 10000 else "中" if min_score_rank < 50000 else "低"
                st.metric("录取难度（估计）", difficulty)

            # 输入详情
            st.markdown("### 输入参数详情")
            st.table(pd.DataFrame({
                "参数": ["年份","省份","学校层次","科类","计划招生","报考人数","最低排名"],
                "值": [f"{year}年", province, school_tier, category, f"{plan_quota}人", f"{apply_num}人", f"第{min_score_rank}名"]
            }))

            # 相似样本
            st.markdown("### 相似样本推荐")
            recs = recommend_similar(df, school_tier, category, pred, top_k=5)
            if not recs.empty:
                st.dataframe(recs, use_container_width=True)
            else:
                st.info("未找到相似样本。")

            # 导出结果
            if st.button("⬇️ 导出结果（CSV）"):
                out_df = pd.DataFrame([{
                    'year': year, 'province': province, 'school_tier': school_tier,
                    'category': category, 'plan_quota': plan_quota, 'apply_num': apply_num,
                    'min_score_rank': min_score_rank, 'predicted_hotness': pred
                }])
                st.download_button("下载 CSV", out_df.to_csv(index=False, encoding='utf-8-sig'), file_name="prediction.csv", mime="text/csv")

        except Exception as e:
            st.error(f"预测出错: {e}")

    # 侧栏：模型指标
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 模型指标（从训练保存）")
    if metrics:
        st.sidebar.write(f"CV R² 平均: **{metrics.get('cv_r2_mean','-')}**")
        st.sidebar.write(f"CV R² 标准差: **{metrics.get('cv_r2_std','-')}**")
        if 'test_metrics' in metrics:
            st.sidebar.write("测试集指标：")
            for k,v in metrics['test_metrics'].items():
                st.sidebar.write(f"- {k}: **{v:.4f}**")
    else:
        st.sidebar.info("未找到指标文件。可能训练脚本未保存 .metrics.json。")

    # 页脚说明
    st.markdown("---")
    st.markdown("#### 使用说明")
    st.markdown("""
    - 这个应用用于测试模型：输入参数（年份、省份、层次、科类、计划招生、报考人数、最低排名），运行预测，查看热度指数。
    - 要提升准确性：在训练脚本中使用 K-fold 交叉验证，检查数据泄露，并调整特征（训练脚本已有这些步骤）。
    - 运行方式：用 Streamlit 命令启动，逐步输入参数，观察模型输出。
    """)
    st.write("—— 专注模型测试。")

if __name__ == "__main__":
    main()
