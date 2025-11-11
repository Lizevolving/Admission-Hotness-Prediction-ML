# -*- coding: utf-8 -*-
"""
高校招生报考热度分析与预测系统 V3.1

核心功能：
- 根据用户输入的高考分数和意向专业方向，提供个性化的院校专业推荐。
- 利用机器学习模型预测各专业的热度，并结合历史录取数据进行科学排序。
- 提供“冲刺”、“稳妥”、“保底”三个档位的推荐列表，辅助考生决策。

版本亮点 (V3.1):
- 用户体验优化：增强界面引导，提供结果解读，优化空状态提示。
- 代码质量提升：完善函数文档字符串（Docstrings），增加关键逻辑注释。
- 可维护性增强：模块化UI组件，使用常量管理配置。
- 无障碍化（Accessibility）：为所有输入控件添加了详细的帮助文本。
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

# --- 全局常量配置 ---
# 推荐分数计算权重
WEIGHT_HOTNESS = 0.3
WEIGHT_MATCH_SCORE = 0.7

# 页面基础配置
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# 模块一: 资源加载与缓存
# -----------------------

@st.cache_resource
def load_model_and_columns():
    """
    加载预训练的机器学习模型和特征列定义。
    利用 Streamlit 的 @st.cache_resource 装饰器缓存加载结果，避免重复IO操作。

    Returns:
        tuple: (model, feature_columns) 或 (None, None) 如果加载失败。
    """
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(COLUMNS_PATH)
        return model, feature_columns
    except FileNotFoundError:
        st.error(f"错误：模型或特征列文件未找到。请检查路径配置：{MODEL_PATH}, {COLUMNS_PATH}")
        return None, None
    except Exception as e:
        st.error(f"加载模型或特征列时发生未知错误: {e}")
        return None, None

@st.cache_data
def load_data():
    """
    加载、验证并预处理高校招生数据。
    利用 @st.cache_data 缓存数据加载与处理结果，提高应用响应速度。

    Returns:
        pd.DataFrame or None: 处理后的数据帧，如果失败则返回None。
    """
    try:
        df = pd.read_csv(DATA_PATH)
        
        required_cols = [
            'year', 'province', 'school_name', 'major_name', 'school_tier',
            'category', 'plan_quota', 'apply_num', 'min_score', 'min_score_rank'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"数据文件 '{DATA_PATH}' 缺少必要的列: {', '.join(missing_cols)}")
            return None
        
        numeric_cols = ['plan_quota', 'apply_num', 'min_score', 'min_score_rank']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required_cols)
        return df
    except FileNotFoundError:
        st.error(f"错误：数据文件未找到。请检查路径：{DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"加载数据时发生未知错误: {e}")
        return None

# -----------------------
# 模块二: 核心算法
# -----------------------

def feature_engineering_for_prediction(data_row: dict, feature_columns: list) -> pd.DataFrame:
    """
    将单行输入数据转换为模型可接受的特征向量。
    该过程与模型训练时的特征工程步骤严格保持一致。

    Args:
        data_row (dict): 包含模型所需原始特征的字典。
        feature_columns (list): 模型训练时确定的最终特征列表。

    Returns:
        pd.DataFrame: 经过处理和对齐后的单行特征数据帧。
    """
    df_input = pd.DataFrame([data_row])
    
    # 对数变换，处理数据偏态
    for col in ['plan_quota', 'apply_num', 'min_score_rank']:
        if col in df_input.columns:
            df_input[f'log_{col}'] = np.log1p(df_input[col].astype(float))
    
    # One-Hot编码处理分类变量
    categorical_features = ['province', 'school_tier', 'category']
    df_encoded = pd.get_dummies(df_input, columns=categorical_features, drop_first=True)
    
    # 特征对齐，确保输入模型的特征与训练时完全一致
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_aligned.astype(float)


def estimate_rank_from_score(df: pd.DataFrame, score: int, province: str) -> int:
    """
    根据考生分数，基于历史数据估算其在省内的排名位次。
    这是一个关键步骤，用于将用户的分数输入与基于位次的推荐系统连接起来。

    Args:
        df (pd.DataFrame): 包含历史录取数据的完整数据帧。
        score (int): 考生的高考分数。
        province (str): 考生所在的省份。

    Returns:
        int: 估算的省内排名位次。
    """
    province_df = df[df['province'] == province][['min_score', 'min_score_rank']].dropna()
    if province_df.empty:
        # 如果没有该省份数据，返回一个默认的中等排名
        return 50000
    
    # 查找分数最接近的5个历史样本，取其中位数的排名作为估算结果，以增强稳定性
    province_df['abs_diff'] = (province_df['min_score'] - score).abs()
    nearest_samples = province_df.nsmallest(min(5, len(province_df)), 'abs_diff')
    return int(nearest_samples['min_score_rank'].median())


def generate_recommendations(df, model, feature_columns, user_score, user_rank, province, category):
    """
    执行智能推荐的核心算法。

    流程:
    1. 根据省份和科类筛选出候选院校专业。
    2. 对每个候选专业，调用机器学习模型预测其未来的热度指数。
    3. 计算每个候选专业的匹配度分数。
    4. 结合热度与匹配度，计算综合推荐分数。
    5. 根据历史录取位次与用户估算位次的关系，将候选集划分为“冲、稳、保”三档。
    6. 在每个档内，根据综合推荐分数进行排序。

    Args:
        df (pd.DataFrame): 完整的数据集。
        model: 已加载的机器学习模型。
        feature_columns (list): 模型所需的特征列表。
        user_score (int): 用户分数。
        user_rank (int): 用户估算位次。
        province (str): 用户所在省份。
        category (str): 用户意向科类。

    Returns:
        tuple: 包含三个DataFrame，分别对应冲刺、稳妥、保底的推荐结果。
    """
    candidates = df[(df['province'] == province) & (df['category'] == category)].copy()
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 批量预测热度
    predictions = []
    for _, row in candidates.iterrows():
        model_input = row.to_dict()
        model_input['year'] = pd.Timestamp.now().year # 使用当前年份进行预测
        try:
            X_aligned = feature_engineering_for_prediction(model_input, feature_columns)
            pred_hotness = float(model.predict(X_aligned)[0])
            predictions.append(pred_hotness)
        except Exception:
            # 如果单次预测失败，赋予一个默认中等热度值，保证系统健壮性
            predictions.append(5.0)
    candidates['predicted_hotness'] = predictions

    # 计算匹配度和综合推荐分
    candidates['match_score'] = np.maximum(0, 100 - abs(candidates['min_score'] - user_score) * 2)
    candidates['recommend_score'] = (candidates['predicted_hotness'] * WEIGHT_HOTNESS +
                                     candidates['match_score'] / 10 * WEIGHT_MATCH_SCORE)

    # "冲稳保"分档策略
    reach_df = candidates[(candidates['min_score_rank'] < user_rank) & (candidates['min_score_rank'] >= user_rank * 0.8)]
    match_df = candidates[(candidates['min_score_rank'] >= user_rank * 0.9) & (candidates['min_score_rank'] <= user_rank * 1.1)]
    safety_df = candidates[(candidates['min_score_rank'] > user_rank * 1.1) & (candidates['min_score_rank'] <= user_rank * 1.4)]
    
    # 排序并格式化输出
    display_cols = ['school_name', 'major_name', 'min_score', 'min_score_rank', 'predicted_hotness', 'match_score', 'recommend_score']
    
    def sort_and_format(df):
        return df.sort_values('recommend_score', ascending=False).head(10)[display_cols]
    
    return sort_and_format(reach_df), sort_and_format(match_df), sort_and_format(safety_df)

# -----------------------
# 模块三: UI 界面与交互
# -----------------------

def display_recommendation_table(df, title):
    """
    以格式化的表格形式展示推荐结果。

    Args:
        df (pd.DataFrame): 包含推荐结果的数据帧。
        title (str): 表格的标题。
    """
    st.subheader(title)
    
    if df.empty:
        st.info("暂无符合该类别的推荐。建议可适当调整分数或更换专业大类再次尝试。")
        return
        
    # 重命名列以提高可读性
    df_display = df.rename(columns={
        'school_name': '院校名称',
        'major_name': '专业名称',
        'min_score': '去年分数',
        'min_score_rank': '去年位次',
        'predicted_hotness': '预测热度',
        'match_score': '匹配度',
        'recommend_score': '推荐指数'
    })

    # 应用样式和格式化
    st.dataframe(
        df_display.style
        .format({
            '去年分数': '{:.0f}',
            '去年位次': '{:.0f}',
            '预测热度': '{:.1f}/10',
            '匹配度': '{:.0f}%',
            '推荐指数': '{:.1f}/10'
        })
        .background_gradient(cmap='viridis', subset=['推荐指数'])
        .highlight_max(subset=['预测热度'], color='lightcoral')
        .set_properties(**{'text-align': 'left'}),
        use_container_width=True,
        hide_index=True
    )
    
    # 增加结果解读，帮助用户理解
    with st.expander("如何解读推荐结果？"):
        st.markdown("""
        - **预测热度**: 基于机器学习模型对该专业明年报考热度的预测（满分10），分数越高代表可能越热门。
        - **匹配度**: 衡量该专业去年的录取分数与您的分数的接近程度（满分100%）。
        - **推荐指数**: 综合“预测热度”和“匹配度”得出的最终分数，是排序的核心依据。
        """)

def setup_sidebar(df):
    """
    配置并显示侧边栏的用户输入区域。

    Args:
        df (pd.DataFrame): 包含选项所需的数据。

    Returns:
        tuple: (score, province, category) 用户输入的值。
    """
    st.sidebar.header("输入您的信息")
    
    score = st.sidebar.number_input(
        "高考分数",
        min_value=150, max_value=750, value=550, step=1,
        help="请输入您的预估或实际高考总分。"
    )
    province = st.sidebar.selectbox(
        "所在省份",
        options=sorted(df['province'].unique()),
        help="请选择您参加高考的省份。"
    )
    category = st.sidebar.selectbox(
        "感兴趣的专业方向",
        options=sorted(df['category'].unique()),
        help="请选择您感兴趣的专业大类，系统将为您推荐该类别下的专业。"
    )
    return score, province, category

# -----------------------
# 主应用入口
# -----------------------

def main():
    """
    应用的主函数，负责整体流程控制。
    """
    st.title(f"🎓 {APP_TITLE}")
    
    # --- 1. 资源加载 ---
    model, feature_columns = load_model_and_columns()
    df = load_data()
    
    if model is None or df is None:
        st.error("系统核心组件加载失败，无法继续运行。请联系管理员检查后台配置。")
        st.stop()

    # --- 2. 用户输入 ---
    score, province, category = setup_sidebar(df)
    
    # --- 3. 主页面引导与执行 ---
    # 如果用户还未点击按钮，显示引导信息
    if 'recommendations' not in st.session_state:
        st.info("👈 请在左侧侧边栏输入您的信息，然后点击“开始智能推荐”按钮。")
        st.markdown("""
        #### 系统如何工作？
        1.  **输入您的信息**：在左侧提供您的高考分数、省份和感兴趣的专业方向。
        2.  **AI智能分析**：系统将基于您的分数估算全省排名，并调用机器学习模型预测备选专业的未来热度。
        3.  **获取个性化推荐**：您将得到“稳妥”、“冲刺”和“保底”三个档位的专业列表，每个列表都按“推荐指数”智能排序。
        """)
    
    # 主执行按钮
    if st.sidebar.button("开始智能推荐", type="primary", use_container_width=True):
        
        # --- 3.1 核心计算 ---
        with st.spinner("正在进行智能分析，请稍候..."):
            user_rank = estimate_rank_from_score(df, score, str(province))
            reach, match, safety = generate_recommendations(
                df, model, feature_columns, score, user_rank, str(province), str(category)
            )
            # 将结果存入会话状态，以便重新渲染时保留
            st.session_state['recommendations'] = (reach, match, safety)
            st.session_state['user_info'] = (score, user_rank, category)

    # --- 4. 结果展示 ---
    if 'recommendations' in st.session_state:
        reach, match, safety = st.session_state['recommendations']
        score, user_rank, category = st.session_state['user_info']
        
        st.markdown("---")
        st.header("您的个性化推荐报告")
        
        # 用户信息摘要
        col1, col2, col3 = st.columns(3)
        col1.metric("您的分数", f"{score} 分")
        col2.metric("预估省内位次", f"~ {user_rank} 名")
        col3.metric("意向专业方向", category)
        
        # 结果标签页
        tab1, tab2, tab3 = st.tabs(["🛡️ 稳妥推荐 (Match)", "🚀 冲刺机会 (Reach)", "📉 保底选择 (Safety)"])
        
        with tab1:
            display_recommendation_table(match, "稳妥推荐：录取概率较大，建议重点关注")
        with tab2:
            display_recommendation_table(reach, "冲刺推荐：可以大胆尝试，争取更好的机会")
        with tab3:
            display_recommendation_table(safety, "保底推荐：录取把握较大，确保有学可上")

        # --- 5. 导出功能 ---
        all_recs = pd.concat([match, reach, safety], ignore_index=True)
        if not all_recs.empty:
            st.markdown("---")
            st.download_button(
                label="📥 下载完整的推荐结果 (CSV格式)",
                data=all_recs.to_csv(index=False, encoding='utf-8-sig'),
                file_name=f"志愿推荐_{score}分_{province}_{category}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()