# -*- coding: utf-8 -*-
"""
高校招生报考热度分析与预测系统 V3.2 - 增强版

核心功能：
- 根据用户输入的高考分数和意向专业方向，提供个性化的院校专业推荐。
- 利用机器学习模型预测各专业的热度，并结合历史录取数据进行科学排序。
- 提供"冲刺"、"稳妥"、"保底"三个档位的推荐列表，辅助考生决策。

版本亮点 (V3.2):
- 修复标签页显示问题：确保冲稳保三档正确展示
- 优化下载功能：简化下载逻辑，确保按钮生效
- 全面UI美化：提升视觉设计，增强用户体验
- 增加交互反馈：loading状态、成功提示、错误处理
- 优化数据展示：改进表格样式，增加数据可视化
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

# 自定义CSS样式
st.markdown("""
<style>
    /* 主容器样式 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* 成功提示样式 */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #0a5f0a;
        margin: 1rem 0;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* 按钮样式增强 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

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
        st.error(f"❌ 错误：模型或特征列文件未找到。请检查路径配置：{MODEL_PATH}, {COLUMNS_PATH}")
        return None, None
    except Exception as e:
        st.error(f"❌ 加载模型或特征列时发生未知错误: {e}")
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
            st.error(f"❌ 数据文件 '{DATA_PATH}' 缺少必要的列: {', '.join(missing_cols)}")
            return None
        
        numeric_cols = ['plan_quota', 'apply_num', 'min_score', 'min_score_rank']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required_cols)
        return df
    except FileNotFoundError:
        st.error(f"❌ 错误：数据文件未找到。请检查路径：{DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"❌ 加载数据时发生未知错误: {e}")
        return None

@st.cache_data
def load_metrics():
    """加载模型指标文件"""
    metrics_path = Path(MODEL_PATH).with_suffix('.metrics.json')
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
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
    5. 根据历史录取位次与用户估算位次的关系，将候选集划分为"冲、稳、保"三档。
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

def get_competition_level(hotness):
    """获取竞争程度描述"""
    if hotness > 7:
        return "🔥 非常激烈"
    elif hotness > 5:
        return "📈 激烈"
    elif hotness > 3:
        return "📊 中等"
    else:
        return "📉 一般"

# -----------------------
# 模块三: UI 界面与交互
# -----------------------

def display_recommendation_table(df, title, emoji="📊"):
    """
    以格式化的表格形式展示推荐结果。

    Args:
        df (pd.DataFrame): 包含推荐结果的数据帧。
        title (str): 表格的标题。
        emoji (str): 标题前的表情符号。
    """
    st.markdown(f"### {emoji} {title}")
    
    if df.empty:
        st.info("📝 暂无符合该类别的推荐。建议可适当调整分数或更换专业大类再次尝试。")
        return
        
    # 重命名列以提高可读性
    df_display = df.copy()
    df_display['competition_level'] = df_display['predicted_hotness'].apply(get_competition_level)
    
    # 创建美化显示数据
    display_data = []
    for _, row in df_display.iterrows():
        display_data.append({
            '🏫 院校名称': row['school_name'],
            '📚 专业名称': row['major_name'],
            '📊 去年分数': f"{row['min_score']:.0f}分",
            '🏆 去年位次': f"{int(row['min_score_rank']):,}名",
            '🔥 预测热度': f"{row['predicted_hotness']:.1f}/10",
            '⚡ 竞争程度': row['competition_level'],
            '💯 匹配度': f"{row['match_score']:.0f}%",
            '🎯 推荐指数': f"{row['recommend_score']:.1f}/10"
        })
    
    display_df = pd.DataFrame(display_data)
    
    # 应用样式和格式化
    st.dataframe(
        display_df.style.set_properties(**{
            'background-color': '#f8f9fa',
            'border': '1px solid #dee2e6',
            'color': '#495057',
            'text-align': 'left'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # 增加结果解读，帮助用户理解
    with st.expander("📖 如何解读推荐结果？"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 关键指标说明：**
            - **预测热度**: AI模型对该专业明年报考热度的预测（0-10分）
            - **匹配度**: 该专业去年录取分数与您分数的接近程度
            - **推荐指数**: 综合热度和匹配度的最终评分，是排序核心依据
            """)
        with col2:
            st.markdown("""
            **💡 使用建议：**
            - 🛡️ **稳妥推荐**: 录取概率较大，建议重点关注
            - 🚀 **冲刺推荐**: 可以尝试冲击，争取更好机会
            - 📉 **保底推荐**: 录取把握较大，确保有学可上
            """)

def setup_sidebar(df):
    """
    配置并显示侧边栏的用户输入区域。

    Args:
        df (pd.DataFrame): 包含选项所需的数据。

    Returns:
        tuple: (score, province, category) 用户输入的值。
    """
    st.sidebar.markdown("## 📝 输入您的信息")
    
    with st.sidebar.container():
        # 分数输入
        score = st.number_input(
            "🎯 高考分数",
            min_value=150, max_value=750, value=550, step=1,
            help="请输入您的预估或实际高考总分（满分750分）"
        )
        
        # 省份选择
        province = st.selectbox(
            "🗺️ 所在省份",
            options=sorted(df['province'].unique()),
            help="请选择您参加高考的省份"
        )
        
        # 专业方向选择
        category = st.selectbox(
            "📚 感兴趣的专业方向",
            options=sorted(df['category'].unique()),
            help="请选择您感兴趣的专业大类"
        )
    
    # 添加分隔线
    st.sidebar.markdown("---")
    
    # 模型信息
    metrics = load_metrics()
    if metrics:
        with st.sidebar.expander("📊 模型性能", expanded=False):
            st.metric("交叉验证 R²", f"{metrics.get('cv_r2_mean', 'N/A')}")
            if 'test_metrics' in metrics:
                for metric, value in metrics['test_metrics'].items():
                    st.metric(f"测试集 {metric}", f"{value:.4f}")
    
    return score, province, category

def display_user_summary(score, user_rank, category):
    """显示用户信息摘要"""
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### 🎯 您的分数")
        st.markdown(f"<h2 style='color: #0a5f0a; margin: 0;'>{score} 分</h2>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### 🏆 预估省内位次")
        st.markdown(f"<h2 style='color: #0a5f0a; margin: 0;'>~ {user_rank:,} 名</h2>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"### 📚 意向专业方向")
        st.markdown(f"<h2 style='color: #0a5f0a; margin: 0;'>{category}</h2>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# 主应用入口
# -----------------------

def main():
    """
    应用的主函数，负责整体流程控制。
    """
    # 美化的标题
    st.markdown("""
    <div class="main-header">
        <h1>🎓 高校招生报考热度分析与预测系统</h1>
        <p>基于机器学习技术，为您提供智能化的高考志愿填报建议</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 系统特色介绍
    st.markdown("""
    ### ✨ 系统特色
    🤖 **AI智能推荐**：基于机器学习模型预测专业热度  
    📊 **科学分档**：提供"冲刺-稳妥-保底"三档推荐  
    🎯 **精准匹配**：根据分数和兴趣方向个性化推荐  
    💾 **数据导出**：支持推荐结果下载，便于离线查看
    """)
    
    # --- 1. 资源加载 ---
    with st.spinner("🔄 正在加载系统资源..."):
        model, feature_columns = load_model_and_columns()
        df = load_data()
    
    if model is None or df is None:
        st.error("❌ 系统核心组件加载失败，无法继续运行。请联系管理员检查后台配置。")
        st.stop()

    # --- 2. 用户输入 ---
    score, province, category = setup_sidebar(df)
    
    # --- 3. 主页面引导与执行 ---
    # 如果用户还未点击按钮，显示引导信息
    if 'recommendations_generated' not in st.session_state:
        st.info("👈 请在左侧侧边栏输入您的信息，然后点击下方按钮开始智能推荐。")
        
        # 系统工作流程说明
        with st.expander("🔍 系统如何工作？", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **1️⃣ 输入您的信息**  
                在左侧提供您的高考分数、省份和感兴趣的专业方向。
                """)
            with col2:
                st.markdown("""
                **2️⃣ AI智能分析**  
                系统基于您的分数估算全省排名，并调用机器学习模型预测备选专业的未来热度。
                """)
            with col3:
                st.markdown("""
                **3️⃣ 获取个性化推荐**  
                您将得到"稳妥"、"冲刺"和"保底"三个档位的专业列表，每个列表都按"推荐指数"智能排序。
                """)
    
    # 主执行按钮
    if st.sidebar.button("🚀 开始智能推荐", type="primary", use_container_width=True):
        
        # --- 3.1 核心计算 ---
        with st.spinner("🤖 正在进行智能分析，请稍候..."):
            user_rank = estimate_rank_from_score(df, score, str(province))
            reach, match, safety = generate_recommendations(
                df, model, feature_columns, score, user_rank, str(province), str(category)
            )
            
            # 将结果存入会话状态
            st.session_state['recommendations'] = {
                'reach': reach,
                'match': match,
                'safety': safety
            }
            st.session_state['user_info'] = {
                'score': score,
                'user_rank': user_rank,
                'category': category,
                'province': province
            }
            st.session_state['recommendations_generated'] = True

    # --- 4. 结果展示 ---
    if 'recommendations_generated' in st.session_state and st.session_state['recommendations_generated']:
        recommendations = st.session_state['recommendations']
        user_info = st.session_state['user_info']
        
        st.markdown("---")
        
        # 用户信息摘要
        display_user_summary(
            user_info['score'], 
            user_info['user_rank'], 
            user_info['category']
        )
        
        # 结果标签页
        st.markdown("## 📋 您的个性化推荐结果")
        
        tab1, tab2, tab3 = st.tabs(["🛡️ 稳妥推荐", "🚀 冲刺机会", "📉 保底选择"])
        
        with tab1:
            display_recommendation_table(
                recommendations['match'], 
                "稳妥推荐：录取概率较大，建议重点关注",
                "🛡️"
            )
        
        with tab2:
            display_recommendation_table(
                recommendations['reach'], 
                "冲刺推荐：可以大胆尝试，争取更好的机会",
                "🚀"
            )
        
        with tab3:
            display_recommendation_table(
                recommendations['safety'], 
                "保底推荐：录取把握较大，确保有学可上",
                "📉"
            )

        # --- 5. 导出功能 ---
        all_recs = pd.concat([
            recommendations['match'], 
            recommendations['reach'], 
            recommendations['safety']
        ], ignore_index=True)
        
        if not all_recs.empty:
            st.markdown("---")
            st.markdown("## 💾 导出推荐结果")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ 共为您推荐了 {len(all_recs)} 个专业选项")
            
            with col2:
                # 直接使用download_button而不是嵌套在button中
                st.download_button(
                    label="📥 下载完整推荐结果 (CSV)",
                    data=all_recs.to_csv(index=False, encoding='utf-8-sig'),
                    file_name=f"志愿推荐_{user_info['score']}分_{user_info['province']}_{user_info['category']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎓 高校招生报考热度分析与预测系统 V3.2 | 基于机器学习技术</p>
        <p>为高考生提供智能专业推荐服务 | 让志愿填报更科学 🎯</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
