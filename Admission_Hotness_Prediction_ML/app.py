# -*- coding: utf-8 -*-
"""
高校招生报考热度分析与预测系统
核心功能：检索+推荐
输入：分数 + 感兴趣方向 → 输出：相关专业推荐 + 分数线预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from config import APP_TITLE, MODEL_PATH, COLUMNS_PATH, DATA_PATH

# 页面配置
st.set_page_config(
    page_title="高校招生报考热度预测系统", 
    page_icon="🎓", 
    layout="wide"
)

# 缓存加载
@st.cache_data
def load_model_and_data():
    """加载模型和数据"""
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(COLUMNS_PATH)
        df = pd.read_csv(DATA_PATH)
        return model, feature_columns, df
    except Exception as e:
        st.error(f"加载失败: {e}")
        return None, None, None

def predict_score_and_hotness(model, feature_columns, df, score, category, province=None):
    """
    核心预测功能：根据分数和方向预测分数线和热度
    """
    try:
        # 筛选相同科类的历史数据
        category_data = df[df['category'] == category].copy()
        
        if province:
            category_data = category_data[category_data['province'] == province]
        
        if category_data.empty:
            return pd.DataFrame()
        
        # 计算该科类的分数线范围
        if 'min_score' in category_data.columns:
            score_stats = category_data['min_score'].describe()
            recommended_score_range = (score_stats['25%'], score_stats['75%'])
        else:
            recommended_score_range = (score - 20, score + 20)
        
        # 为每个专业预测热度
        recommendations = []
        
        for _, school_major_info in category_data.iterrows():
            # 构造预测输入
            user_input = {
                'year': 2024,
                'province': school_major_info.get('province', '北京'),
                'school_tier': school_major_info.get('school_tier', '普通本科'),
                'category': category,
                'plan_quota': school_major_info.get('plan_quota', 100),
                'apply_num': school_major_info.get('apply_num', 1000),
                'min_score_rank': estimate_rank_from_score(df, score)
            }
            
            # 预测热度
            predicted_hotness = predict_single_hotness(model, feature_columns, user_input)
            
            # 预测分数线（基于历史数据 + 热度调整）
            historical_score = school_major_info.get('min_score', score)
            score_adjustment = (predicted_hotness - 5) * 2  # 热度影响分数
            predicted_score = max(0, historical_score + score_adjustment)
            
            recommendations.append({
                'school_name': school_major_info.get('school_name', '未知大学'),
                'major_name': school_major_info.get('major_name', '未知专业'),
                'province': school_major_info.get('province', '未知'),
                'school_tier': school_major_info.get('school_tier', '普通本科'),
                'historical_score': historical_score,
                'predicted_score': round(predicted_score, 1),
                'predicted_hotness': round(predicted_hotness, 2),
                'match_score': calculate_match_score(score, predicted_score),
                'competition_level': get_competition_level(predicted_hotness)
            })
        
        # 转换为DataFrame并排序
        rec_df = pd.DataFrame(recommendations)
        if not rec_df.empty:
            # 按匹配度和热度综合排序
            rec_df['sort_score'] = rec_df['match_score'] * 0.6 + (10 - rec_df['predicted_hotness']) * 0.4
            rec_df = rec_df.sort_values('sort_score', ascending=False)
        
        return rec_df, recommended_score_range
        
    except Exception as e:
        st.error(f"预测错误: {e}")
        return pd.DataFrame(), (0, 750)

def estimate_rank_from_score(df, score):
    """根据分数估算排名"""
    try:
        if 'min_score' not in df.columns or 'min_score_rank' not in df.columns:
            return 50000
        
        df_score = df[['min_score', 'min_score_rank']].copy().dropna()
        if df_score.empty:
            return 50000
        
        df_score['abs_diff'] = (df_score['min_score'] - score).abs()
        nearest = df_score.nsmallest(min(5, len(df_score)), 'abs_diff')
        return int(nearest['min_score_rank'].median())
    except:
        return 50000

def predict_single_hotness(model, feature_columns, user_input):
    """预测单个热度的辅助函数"""
    try:
        df_input = pd.DataFrame([user_input])
        
        # 特征工程
        df_input['log_plan_quota'] = np.log1p(df_input['plan_quota'])
        df_input['log_apply_num'] = np.log1p(df_input['apply_num'])
        df_input['log_min_score_rank'] = np.log1p(df_input['min_score_rank'])
        
        # One-hot编码
        categorical_features = ['province', 'school_tier', 'category']
        df_encoded = pd.get_dummies(df_input, columns=categorical_features, drop_first=True)
        
        # 移除不需要的特征
        features_to_remove = ['school_name', 'major_name', 'plan_quota', 'apply_num', 'min_score_rank']
        for feature in features_to_remove:
            if feature in df_encoded.columns:
                df_encoded = df_encoded.drop(feature, axis=1)
        
        # 对齐特征列
        df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
        
        # 预测
        return model.predict(df_aligned)[0]
    except:
        return 5.0  # 默认中等热度

def calculate_match_score(user_score, predicted_score):
    """计算匹配分数"""
    score_diff = abs(user_score - predicted_score)
    if score_diff <= 10:
        return 100
    elif score_diff <= 20:
        return 80
    elif score_diff <= 30:
        return 60
    else:
        return 40

def get_competition_level(hotness):
    """获取竞争程度"""
    if hotness > 7:
        return "非常激烈"
    elif hotness > 5:
        return "激烈"
    elif hotness > 3:
        return "中等"
    else:
        return "一般"

def main():
    """主函数"""
    # 标题
    st.title("🎓 高校招生报考热度分析与预测系统")
    st.markdown("---")
    
    # 简洁说明
    st.markdown("""
    ### 📖 系统功能
    基于机器学习技术，根据你的**高考分数**和**兴趣方向**，为你推荐合适的专业，并预测录取分数线。
    
    ✅ **输入简单**：只需分数+兴趣方向  
    ✅ **智能推荐**：基于历史数据和AI预测  
    ✅ **分数线预测**：预测各专业录取分数  
    """)
    
    # 加载模型和数据
    model, feature_columns, df = load_model_and_data()
    
    if model is None:
        st.error("系统加载失败，请稍后重试")
        st.stop()
    
    # 主要输入区域
    st.markdown("### 📝 请输入你的信息")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = st.number_input(
            "🎯 你的高考分数", 
            min_value=0, 
            max_value=750, 
            value=500,
            help="请输入你的高考总分（满分750分）"
        )
    
    with col2:
        category_options = sorted(df['category'].unique()) if df is not None and 'category' in df.columns else ['理科', '文科', '工科']
        category = st.selectbox(
            "📚 感兴趣的科类", 
            options=category_options,
            help="选择你感兴趣的专业科类"
        )
    
    with col3:
        province_options = ['全部省份'] + (sorted(df['province'].unique()) if df is not None and 'province' in df.columns else [])
        province = st.selectbox(
            "🗺️ 目标省份（可选）", 
            options=province_options,
            help="选择你希望上大学的省份，不选择则查看全国"
        )
    
    # 预测按钮
    if st.button("🔮 开始推荐专业", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### 📊 推荐结果")
        
        # 处理省份选择
        selected_province = None if province == '全部省份' else province
        
        # 获取推荐
        recommendations, score_range = predict_score_and_hotness(
            model, feature_columns, df, score, category, selected_province
        )
        
        if recommendations.empty:
            st.warning("抱歉，没有找到符合条件的专业推荐。请尝试其他科类或调整分数。")
            return
        
        # 显示分数范围分析
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 你的分数", f"{score}分")
        
        with col2:
            st.metric("📈 匹配专业数量", f"{len(recommendations)}个")
        
        with col3:
            if score_range[0] <= score <= score_range[1]:
                match_status = "✅ 匹配良好"
            else:
                match_status = "⚠️ 需要调整"
            st.metric("💯 分数匹配度", match_status)
        
        # 显示推荐专业列表
        st.markdown("### 🏫 推荐专业列表")
        st.markdown(f"根据你的**{score}分**和**{category}**方向，为你推荐以下专业：")
        
        # 格式化显示数据
        display_data = []
        for _, row in recommendations.iterrows():
            display_data.append({
                '🏫 学校': row['school_name'],
                '📚 专业': row['major_name'],
                '📍 地区': row['province'],
                '🎓 层次': row['school_tier'],
                '📊 历史分数': f"{row['historical_score']}分",
                '🔮 预测分数': f"{row['predicted_score']}分",
                '🔥 热度指数': f"{row['predicted_hotness']}/10",
                '⚡ 竞争程度': row['competition_level'],
                '💯 匹配度': f"{row['match_score']}%"
            })
        
        display_df = pd.DataFrame(display_data)
        
        # 显示表格
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 提供下载功能
        if st.button("📥 下载推荐结果", use_container_width=True):
            csv_data = recommendations.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载 CSV 文件",
                data=csv_data,
                file_name=f"专业推荐_{score}分_{category}.csv",
                mime="text/csv"
            )
        
        # 显示分析说明
        st.markdown("---")
        st.markdown("### 📋 分析说明")
        st.markdown(f"""
        - **预测分数线**：基于历史录取数据和AI模型预测，实际录取分数可能有所浮动
        - **热度指数**：反映该专业的报考竞争激烈程度（0-10分，分数越高竞争越激烈）
        - **匹配度**：你的分数与预测分数的匹配程度，越高越适合报考
        - **推荐排序**：综合考虑匹配度和竞争程度，优先推荐最适合的专业
        """)
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🎓 高校招生报考热度分析与预测系统 | 基于机器学习技术 | 为高考生提供智能专业推荐
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
