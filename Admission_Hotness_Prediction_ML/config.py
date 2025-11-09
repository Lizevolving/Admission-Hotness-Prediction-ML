# config.py
"""
配置文件 - 统一管理核心路径和模型超参数。
所有数据相关的配置（如特征列表、可选类别）都应在代码中从数据源动态生成。
"""


from pathlib import Path

# --- 核心路径定义 (使用 pathlib 保证跨平台兼容性) ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/admissions.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "admissions_model.pkl"
COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

# --- 模型超参数 ---
MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'n_jobs': -1
}

# --- 数据字段定义 (只定义关键的目标变量和ID类变量) ---
TARGET_COLUMN = 'hotness_index'
# 注意：特征列表将从数据中动态推断，不再硬编码

# --- 应用UI配置 ---
APP_TITLE = "高校招生热度分析与预测系统"

# --- (可选) 为UI提供完整的选项列表，但推荐从数据动态生成 ---
# 如果数据不全，这些可以作为备用或完整列表
ALL_PROVINCES = [
    '北京', '上海', '广东', '江苏', '浙江', '山东', '河南', '四川', 
    '湖北', '湖南', '河北', '福建', '安徽', '陕西', '辽宁', '江西',
    '重庆', '云南', '广西', '山西', '内蒙古', '吉林', '黑龙江', '贵州',
    '新疆', '甘肃', '海南', '宁夏', '青海', '西藏'
]
ALL_SCHOOL_TIERS = ['985', '211', '普通本科', '专科']
ALL_CATEGORIES = ['理科', '文科', '综合', '工科'] # 已去重