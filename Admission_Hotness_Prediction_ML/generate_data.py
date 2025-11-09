# -*- coding: utf-8 -*-
"""
生成模拟高校招生数据脚本

融合了清晰的配置与可解释的逻辑，旨在生成一个既真实又易于理解的数据集。
所有参数均在脚本顶部的【配置区】定义，便于调整和理解。
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# ==============================================================================
# --- 配置区 (Configuration Area) ---
# ==============================================================================

# --- 1. 基本参数 ---
N_SAMPLES = 200  # 生成样本数量
RANDOM_SEED = 42  # 随机种子，确保每次生成结果一致
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" # 保证路径正确
OUTPUT_FILE = OUTPUT_DIR / "admissions.csv"

# --- 2. 基础数据定义 ---
YEARS = [2021, 2022, 2023]
PROVINCES = [
    '北京', '上海', '广东', '江苏', '浙江', '山东', '河南', '四川',
    '湖北', '湖南', '河北', '福建', '安徽', '陕西', '辽宁'
]
SCHOOL_DATA = {
    '985': ['清华大学', '北京大学', '复旦大学', '上海交通大学', '浙江大学', '南京大学', '中山大学'],
    '211': ['北京邮电大学', '华东师范大学', '苏州大学', '上海财经大学', '中央财经大学', '武汉理工大学'],
    '普通本科': ['首都师范大学', '广东工业大学', '浙江工业大学', '山东科技大学', '河南大学', '成都理工大学']
}
MAJOR_DATA = {
    '理科': ['数学与应用数学', '物理学', '化学'],
    '工科': ['计算机科学与技术', '软件工程', '人工智能', '电子信息工程', '自动化', '机械工程'],
    '文科': ['汉语言文学', '历史学', '新闻学', '法学'],
    '综合': ['经济学', '金融学', '工商管理', '会计学']
}
HOT_MAJORS = [
    '计算机科学与技术', '软件工程', '人工智能', '金融学', '会计学'
]

# --- 3. 数据生成规则 (核心：简单、可解释的规则) ---
# (1) 不同层次学校的基础分数线
TIER_BASE_SCORES = {
    '985': 620,
    '211': 570,
    '普通本科': 500
}
# (2) 不同层次学校的基础热度指数 (报录比)
TIER_BASE_HOTNESS = {
    '985': 5.0,
    '211': 3.0,
    '普通本科': 1.8
}
# (3) 热门专业获得的额外加成
HOT_MAJOR_SCORE_BONUS = 30  # 热门专业分数普遍高30分
HOT_MAJOR_HOTNESS_MULTIPLIER = 1.8  # 热门专业热度再乘以1.8倍

# (4) 计划招生名额范围
PLAN_QUOTA_RANGE = (40, 150)

# (5) 排名转换规则：一个简化的分数到排名的映射
SCORE_TO_RANK_MAP = {
    650: (1000, 5000),
    600: (5000, 15000),
    550: (15000, 40000),
    500: (40000, 80000)
}

# ==============================================================================
# --- 数据生成逻辑 (Generation Logic) ---
# ==============================================================================

def generate_data(n_samples: int) -> pd.DataFrame:
    """根据顶部配置生成模拟数据"""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    data_records = []
    all_tiers = list(SCHOOL_DATA.keys())
    
    for _ in range(n_samples):
        # --- 步骤 1: 随机选择基础信息 ---
        year = random.choice(YEARS)
        province = random.choice(PROVINCES)
        school_tier = random.choice(all_tiers)
        school_name = random.choice(SCHOOL_DATA[school_tier])
        category = random.choice(list(MAJOR_DATA.keys()))
        major_name = random.choice(MAJOR_DATA[category])

        # --- 步骤 2: 根据规则计算核心数值 ---
        # (A) 计算分数
        base_score = TIER_BASE_SCORES[school_tier]
        score = base_score + np.random.randint(-15, 15)  # 增加随机波动
        if major_name in HOT_MAJORS:
            score += HOT_MAJOR_SCORE_BONUS

        # (B) 计算热度
        base_hotness = TIER_BASE_HOTNESS[school_tier]
        hotness = base_hotness * np.random.uniform(0.8, 1.2) # 增加随机波动
        if major_name in HOT_MAJORS:
            hotness *= HOT_MAJOR_HOTNESS_MULTIPLIER
            
        # (C) 生成计划名额和报考人数
        plan_quota = np.random.randint(*PLAN_QUOTA_RANGE)
        apply_num = int(plan_quota * hotness)
        apply_num = max(plan_quota, apply_num) # 保证报考人数不低于计划数
        
        # (D) 重新计算精确的热度指数
        hotness_index = round(apply_num / plan_quota, 2)

        # (E) 估算排名
        rank_range = (80000, 120000) # 默认排名范围
        for score_threshold, r_range in SCORE_TO_RANK_MAP.items():
            if score >= score_threshold:
                rank_range = r_range
                break
        min_score_rank = np.random.randint(*rank_range)

        data_records.append({
            'year': year,
            'province': province,
            'school_name': school_name,
            'school_tier': school_tier,
            'major_name': major_name,
            'category': category,
            'plan_quota': plan_quota,
            'apply_num': apply_num,
            'min_score': score,
            'min_score_rank': min_score_rank,
            'hotness_index': hotness_index
        })

    return pd.DataFrame(data_records)

def main():
    """主执行函数"""
    print("开始生成模拟招生数据...")
    
    df = generate_data(N_SAMPLES)
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存到CSV文件
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n成功！已生成 {len(df)} 条数据。")
    print(f"文件已保存至: {OUTPUT_FILE}")
    print("\n数据预览 (前5行):")
    print(df.head())
    print("\n数据统计信息:")
    print(df.describe())

if __name__ == "__main__":
    main()