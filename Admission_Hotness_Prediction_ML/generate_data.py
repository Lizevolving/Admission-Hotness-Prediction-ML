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
# --- 配置区  (Configuration Area ) ---
# ==============================================================================

# --- 1. 基本参数 ---
N_SAMPLES = 300      # 样本量设定
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "admissions.csv"


# --- 2. 基础数据定义 (核心升级区) ---
YEARS = [2021, 2022, 2023, 2024, 2025]  # 包含历史年份数据

PROVINCES = [  # 使用的省份列表
    '北京','上海','广东','江苏','浙江','山东','河南','四川','湖北',
    '湖南','河北','福建','安徽','陕西','辽宁','天津','黑龙江','吉林','重庆'
]

# 院校数据，分为985, 211, 普通本科三个层次

SCHOOL_DATA = {
    '985': [
        '清华大学', '北京大学', '复旦大学', '上海交通大学', '浙江大学', '南京大学',
        '中国科学技术大学', '哈尔滨工业大学', '西安交通大学', '中山大学', '武汉大学',
        '华中科技大学', '四川大学', '北京航空航天大学', '同济大学', '东南大学',
        '南开大学', '天津大学', '山东大学', '中南大学'
    ], 

    '211': [
        '北京邮电大学', '上海财经大学', '中央财经大学', '对外经济贸易大学', '西安电子科技大学',
        '武汉理工大学', '南京航空航天大学', '苏州大学', '华东理工大学', '暨南大学',
        '合肥工业大学', '西南交通大学', '华南师范大学', '北京科技大学', '中国传媒大学'
    ], 

    '普通本科': [
        '首都师范大学', '广东工业大学', '浙江工业大学', '深圳大学', '杭州电子科技大学',
        '山东科技大学', '河南大学', '成都理工大学', '上海理工大学', '燕山大学',
        '扬州大学', '南京工业大学', '东北财经大学', '天津工业大学', '重庆邮电大学'
    ] 
}


# 专业数据，覆盖12个学科门类
MAJOR_DATA = {
    '工学': [
        '计算机科学与技术', '软件工程', '人工智能', '电子信息工程', '通信工程',
        '机械设计制造及其自动化', '电气工程及其自动化', '自动化', '数据科学与大数据技术'
    ],
    '理学': ['数学与应用数学', '物理学', '化学', '统计学', '生物科学'],
    '管理学': ['会计学', '工商管理', '财务管理', '人力资源管理', '市场营销'],
    '经济学': ['经济学', '金融学', '财政学', '国际经济与贸易'],
    '医学': ['临床医学', '口腔医学', '护理学', '药学'],
    '文学': ['汉语言文学', '英语', '新闻学', '播音与主持艺术'],
    '法学': ['法学', '政治学', '社会学'],
    '哲学': ['哲学', '逻辑学'],
    '历史学': ['历史学', '世界史', '考古学'],
    '教育学': ['教育学', '学前教育'],
    '农学': ['植物生产', '动物医学'],
    '艺术学': ['美术学', '设计学']
}


# 定义热门专业列表

HOT_MAJORS = [
    '计算机科学与技术', '软件工程', '人工智能', '电子信息工程', '通信工程',
    '电气工程及其自动化', '临床医学', '口腔医学', '金融学', '会计学', '法学'
]


# --- 3. 数据生成规则 (保持不变，维持项目核心的简单性与可控性) ---

TIER_BASE_SCORES = {
    '985': 630,       # 985院校基础分数设定
    '211': 580,
    '普通本科': 520
}
TIER_BASE_HOTNESS = {
    '985': 5.0,
    '211': 3.0,
    '普通本科': 1.8
}

HOT_MAJOR_SCORE_BONUS = 25 # 热门专业分数加成
HOT_MAJOR_HOTNESS_MULTIPLIER = 1.8
PLAN_QUOTA_RANGE = (40, 150)

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