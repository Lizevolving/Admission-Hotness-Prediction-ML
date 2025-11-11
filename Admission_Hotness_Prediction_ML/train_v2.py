# -*- coding: utf-8 -*-
"""
训练脚本
目标：快速训练一个稳健的模型，可复现，可解释。
流程：读取数据 → 处理特征 → 检查数据泄露 → 交叉验证 → 最终训练 → 输出结果
"""

import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# 从 config.py 导入配置（保持 config 不变）
from config import (
    DATA_PATH, MODEL_PATH, COLUMNS_PATH,
    TARGET_COLUMN, MODEL_PARAMS
)

RANDOM_STATE = 42
CV_FOLDS = 5
LEAKAGE_CORR_THRESHOLD = 0.85 # 数值特征与目标的相关性超过此阈值时，认为可能泄露，会自动移除


def load_data(path):
    print(f"正在加载数据: {path}")
    df = pd.read_csv(path)
    print(f"数据加载完成，数据维度: {df.shape}")
    return df


def feature_engineering(df):
    """
    特征处理（简单透明）：
    1. 对部分数值做 log 转换，减少极端值影响
    2. 对类别变量做 one-hot（只保留必要的信息）
    3. 删除训练用不到的原始列，减少噪音
    """
    print("开始处理特征...")
    df_proc = df.copy()

    # log 转换（避免极端数值影响模型）
    for col in ['plan_quota', 'apply_num', 'min_score_rank']:
        if col in df_proc.columns:
            df_proc[f'log_{col}'] = np.log1p(df_proc[col])

    # one-hot 编码（把文字变成模型可识别的数字）
    categorical_features = [c for c in ['province', 'school_tier', 'category'] if c in df_proc.columns]
    df_encoded = pd.get_dummies(df_proc, columns=categorical_features, drop_first=True)

    # 删除训练中不需要的字段（如名称类文本）
    remove_candidates = ['school_name', 'major_name', 'plan_quota', 'apply_num', 'min_score_rank']
    for c in remove_candidates:
        if c in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=c)

    # 确保目标列不存在于特征表中（避免信息泄露）
    if TARGET_COLUMN in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=[TARGET_COLUMN])

    print(f"特征处理完成，共 {df_encoded.shape[1]} 个特征")
    return df_proc, df_encoded


def leakage_check_and_drop(df_proc, df_encoded):
    """
    检查是否有与目标高度相关的数值特征（> 阈值），避免“提前知道答案”导致作弊效果。
    如果发现，则自动移除。
    """
    print("检查特征与目标的相关性（避免数据泄露）...")

    if TARGET_COLUMN not in df_proc.columns:
        print("⚠️ 未找到目标列，跳过泄露检查")
        return df_encoded, []

    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COLUMN]
    corr = df_proc[numeric_cols + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False)

    to_drop = []
    for feat, corr_val in corr.items():
        if corr_val >= LEAKAGE_CORR_THRESHOLD:
            print(f"⚠️ 特征过于接近答案（可能泄露）: {feat} 与 {TARGET_COLUMN} 的相关性为 {corr_val:.3f}，已自动移除。")
            for c in [feat, f'log_{feat}']:
                if c in df_encoded.columns:
                    df_encoded = df_encoded.drop(columns=[c])
                    to_drop.append(c)

    if not to_drop:
        print(f"未发现相关性超过 {LEAKAGE_CORR_THRESHOLD} 的特征。")

    return df_encoded, to_drop


def evaluate_cv_baseline(model, X, y, cv=CV_FOLDS):
    """
    用 KFold 做交叉验证，看模型在不同数据分片上的表现是否稳定。
    同时对比一个“什么都不做的基线模型”（预测平均值）。
    """
    print(f"正在进行 {cv} 折交叉验证 (R²)...")
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    baseline = DummyRegressor(strategy='mean')
    baseline_scores = cross_val_score(baseline, X, y, cv=kf, scoring='r2')

    print(f"模型 R² 平均值: {scores.mean():.4f}  标准差: {scores.std():.4f}")
    print(f"基线（预测平均值）R²: {baseline_scores.mean():.4f}  标准差: {baseline_scores.std():.4f}")
    return scores, baseline_scores


def train_final_and_evaluate(model, X_train, y_train, X_test, y_test):
    print("正在训练最终模型，并评估测试集...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("测试集结果：")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred
    }


def permutation_importance_report(model, X_test, y_test, n_repeats=20):
    print("正在计算特征重要性（置换重要性）...")
    res = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)

    imp_df = pd.DataFrame({
        'feature': X_test.columns,
        'perm_importance_mean': res.importances_mean,
        'perm_importance_std': res.importances_std
    }).sort_values('perm_importance_mean', ascending=False)

    print("\n最重要的前 10 个特征：")
    for _, r in imp_df.head(10).iterrows():
        print(f"  {r['feature']}: mean={r['perm_importance_mean']:.4f} std={r['perm_importance_std']:.4f}")

    return imp_df


def save_artifacts(model, feature_columns, metrics, out_model_path=MODEL_PATH, out_columns_path=COLUMNS_PATH, out_metrics_path=None):
    os.makedirs(Path(out_model_path).parent, exist_ok=True)
    os.makedirs(Path(out_columns_path).parent, exist_ok=True)

    joblib.dump(model, out_model_path)
    joblib.dump(feature_columns, out_columns_path)

    print(f"✅ 模型已保存: {out_model_path}")
    print(f"✅ 特征列表已保存: {out_columns_path}")

    if out_metrics_path:
        with open(out_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"✅ 评估结果已保存: {out_metrics_path}")


def main():
    print("=" * 50)
    print("开始执行训练流程")
    print("=" * 50)

    df = load_data(DATA_PATH)

    df_proc, df_encoded = feature_engineering(df)

    df_encoded, dropped_features = leakage_check_and_drop(df_proc, df_encoded)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"找不到目标列: {TARGET_COLUMN}")

    X = df_encoded.copy()
    y = df[TARGET_COLUMN].copy()

    if X.shape[0] != y.shape[0]:
        raise ValueError("X 与 y 的行数不一致，请检查数据。")

    rf = RandomForestRegressor(**MODEL_PARAMS)

    scores, baseline_scores = evaluate_cv_baseline(rf, X, y, cv=CV_FOLDS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    final_metrics = train_final_and_evaluate(rf, X_train, y_train, X_test, y_test)

    perm_imp_df = permutation_importance_report(rf, X_test, y_test, n_repeats=20)

    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    metrics_to_save = {
        'cv_r2_mean': float(scores.mean()), 'cv_r2_std': float(scores.std()),
        'baseline_cv_r2_mean': float(baseline_scores.mean()), 'baseline_cv_r2_std': float(baseline_scores.std()),
        'test_metrics': {k: float(v) for k, v in final_metrics.items() if k in ['mse','rmse','mae','r2']},
        'dropped_features': dropped_features,
        'top_permutation_importance': perm_imp_df.head(10).to_dict(orient='records'),
        'top_feature_importance': feat_imp.head(10).to_dict(orient='records')
    }

    metrics_path = Path(MODEL_PATH).with_suffix('.metrics.json')
    save_artifacts(rf, X.columns.tolist(), metrics_to_save,
                   out_model_path=MODEL_PATH,
                   out_columns_path=COLUMNS_PATH,
                   out_metrics_path=metrics_path)

    print("=" * 50)
    print("🎉 训练完成")
    print(f"交叉验证 R² 平均值: {scores.mean():.4f}")
    print(f"测试集 R²: {final_metrics['r2']:.4f}")
    if dropped_features:
        print(f"以下特征因为与目标过于相关，已被自动移除: {dropped_features}")
    print("=" * 50)


if __name__ == "__main__":
    main()
