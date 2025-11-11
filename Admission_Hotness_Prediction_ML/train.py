"""
模型训练脚本 (Model Training Script)
完成从数据读取到模型产出的完整离线训练流程
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import (
    DATA_PATH, MODEL_PATH, COLUMNS_PATH, 
    TARGET_COLUMN, MODEL_PARAMS
)


def load_data():
    """
    加载数据
    """
    print(f"正在加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，形状: {df.shape}")
    return df


def feature_engineering(df):
    """
    特征工程处理
    """
    print("正在进行特征工程...")
    df_processed = df.copy()
    
    # 1. 数值特征对数变换
    df_processed['log_plan_quota'] = np.log1p(df_processed['plan_quota'])
    df_processed['log_apply_num'] = np.log1p(df_processed['apply_num'])
    df_processed['log_min_score_rank'] = np.log1p(df_processed['min_score_rank'])
    
    # 2. 分类特征One-hot编码
    categorical_features = ['province', 'school_tier', 'category']
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)
    
    # 3. 移除不需要的特征
    features_to_remove = ['school_name', 'major_name', 'plan_quota', 'apply_num', 
                          'min_score_rank', 'hotness_index']
    for feature in features_to_remove:
        if feature in df_encoded.columns:
            df_encoded = df_encoded.drop(feature, axis=1)
    
    print(f"特征工程完成，特征数量: {df_encoded.shape[1]}")
    return df_processed, df_encoded


def prepare_data(df, df_encoded):
    """
    准备训练数据
    """
    print("正在准备训练数据...")
    X = df_encoded
    y = df[TARGET_COLUMN]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_model(X_train, y_train):
    """
    训练模型
    """
    print("正在训练模型...")
    
    # 创建随机森林回归模型
    model = RandomForestRegressor(**MODEL_PARAMS)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("模型训练完成")
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    print("正在评估模型性能...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"模型性能评估结果:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 重要特征:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'feature_importance': feature_importance
    }


def save_model(model, feature_columns):
    """
    保存模型和特征列
    """
    print("正在保存模型...")
    
    # 保存模型
    joblib.dump(model, MODEL_PATH)
    print(f"模型已保存到: {MODEL_PATH}")
    
    # 保存特征列
    joblib.dump(feature_columns, COLUMNS_PATH)
    print(f"特征列已保存到: {COLUMNS_PATH}")
    
    print("模型保存完成")


def main():
    """
    主函数：完整的模型训练流程
    """
    print("=" * 50)
    print("开始模型训练流程")
    print("=" * 50)
    
    try:
        # 1. 加载数据
        df = load_data()
        
        # 2. 特征工程
        df_processed, df_encoded = feature_engineering(df)
        
        # 3. 准备数据
        X_train, X_test, y_train, y_test, feature_columns = prepare_data(df, df_encoded)
        
        # 4. 训练模型
        model = train_model(X_train, y_train)
        
        # 5. 评估模型
        metrics = evaluate_model(model, X_test, y_test)
        
        # 6. 保存模型
        save_model(model, feature_columns)
        
        print("=" * 50)
        print("模型训练流程完成！")
        print(f"最终R²得分: {metrics['r2']:.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
