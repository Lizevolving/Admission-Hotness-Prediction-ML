# 高校招生热度分析与预测系统 (College Admission Hotness Analysis & Prediction System)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

一个轻量级的、基于机器学习的数据应用，旨在分析和预测高校专业的招生热度，为考生提供数据驱动的报考参考。

![Demo Screenshot](https://raw.githubusercontent.com/your-username/your-repo/main/assets/demo.png)  
*(请将此处的图片链接替换为你的项目截图)*

---

## 🚀 项目特色 (Features)

*   **交互式预测**: 用户只需输入分数、省内排名和科类，即可获得相关专业的预测热度排名。
*   **数据驱动**: 基于历史招生数据，利用机器学习模型（随机森林）进行热度指数预测。
*   **极简架构**: 采用 Streamlit 构建，整个应用由单一 Python 脚本驱动，无需复杂的前后端分离和数据库配置。
*   **快速部署**: 最小化依赖，通过简单的命令行指令即可在本地运行。
*   **模块化设计**: 训练脚本与应用脚本分离，方便模型的独立更新与迭代。

## 🛠️ 技术栈 (Technology Stack)

*   **应用框架**: Streamlit
*   **数据处理**: Pandas
*   **机器学习**: Scikit-learn
*   **模型持久化**: Joblib
*   **核心语言**: Python 3.8+

## ⚡ 快速开始 (Quick Start)

请确保你已安装 Python 3.8+、Git 和 PDM。

**1. 克隆仓库**
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

**2. 初始化 PDM 环境**
```bash
# 初始化项目并安装依赖
pdm install
```

**3. 运行应用**
```bash
pdm run streamlit run app.py
```
应用将在你的默认浏览器中自动打开，地址通常为 `http://localhost:8501`。

## 📁 项目结构 (Project Structure)

```
.
├── pyproject.toml                      # 现代化的项目定义与依赖管理 (PDM)
├── README.md                           # 项目说明、运行指南（门面）
├── data_dictionary.md                  # 数据字典，解释数据字段（严谨性）
├── config.py                           # 统一管理所有配置，确保单一事实来源
│  
├── admissions.csv                      # 1. 原始数据源
├── eda_and_feature_engineering.ipynb   # 2. 数据探索与分析过程
├── train.py                            # 3. 模型训练脚本
├── admissions_model.pkl                # 4. 训练产物：模型文件
├── app.py                              # 5. 最终应用：可交互的演示系统
│  
├── screenshot.png                      # 用于README的系统截图
└── .gitignore                          

```

## 🔄 工作流程 (Workflow)

本项目的工作流程分为两个核心阶段：

**1. 模型训练 (Offline)**
   - 运行 `train.py` 脚本。
   - 该脚本会读取 `data/data.csv` 中的原始数据。
   - 对数据进行预处理和特征工程。
   - 使用 `RandomForestRegressor` 训练模型。
   - 将训练好的模型保存为 `model/model.pkl`，并将特征列保存到 `model/feature_columns.pkl`。
   - **注意**: 当数据更新或模型算法需要调整时，重新执行此脚本即可。
   ```bash
   pdm run python train.py
   ```

**2. 应用服务 (Online)**
   - 运行 `app.py` 脚本。
   - 应用启动时，会加载 `model/` 目录下的模型和特征列文件。
   - Streamlit 框架会渲染一个Web用户界面。
   - 用户在界面上输入参数后，应用将输入数据构造成符合模型要求的特征向量。
   - 调用加载的模型进行预测，并将结果实时展示在前端。


---

## 📊 关于数据 (About the Data)

本项目使用的 `data.csv` 是一个用于演示目的的样本数据集，其结构经过精心设计以平衡真实性与简洁性。包含以下关键字段：

*   `year`: 招生年份。
*   `province`: 招生省份，如 '北京', '四川'。
*   `school_name`: 高校名称。
*   `school_tier`: 高校层次，一个关键的分类特征（如 '985', '211', '普通'）。
*   `major_name`: 专业名称。
*   `category`: 考生科类，如 '理科', '文科'。
*   `plan_quota`: 计划招生名额。
*   `apply_num`: 实际报考人数。
*   `min_score`: 当年该专业的最低录取分数。
*   `min_score_rank`: 最低录取分数对应的省内排名，一个比原始分数更具可比性的关键特征。
*   `hotness_index`: **目标变量 (Target)**，代表报考热度指数。它由 `apply_num / plan_quota` (报录比) 计算得出，直观地反映了竞争激烈程度。


在实际应用中，此数据集可替换为来自官方渠道的真实、大规模数据。

## 📝 未来的工作 (Future Work)

- [ ] 集成更丰富的数据源（如：就业率、师资力量、社会评价）。
- [ ] 尝试更多样的模型（如：LightGBM, XGBoost）并进行性能对比。
- [ ] 增加数据可视化模块，展示历史热度变化趋势。
- [ ] 将应用容器化（使用 Docker）并部署到云平台。

## 📄 许可证 (License)

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 授权。