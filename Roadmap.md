### **项目开发路线图 (Roadmap & Sprint Plan)**

**项目:** 高校招生热度分析与预测系统 (College Admission Hotness Analysis & Prediction System)
**目标:** 构建一个健壮、可展示、遵循最佳实践的机器学习应用原型。

#### **核心策略与决策 (Key Decisions)**

*   **依赖管理:** 采用 `PDM`，摒弃 `venv` + `requirements.txt`，保持项目现代化。
*   **架构:** `模型训练 (train.py)` 与 `应用服务 (app.py)` 彻底分离，保证模块化和可维护性。
*   **数据流:** `CSV -> Pandas DataFrame -> Scikit-learn Model -> Streamlit UI`，路径最短，无数据库开销。
*   **版本控制:** `Git` 全程管理，分支策略从简（主干开发），但 `commit` 信息需清晰。

---

### **Sprint 0: 项目奠基 (Foundation)**

**目标:** 搭建一个专业、规范、可立即开始编码的项目框架。

*   **1. 初始化 Git 仓库:**
    *   `git init`
    *   创建`.gitignore`，至少忽略：`__pycache__/`, `*.pkl`, `.pdm-build/`, `*.ipynb_checkpoints`, `.DS_Store`。

*   **2. 建立项目结构:**
    *   一次性创建所有规划中的文件和目录，保持结构清晰。
        ```bash
        mkdir -p data models
        touch pyproject.toml README.md data_dictionary.md data/admissions.csv eda_and_feature_engineering.ipynb train.py app.py .gitignore
        ```

*   **3. 配置 PDM 环境:**
    *   `pdm init`
    *   `pdm add streamlit pandas scikit-learn joblib`，锁定核心依赖。

*   **4. 数据定义与生成:**
    *   **[关键]** 在 `data_dictionary.md` 中明确定义每个字段的类型、含义和约束。这是后续所有工作的基础。
    *   生成高质量的模拟数据（~100-200条），确保覆盖不同年份、层次、科类，数据分布要合理，避免模型训练失败。

---

### **Sprint 1: 模型核心管道 (Core Model Pipeline)**

**目标:** 完成从数据读取到模型产出的完整离线训练流程。

*   **1. 探索性数据分析 (EDA):**
    *   在 `eda_and_feature_engineering.ipynb` 中进行。
    *   目标：理解数据分布、特征关系，确定特征工程方案（如：哪些分类变量需要 one-hot 编码）。
    *   产出：清晰的特征工程逻辑，为 `train.py` 的编写提供依据。

*   **2. 编写模型训练脚本 (`train.py`):**
    *   **函数化/模块化:** 将数据加载、特征工程、模型训练、模型保存等步骤封装成独立的函数。
    *   **配置集中化**: 创建config.py文件，用来存放所有文件路径（如DATA_PATH = 'data/admissions.csv'）和模型参数。train.py和app.py都从这里导入，避免硬编码，便于维护。
    *   **核心逻辑:**
        *   加载 `data/admissions.csv`。
        *   执行在 EDA 中确定的特征工程（如 `pd.get_dummies`）。
        *   **[关键]** 训练模型后，必须使用 `joblib` 同时序列化 `model` 和 `feature_columns` 列表。
        ```python
        # train.py 伪代码
        joblib.dump(model, 'models/admissions_model.pkl')
        joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
        ```
    *   **验证:** 运行 `pdm run python train.py`，确保 `models/` 目录下成功生成两个 `.pkl` 文件。

---

### **Sprint 2: 应用层实现 (Application Layer)**

**目标:** 构建一个功能完整、交互流畅的 Streamlit 应用。

*   **1. 应用骨架与资源加载 (`app.py`):**
    *   在应用启动时，一次性加载模型和特征列，存入全局变量或缓存，避免重复IO。
    *   使用 `st.set_page_config` 设置页面标题和布局。

*   **2. 构建用户输入界面:**
    *   使用 `st.sidebar` 组织所有输入控件，保持主页面清爽。
    *   为输入控件设置合理的默认值和范围（e.g., `min_value`, `max_value` for `st.number_input`）。

*   **3. 实现预测与展示逻辑:**
    *   **输入处理:** 当用户点击“预测”按钮后，收集所有输入值，构造成一个单行的 Pandas DataFrame。
    *   **[关键]** 对齐特征：
        1.  对用户输入的分类变量进行 one-hot 编码。
        2.  使用加载的 `feature_columns` 列表，通过 `df.reindex(columns=feature_columns, fill_value=0)` 来确保预测数据的维度和顺序与训练时完全一致。
    *   **结果输出:**
        *   调用 `model.predict()`。
        *   将预测结果（一个数值）与相关信息（如模拟的大学列表）整合成一个结果 DataFrame。
        *   使用 `st.dataframe` 或 `st.table` 展示，并按热度指数降序排序。
        *   增加 `st.success` 或 `st.info` 提示，优化交互体验。

---

### **Sprint 3: 质量保障与文档完善 (QA & Documentation)**

**目标:** 确保项目代码健壮、文档清晰，达到可交付状态。

*   **1. 代码自检与重构:**
    *   清理所有 `print` 调试语句。
    *   为关键函数添加简洁的文档字符串 (docstrings)。
    *   确保代码风格一致 (可使用 `pdm run ruff format .` 和 `pdm run ruff check --fix .` 自动化)。

*   **2. 完善 `README.md`:**
    *   确保所有安装、运行指令准确无误（特别是 PDM 相关指令）。
    *   补充项目特色、技术栈、工作流程等章节。
    *   嵌入最终的应用截图 (`screenshot.png`)。

*   **3. 最终测试:**
    *   完整地从 `git clone` 开始，按 `README.md` 的指引走一遍流程，确保任何人都可复现。
    *   测试边界输入，确保应用不会因非预期输入而崩溃。

