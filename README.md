# 面霸 — AI PM校招面试模拟与复盘系统

## 这是什么？

一个面向AI产品经理校招方向的模拟面试系统，包含：
- **追问考官（Module A）**：结合RAG检索技术文档和面经，模拟大厂面试官，动态生成有压力的追问
- **STAR复盘专家（Module B）**：面试结束后逐题诊断，用STAR法则重构高分回答，标注穿帮风险
- **RAG知识库**：33条精选条目（23篇技术文档 + 10条面试经验）
- **CoT推理分析**：追问前对面试者回答进行推理分析，再生成追问
- **全量日志记录**：8轮问答（AaBbCcDd）全部入库，供复盘环节逐题诊断
- **A/B/C三组对比**：基础LLM vs 简化系统 vs 完整系统

## 面试流程

```
面试配置（选简历/JD/面试官风格）
    ↓
生成面试大纲（后台，用户不可见）
    ↓
A板块·主问题 → A板块·追问（CoT推理 → RAG检索 → 生成追问）
    ↓
B板块·主问题 → B板块·追问
    ↓
C板块·主问题 → C板块·追问
    ↓
D板块·主问题 → D板块·追问
    ↓
复盘报告（两阶段RAG → STAR诊断 → 穿帮预警 → 话术校准）
```

## 快速开始

### 环境要求

- Python 3.9 或更高版本
- pip（Python 包管理器）

### 第1步：安装依赖

```bash
cd mianba
pip install -r requirements.txt
```

> 首次启动时Chroma会自动下载embedding模型（约80MB），需要联网。下载后后续启动无需联网。

### 第2步：启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

> **Windows下若遇到 `~/.streamlit` 目录权限错误**，设置环境变量：
> ```powershell
> $env:STREAMLIT_HOME = "$env:TEMP\streamlit_home"
> streamlit run app.py
> ```

### 第3步：开始使用

1. 在「📋 面试配置」页粘贴简历 + 选择JD + 选择面试官风格
2. 点击「🚀 开始面试」进入面试
3. 在「💬 模拟面试」页回答面试官问题（共8轮：AaBbCcDd）
4. 面试结束后点击「📊 生成复盘报告」
5. 在「复盘报告」页查看STAR诊断 + 穿帮预警 + 话术校准

> **没有API Key？** 直接用Mock模式即可体验全部UI流程。接入真实API后追问质量会大幅提升。

## 6个页面说明

| 页面 | 功能 | 谁用 |
|---|---|---|
| 📋 面试配置 | 粘贴简历、选JD、选面试官风格、设API Key | 所有人 |
| 💬 模拟面试 | 对话界面 + 右侧RAG面板 + CoT推理分析 | 所有人 |
| 📊 复盘报告 | STAR诊断卡 + 穿帮预警 + 话术校准 + 能力画像 | 所有人 |
| ⚖️ A/B/C对比 | 选测试用例→运行三组变体→对比追问深度 | 成员C |
| ⚙️ Prompt配置 | 改Module A/B Prompt + few-shot + 看token消耗 | 成员B |
| 📁 知识库管理 | 查看33条RAG条目 + 检索测试 + 重新索引 | 成员A |

## 文件结构

```
mianba/
├── app.py                      ← Streamlit 主应用（6个页面）
├── requirements.txt            ← 依赖列表
├── README.md                   ← 本文件
├── rag/
│   ├── knowledge_base.json     ← 33条知识库条目（23技术文档 + 10面试经验）
│   ├── retriever.py            ← BM25 + Chroma 混合检索（支持类别筛选）
│   └── chroma_db/              ← Chroma 向量索引（自动生成）
├── core/
│   ├── llm_client.py           ← 多厂商LLM封装 + Token追踪 + Mock模式
│   ├── file_parser.py          ← 简历/JD 文件解析（PDF / Word(.docx) / TXT）
│   └── interview.py            ← 面试流程编排（大纲→ABCD主问→追问→复盘）
├── prompts/
│   ├── module_a.py             ← 追问考官 Prompt（大纲生成/术语提取/追问生成）
│   └── module_b.py             ← STAR复盘 Prompt
└── test_cases/
    └── cases.py                ← 测试用例（★ 成员C改这里）
```

## 输入与文件上传

- 简历框、JD 框都支持**直接上传文件**：`.pdf` / `.docx` / `.txt` / `.md`（旧版 `.doc` 会提示另存为 `.docx`）。
- 解析后的文字会填入文本框，仍可手动编辑；只有上传**新文件**时才会覆盖，手动修改不会被 rerun 冲掉。

## RAG知识库（33条）

| 类别 | 条目数 | 内容 |
|---|---|---|
| tech_docs（技术文档） | 23 | 20篇AI前沿报告 + 2篇Qwen文档 + 1篇阿里白皮书 |
| interview_tips（面试经验） | 10 | 阿酥面试技能 + 3层穿透验证 + 5类警惕信号 + 项目证据链 |

## 支持的LLM厂商

| 厂商 | 推荐模型 | 输入价格/1K | 输出价格/1K |
|---|---|---|---|
| OpenAI | GPT-4o | $0.0025 | $0.010 |
| DeepSeek | DeepSeek-V3 | $0.00014 | $0.00028 |
| 通义千问 | Qwen-Plus | $0.00040 | $0.00120 |
| 智谱AI | GLM-4-Flash | 免费至0.0001 | 免费至0.0001 |
| 自定义 | 任意OpenAI兼容API | - | - |

## 队友协作指南

### 成员B：调Prompt

**方式1（推荐）：在应用内改**
1. 打开应用 → 左侧导航 → 「⚙️ Prompt配置」
2. 在文本框中修改Module A或B的Prompt
3. 点击「保存」→ 下次面试立即生效
4. Token消耗在页面底部实时显示

**方式2：改代码文件**
- Module A Prompt → `prompts/module_a.py` 中的 `MODULE_A_SYSTEM_PROMPT`
- Module B Prompt → `prompts/module_b.py` 中的 `MODULE_B_SYSTEM_PROMPT`
- Few-shot 示例 → `prompts/module_a.py` 中的 `FEW_SHOT_EXAMPLES`
- 改完保存即可，Streamlit会自动重载

### 成员C：跑测试

1. 打开应用 → 「⚖️ A/B/C对比」
2. 选择测试用例
3. 点击「运行三组对比」
4. 查看三组追问序列和穿透层分布

### 成员A：管知识库

1. 「📁 知识库管理」页查看全部33条
2. 按类别筛选（tech_docs / interview_tips）
3. 底部「检索测试」可验证检索效果
4. 要改知识库 → 编辑 `rag/knowledge_base.json` → 回到页面点「重新索引」

## 常见问题

**Q: 没有API Key能用吗？**
A: 能。系统自动切换到Mock模式，可体验全部UI流程，但追问内容是预设的。接入真实API后追问质量会大幅提升。

**Q: 改了Prompt不生效？**
A: 确保在「Prompt配置」页点了「保存」按钮。如果改的是代码文件，重启 `streamlit run app.py`。

**Q: 改了知识库后需要做什么？**
A: 编辑 `rag/knowledge_base.json` → 在「知识库管理」页点「重新索引」。不需要改任何代码。

**Q: Token消耗怎么查？**
A: 「Prompt配置」页底部有实时Token追踪，包括总调用次数、总Token、平均Token/次、估计费用。

## 部署上线

### Streamlit Cloud（推荐）

1. 把 `mianba/` 文件夹上传到GitHub仓库
2. 打开 [share.streamlit.io](https://share.streamlit.io)
3. 连接GitHub仓库，选择 `app.py`
4. 在Secrets中添加API Key
5. 部署完成 → 获得公开URL

## 对应作业Stage

| Stage | 对应文件 |
|---|---|
| Stage 2 — 架构 | `core/interview.py` + `app.py` |
| Stage 3 — Prompt | `prompts/module_a.py` + `prompts/module_b.py` |
| Stage 4 — RAG/ICL | `rag/knowledge_base.json` + `rag/retriever.py` + few-shot |
| Stage 5 — 原型 | `app.py`（Streamlit应用） |
| Stage 6 — 评估 | `test_cases/cases.py` + A/B/C对比页 |
| Stage 7 — 失败分析 | 运行测试后从对比结果中提取 |
