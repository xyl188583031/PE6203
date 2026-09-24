# 面霸 — AI PM校招面试模拟与复盘系统

> PE6203 GenAI Group Assignment · Stage 5 原型
> 仓库：https://github.com/xyl188583031/PE6203

## 这是什么？

一个面向AI产品经理校招方向的模拟面试系统（中英文双语界面），包含：

- **追问考官（Module A）**：结合RAG检索技术文档和面经，模拟大厂面试官，动态生成有压力的追问
- **STAR复盘专家（Module B）**：面试结束后逐题诊断，用STAR法则重构高分回答，标注穿帮风险
- **RAG知识库**：61条精选条目（23篇技术文档 + 10条面试经验 + 28条高频题/JD/STAR样例/评估量表等）
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

- Python 3.9 或更高版本（本地验证于 3.13.12）
- pip（Python 包管理器）

### 第1步：获取代码并安装依赖

```bash
git clone https://github.com/xyl188583031/PE6203.git
cd PE6203
pip install -r requirements.txt
```

> 首次启动时Chroma会自动下载embedding模型（约80MB），需要联网。下载后后续启动无需联网。
> `rag/chroma_db/` 未纳入版本控制，首次运行会依据 `rag/knowledge_base.json` 自动重建索引。

### 第2步：配置 API Key（可选）

不配也能跑 —— 系统会自动回落 **Mock 模式**，全部 UI 流程可走通。要接真实模型，二选一：

- 在「📋 面试配置」页的 **API Key** 输入框直接填写；或
- 在项目根目录新建 `.env`，写入 `OPENROUTER_API_KEY=你的密钥`

> `.env` 已在 `.gitignore` 中排除，**请勿提交或外发**。

### 第3步：启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

> **Windows下若遇到 `~/.streamlit` 目录权限错误**，设置环境变量：
> ```powershell
> $env:STREAMLIT_HOME = "$env:TEMP\streamlit_home"
> streamlit run app.py
> ```

### 第4步：开始使用

1. 在「📋 面试配置」页粘贴简历（或直接上传 PDF/Word 文件） + 选择JD + 选择面试官风格
2. 点击「🚀 开始面试」进入面试
3. 在「💬 模拟面试」页回答面试官问题（共8轮：AaBbCcDd）
4. 面试结束后点击「📊 生成复盘报告」
5. 在「复盘报告」页查看STAR诊断 + 穿帮预警 + 话术校准

## 6个页面说明

| 页面 | 功能 | 谁用 |
|---|---|---|
| 📋 面试配置 | 粘贴/上传简历、选JD、选面试官风格、设API Key | 所有人 |
| 💬 模拟面试 | 对话界面 + 右侧RAG面板 + CoT推理分析 | 所有人 |
| 📊 复盘报告 | STAR诊断卡 + 穿帮预警 + 话术校准 + 能力画像 | 所有人 |
| ⚖️ A/B/C对比 | 选测试用例→运行三组变体→对比追问深度 | 成员C |
| ⚙️ Prompt配置 | 改Module A/B Prompt + few-shot + 看token消耗 | 成员B |
| 📁 知识库管理 | 查看61条RAG条目 + 检索测试 + 重新索引 | 成员A |

## 文件结构

```
PE6203/
├── app.py                      ← Streamlit 主应用（6个页面，中英双语）
├── requirements.txt            ← 依赖列表
├── README.md                   ← 本文件
├── .gitignore                  ← 排除 .env / chroma_db / __pycache__ 等
├── .streamlit/
│   └── config.toml             ← Streamlit 主题与服务器配置
├── rag/
│   ├── knowledge_base.json     ← 61条知识库条目（8个类别）
│   ├── retriever.py            ← BM25 + Chroma 混合检索（支持类别筛选）
│   └── chroma_db/              ← Chroma 向量索引（自动生成，不入库）
├── core/
│   ├── llm_client.py           ← 多厂商LLM封装 + Token追踪 + Mock模式
│   ├── file_parser.py          ← 简历/JD 文件解析（PDF / Word(.docx) / TXT）
│   └── interview.py            ← 面试流程编排（大纲→ABCD主问→追问→复盘）
├── prompts/
│   ├── module_a.py             ← 追问考官 Prompt（大纲生成/术语提取/追问生成）
│   └── module_b.py             ← STAR复盘 Prompt
└── test_cases/
    └── cases.py                ← 20条测试用例（★ 成员C改这里）
```

## 输入与文件上传

- 简历框、JD 框都支持**直接上传文件**：`.pdf` / `.docx` / `.txt` / `.md`（旧版 `.doc` 会提示另存为 `.docx`）。
- 解析后的文字会填入文本框，仍可手动编辑；只有上传**新文件**时才会覆盖，手动修改不会被 rerun 冲掉。
- PDF 依赖 `pypdf`；`.docx` 优先用 `python-docx`，未安装时自动走标准库（zipfile + XML）兜底解析，仍可抽出正文与表格文字。

## RAG知识库（61条）

| 类别 | 条目数 | 内容 |
|---|---|---|
| tech_docs（技术文档） | 23 | AI前沿报告 + Qwen文档 + 阿里白皮书 |
| interview_tips（面试经验） | 10 | 面试技能 + 3层穿透验证 + 警惕信号 + 项目证据链 |
| AI PM校招高频题 | 8 | 大厂高频面试题与参考答法 |
| 企业真实JD | 5 | 真实岗位JD（用于关键词对齐） |
| STAR高分回答样例 | 5 | 可迁移的高分回答结构 |
| 行业评估量表 | 4 | 评分维度的行业参照标准 |
| 真实面试实录 | 3 | 完整面试对话记录 |
| 现代面试技巧方法论 | 3 | 方法论框架 |

## 测试用例（20条）

`test_cases/cases.py` 共 20 条，覆盖正常与挑战场景，供 Stage 6 对比 A/B/C 三变体使用：

| 类型 | 数量 | 具体场景 |
|---|---|---|
| 正常 | 12 | 自我介绍、STAR实习/失败经历、用户调研、优先级划分、增长策略、赋能场景、数据平台设计、效率数据、费米估算、产品设计题、留存策略 |
| 挑战-模糊 | 2 | 模糊回答、模糊数据 |
| 挑战-夸大 | 2 | 夸大经历、夸大数据 |
| 挑战-缺失 | 2 | 数据缺失、能力缺失 |
| 挑战-误导 | 1 | 答非所问 |
| 挑战-矛盾 | 1 | 前后矛盾 |

## 支持的LLM厂商

| 厂商 | 推荐模型 | 输入价格/1K | 输出价格/1K |
|---|---|---|---|
| OpenAI | GPT-4o | $0.0025 | $0.010 |
| DeepSeek | DeepSeek-V3 | $0.00014 | $0.00028 |
| 智谱AI (GLM) | GLM-4-Plus | $0.00139 | $0.00139 |
| 月之暗面 (Moonshot) | Moonshot-V1-8K | $0.0017 | $0.0034 |
| 通义千问 (Qwen) | Qwen-Plus | $0.00042 | $0.0014 |
| 豆包 (Doubao) | Doubao-Pro-4K | $0.00035 | $0.0007 |
| Anthropic | Claude 3.5 Sonnet | $0.003 | $0.015 |
| OpenRouter | GPT-4o mini | $0.00015 | $0.0006 |
| 自定义 | 任意OpenAI兼容API | - | - |

> 价格单位：美元 / 1K tokens，仅用于界面上的费用估算；每家还预置了若干可选模型。

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

1. 「📁 知识库管理」页查看全部61条
2. 按类别筛选（8个类别见上表）
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

本仓库已可直接部署：

1. 打开 [share.streamlit.io](https://share.streamlit.io) → 用 GitHub 账号登录
2. **New app** → 选择仓库 `xyl188583031/PE6203`，Branch 选 `main`，Main file path 填 `app.py`
3. （可选）在 **Advanced settings → Secrets** 中添加：
   ```toml
   OPENROUTER_API_KEY = "你的密钥"
   ```
   不配也可正常部署 —— 应用会回落 Mock 模式，UI 全流程可走通。
4. Deploy → 获得形如 `https://xxx.streamlit.app` 的公开 URL

> 注意：`requirements.txt` 中的依赖使用 `>=` 下限。若云端构建失败，首要排查项是 `chromadb`
> 的版本兼容性，可在 `requirements.txt` 中锁定本地验证过的版本
> （chromadb 1.5.9 / streamlit 1.63.0 / pypdf 6.18.0）。

## 对应作业Stage

| Stage | 对应文件 |
|---|---|
| Stage 2 — 架构 | `core/interview.py` + `app.py` |
| Stage 3 — Prompt | `prompts/module_a.py` + `prompts/module_b.py` |
| Stage 4 — RAG/ICL | `rag/knowledge_base.json` + `rag/retriever.py` + few-shot |
| Stage 5 — 原型 | `app.py`（Streamlit应用） |
| Stage 6 — 评估 | `test_cases/cases.py` + A/B/C对比页 |
| Stage 7 — 失败分析 | 运行测试后从对比结果中提取 |
