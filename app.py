"""MianBa — AI PM Interview Simulator & Review System
Streamlit Main Application (Bilingual: EN/ZH)
"""

import json
import os
import sys
import copy

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.retriever import HybridRetriever
from core.llm_client import (
    LLMClient,
    MockLLMClient,
    PROVIDER_PRESETS,
    get_model_pricing,
    load_env_api_key,
    provider_model_label,
)
from core.file_parser import extract_text_from_upload, UPLOAD_EXTENSIONS
from core.interview import InterviewSession, run_variant_test, SECTIONS, FOLLOWUPS_PER_SECTION
from prompts.module_a import MODULE_A_SYSTEM_PROMPT, MODULE_A_SYSTEM_PROMPT_EN, FEW_SHOT_EXAMPLES
from prompts.module_b import MODULE_B_SYSTEM_PROMPT, MODULE_B_SYSTEM_PROMPT_EN
from test_cases.cases import TEST_CASES, get_case_by_id

# ========== i18n ==========
I18N = {
    "en": {
        # Sidebar
        "app_title": "MianBa",
        "app_subtitle": "AI PM Interview Simulator & Review",
        "nav": "Navigation",
        "quick_start": "Quick Start",
        "api_notes": "API Notes",
        # Pages
        "page_setup": "Interview Setup",
        "page_interview": "Mock Interview",
        "page_review": "Review Report",
        "page_comparison": "A/B/C Comparison",
        "page_prompt": "Prompt Config",
        "page_kb": "Knowledge Base",
        # Setup page
        "setup_title": "Interview Setup",
        "setup_desc": "Configure your resume, target JD, and interviewer style. Click Start when ready.",
        "resume_title": "Resume Summary",
        "resume_placeholder": "e.g., 3-month internship at ByteDance AI Product Dept, working on AI customer service, responsible for RAG optimization, data labeling, effect validation...",
        "resume_help": "List your internship companies, roles, projects, and skills. Full resume not required.",
        "resume_upload_label": "Or upload a resume file (PDF / Word / TXT)",
        "resume_upload_help": "Supported: .pdf, .docx, .txt, .md. Text is extracted and filled into the box below — you can still edit it.",
        "jd_upload_label": "Or upload a JD file (PDF / Word / TXT)",
        "jd_upload_help": "Supported: .pdf, .docx, .txt, .md. Text is extracted and filled into the box below.",
        "upload_reading": "Reading file...",
        "upload_ok": "Loaded from file — please review the text below.",
        "upload_clear": "Clear uploaded file",
        "style_title": "Interviewer Style",
        "style_select": "Select style (locked after interview starts)",
        "jd_title": "Target Job Description (JD)",
        "jd_preset": "Select preset JD",
        "jd_custom": "Custom",
        "jd_placeholder": "Paste the target job description here...",
        "jd_editable": "JD content (editable)",
        "api_title": "API Settings",
        "api_key_label": "API Key (empty = Mock mode)",
        "api_key_help": "Leave empty to use Mock mode — full flow but limited answer quality.",
        "api_key_env": "✅ OpenRouter key auto-loaded from .env (edit or clear it; blank falls back to Mock mode)",
        "llm_provider": "LLM Provider",
        "model": "Model",
        "stats_sections": "Sections",
        "stats_followups": "Follow-ups/section",
        "stats_rounds": "Total Rounds",
        "stats_kb": "RAG KB Entries",
        "start_btn": "Start Interview",
        "warn_no_resume": "Please paste your resume summary first.",
        "warn_no_jd": "Please provide the target job description.",
        "spinner_first_q": "Interviewer is preparing the first question...",
        # Interview page
        "interview_title": "Mock Interview",
        "warn_no_setup": "Please complete setup on the Interview Setup page first.",
        "go_setup": "Go to Interview Setup",
        "metric_progress": "Progress",
        "metric_style": "Interviewer Style",
        "metric_api_calls": "API Calls",
        "metric_tokens": "Tokens",
        "qtype_main": "Main",
        "qtype_followup": "Follow-up",
        "label_round": "Round",
        "label_probing_layer": "Probing Layer",
        "label_fabrication_risk": "Fabrication Risk",
        "label_intent": "Intent",
        "label_jd_kw": "JD Keywords",
        "cot_header": "CoT Reasoning",
        "cot_weaknesses": "Answer Weaknesses:",
        "cot_angle": "Probing Angle:",
        "interview_end_msg": "Interview complete! 8 rounds finished (AaBbCcDd).",
        "interview_end_hint": "Click the button below to generate a detailed review report.",
        "btn_review": "Generate Review Report",
        "btn_restart": "Restart Interview",
        "chat_placeholder": "Type your answer here...",
        "spinner_analyzing": "Interviewer is analyzing...",
        "rag_title": "RAG Retrieval",
        "rag_empty": "RAG results will appear here",
        "analysis_title": "Current Analysis",
        "label_section": "Section",
        "label_type": "Type",
        # Review page
        "review_title": "Review Report",
        "review_generate_btn": "Generate Review Report",
        "review_spinner": "Analyzing your interview performance...",
        "warn_no_interview": "Please complete a mock interview first.",
        "review_total_score": "Total Score",
        "review_comment": "Comment:",
        "review_advantages": "Strengths",
        "review_shortcomings": "Weaknesses",
        "review_style_risks": "Style Risks",
        "review_per_q": "Per-Question Diagnosis",
        "review_your_answer": "Your Answer:",
        "review_score_reason": "Score Reason:",
        "review_star": "STAR Diagnosis",
        "review_star_s": "Situation",
        "review_star_t": "Task",
        "review_star_a": "Action",
        "review_star_r": "Result",
        "review_cap_gap": "Capability Gaps",
        "review_fab_risk": "Fabrication Risk Warnings",
        "review_calibration": "Resume Calibration",
        "review_reconstructed": "High-Score Answer Reconstruction",
        "review_improvement": "Improvement Suggestion",
        "review_growth": "Growth Path",
        "review_short_term": "Short-term (before next interview)",
        "review_long_term": "Long-term (1-3 months)",
        "review_usage": "Review Tokens",
        "review_cost": "Cost",
        "btn_abc": "View A/B/C Comparison",
        # Comparison page
        "cmp_title": "A/B/C Three-Group Comparison",
        "cmp_desc": "Run the same test case through three variants to measure incremental value.",
        "cmp_select_case": "Select test case",
        "cmp_run": "Run Comparison",
        "cmp_variant_a": "A — Base LLM",
        "cmp_variant_b": "B — Simplified",
        "cmp_variant_c": "C — Full System",
        "cmp_questions": "Question Sequence:",
        "cmp_probing": "Probing Layer Distribution:",
        "cmp_times": "times",
        # Prompt page
        "prompt_title": "Prompt Configuration",
        "prompt_desc": "Adjust Module A/B prompts and few-shot examples.",
        "prompt_tab_a": "Module A — Interviewer",
        "prompt_tab_b": "Module B — STAR Review",
        "prompt_tab_fs": "Few-shot",
        "prompt_save": "Save",
        "prompt_saved": "Saved!",
        "prompt_fs_type": "Type",
        "prompt_fs_answer": "User Answer",
        "prompt_fs_ideal": "Ideal Follow-up",
        "prompt_fs_layer": "Probing Layer",
        "prompt_fs_risk": "Fabrication Risk",
        "prompt_save_fs": "Save Few-shot",
        "prompt_usage_title": "Token Usage",
        "prompt_total_calls": "Total API Calls",
        "prompt_total_tokens": "Total Tokens",
        "prompt_avg_tokens": "Avg Tokens/Call",
        "prompt_cost": "Cost ($)",
        "prompt_reset": "Reset",
        # KB page
        "kb_title": "Knowledge Base",
        "kb_entries": "KB Entries",
        "kb_filter": "Filter by category",
        "kb_all": "All",
        "kb_search_title": "Search Test",
        "kb_search_placeholder": "e.g., internship responsible for AI customer service product...",
        "kb_search_btn": "Search",
        "kb_reindex": "Re-index",
        "kb_reindexing": "Rebuilding index...",
        "kb_reindex_done": "Done!",
        "kb_relevance": "Relevance",
        "kb_no_title": "Untitled",
        # Quick start guide
        "qs_1": "1. **Interview Setup**: Paste resume + select JD + select style",
        "qs_2": "2. **Mock Interview**: 8 rounds (AaBbCcDd)",
        "qs_3": "3. **Review Report**: STAR diagnosis + fabrication warnings",
        "qs_4": "4. **A/B/C Comparison**: Three-variant comparison",
        "qs_5": "5. **Prompt Config**: Adjust prompts",
        "qs_6": "6. **Knowledge Base**: Browse / test RAG",
        "api_note_1": "- No API Key = Mock mode (simulated answers)",
        "api_note_2": "- Connect real API for higher quality follow-ups",
        "api_note_3": "- View tokens in Prompt Config",
        # Section names
        "section_A": "A · Self-Intro",
        "section_B": "B · Resume Deep-Dive",
        "section_C": "C · Case Design",
        "section_D": "D · HR Interview",
        # JD presets
        "jd_bytedance": "ByteDance — AI PM (Search)",
        "jd_tencent": "Tencent — AI PM Trainee",
        "jd_alibaba": "Alibaba — AI PM (Tongyi/DAMO)",
        "jd_meituan": "Meituan — AI PM (CatPaw Agent)",
        "jd_pdd": "PDD — AI Product (E-commerce)",
        # Style options
        "style_logic": "Logic-Driven — Focus on logic chains and evidence",
        "style_data": "Data-Driven — Probe data sources and methods",
        "style_pressure": "Pressure-Test — Guilty-until-proven approach",
        "style_behavior": "Behavior-Oriented — Focus on STAR structure",
        # Progress
        "progress_ready": "Ready to start",
        "progress_ended": "Interview ended",
        "progress_main": "Main / {total} sections",
        "progress_followup": "Follow-up / {total} sections",
    },
    "zh": {
        "app_title": "面霸",
        "app_subtitle": "AI PM校招面试模拟与复盘",
        "nav": "导航",
        "quick_start": "快速上手",
        "api_notes": "API说明",
        "page_setup": "📋 面试配置",
        "page_interview": "💬 模拟面试",
        "page_review": "📊 复盘报告",
        "page_comparison": "⚖️ A/B/C对比",
        "page_prompt": "⚙️ Prompt配置",
        "page_kb": "📁 知识库管理",
        "setup_title": "📋 面试配置",
        "setup_desc": "配置简历、目标岗位和面试官风格，准备好后点击开始。",
        "resume_title": "📄 简历摘要",
        "resume_placeholder": "示例：字节跳动AI产品部实习3个月，做智能客服方向，负责RAG优化、数据标注、效果验证...",
        "resume_help": "写清楚实习公司、角色、项目、技能即可，无需完整简历",
        "resume_upload_label": "或上传简历文件（PDF / Word / TXT）",
        "resume_upload_help": "支持 .pdf、.docx、.txt、.md；会自动提取文字填入下方文本框，仍可手动编辑",
        "jd_upload_label": "或上传JD文件（PDF / Word / TXT）",
        "jd_upload_help": "支持 .pdf、.docx、.txt、.md；会自动提取文字填入下方文本框",
        "upload_reading": "正在读取文件...",
        "upload_ok": "已从文件载入，请检查下方文本内容",
        "upload_clear": "清除已上传文件",
        "style_title": "🎭 面试官风格",
        "style_select": "选择风格（面试开始后不可更改）",
        "jd_title": "📋 目标岗位JD",
        "jd_preset": "选择预设JD",
        "jd_custom": "自定义",
        "jd_placeholder": "粘贴目标岗位的JD内容...",
        "jd_editable": "JD内容（可编辑）",
        "api_title": "🔑 API 设置",
        "api_key_label": "API Key（留空=Mock模式）",
        "api_key_help": "不填API Key会自动使用Mock模式，可体验全部流程但回答质量有限",
        "api_key_env": "✅ 已从 .env 自动加载 OpenRouter Key（可直接改或清空；清空则回落到 Mock 模式）",
        "llm_provider": "LLM厂商",
        "model": "模型",
        "stats_sections": "板块数",
        "stats_followups": "每板块追问",
        "stats_rounds": "总轮数",
        "stats_kb": "RAG知识库",
        "start_btn": "🚀 开始面试",
        "warn_no_resume": "请先粘贴简历摘要",
        "warn_no_jd": "请提供目标岗位JD",
        "spinner_first_q": "面试官正在准备第一个问题...",
        "interview_title": "💬 模拟面试",
        "warn_no_setup": "请先在「面试配置」页面完成设置",
        "go_setup": "前往面试配置",
        "metric_progress": "面试进度",
        "metric_style": "面试官风格",
        "metric_api_calls": "API调用",
        "metric_tokens": "Token",
        "qtype_main": "主问题",
        "qtype_followup": "追问",
        "label_round": "第{round}轮",
        "label_probing_layer": "穿透层",
        "label_fabrication_risk": "穿帮风险",
        "label_intent": "意图",
        "label_jd_kw": "JD关键词",
        "cot_header": "🧠 CoT推理分析",
        "cot_weaknesses": "**回答弱点:**",
        "cot_angle": "**追问角度:**",
        "interview_end_msg": "🎉 面试已结束！共完成8轮问答（AaBbCcDd）。",
        "interview_end_hint": "点击下方按钮生成详细复盘报告。",
        "btn_review": "📊 生成复盘报告",
        "btn_restart": "🔄 重新面试",
        "chat_placeholder": "输入你的回答...",
        "spinner_analyzing": "面试官分析中...",
        "rag_title": "🔍 RAG检索",
        "rag_empty": "RAG结果将显示在这里",
        "analysis_title": "💡 当前分析",
        "label_section": "板块",
        "label_type": "类型",
        "review_title": "📊 复盘报告",
        "review_generate_btn": "生成复盘报告",
        "review_spinner": "正在分析面试表现...",
        "warn_no_interview": "请先完成一场模拟面试",
        "review_total_score": "面试总分",
        "review_comment": "评语:",
        "review_advantages": "✅ 优势",
        "review_shortcomings": "❌ 短板",
        "review_style_risks": "⚠️ 风格风险",
        "review_per_q": "📝 逐题诊断",
        "review_your_answer": "**你的回答:**",
        "review_score_reason": "**评分理由:**",
        "review_star": "**STAR诊断**",
        "review_star_s": "情境",
        "review_star_t": "任务",
        "review_star_a": "行动",
        "review_star_r": "结果",
        "review_cap_gap": "**能力缺口**",
        "review_fab_risk": "**穿帮风险预警**",
        "review_calibration": "**简历话术校准**",
        "review_reconstructed": "**✅ 高分回答重构**",
        "review_improvement": "**💡 改进建议**",
        "review_growth": "🗺 成长路线",
        "review_short_term": "**短期（下次面试前）**",
        "review_long_term": "**长期（1-3个月）**",
        "review_usage": "复盘Token",
        "review_cost": "费用",
        "btn_abc": "📊 查看A/B/C对比",
        "cmp_title": "⚖️ A/B/C 三组对比评估",
        "cmp_desc": "运行同一测试用例通过三种变体来衡量增量价值。",
        "cmp_select_case": "选择测试用例",
        "cmp_run": "▶️ 运行三组对比",
        "cmp_variant_a": "A 基础LLM",
        "cmp_variant_b": "B 简化系统",
        "cmp_variant_c": "C 完整系统",
        "cmp_questions": "**问题序列:**",
        "cmp_probing": "**穿透层分布:**",
        "cmp_times": "次",
        "prompt_title": "⚙️ Prompt 配置",
        "prompt_desc": "调整Module A/B的Prompt和few-shot示例。",
        "prompt_tab_a": "Module A 追问考官",
        "prompt_tab_b": "Module B STAR复盘",
        "prompt_tab_fs": "Few-shot",
        "prompt_save": "💾 保存",
        "prompt_saved": "已保存！",
        "prompt_fs_type": "类型",
        "prompt_fs_answer": "用户回答",
        "prompt_fs_ideal": "理想追问",
        "prompt_fs_layer": "穿透层",
        "prompt_fs_risk": "穿帮风险",
        "prompt_save_fs": "💾 保存Few-shot",
        "prompt_usage_title": "📊 Token消耗",
        "prompt_total_calls": "总API调用",
        "prompt_total_tokens": "总Token",
        "prompt_avg_tokens": "平均Token/次",
        "prompt_cost": "费用($)",
        "prompt_reset": "🔄 重置",
        "kb_title": "📁 知识库管理",
        "kb_entries": "知识库条目",
        "kb_filter": "按类别筛选",
        "kb_all": "全部",
        "kb_search_title": "🔍 检索测试",
        "kb_search_placeholder": "如：实习期间负责了AI客服产品...",
        "kb_search_btn": "检索",
        "kb_reindex": "🔄 重新索引",
        "kb_reindexing": "重建索引...",
        "kb_reindex_done": "已完成！",
        "kb_relevance": "相关度",
        "kb_no_title": "无标题",
        "qs_1": "1. **面试配置**: 粘贴简历+选JD+选风格",
        "qs_2": "2. **模拟面试**: 8轮问答(AaBbCcDd)",
        "qs_3": "3. **复盘报告**: STAR诊断+穿帮预警",
        "qs_4": "4. **A/B/C对比**: 三组变体对比",
        "qs_5": "5. **Prompt配置**: 调整Prompt",
        "qs_6": "6. **知识库管理**: 查看/测试RAG",
        "api_note_1": "- 无API Key=Mock模式(模拟回答)",
        "api_note_2": "- 接入真实API后追问质量提升",
        "api_note_3": "- Token在「Prompt配置」查看",
        "section_A": "A·自我介绍",
        "section_B": "B·简历深挖",
        "section_C": "C·案例设计",
        "section_D": "D·HR综合",
        "jd_bytedance": "字节跳动-AI产品经理（搜索方向）",
        "jd_tencent": "腾讯-AI产品经理培训生",
        "jd_alibaba": "阿里巴巴-AI产品经理（通义/达摩院）",
        "jd_meituan": "美团-AI产品经理（CatPaw智能体）",
        "jd_pdd": "拼多多-AI产品方向（C端电商）",
        "style_logic": "逻辑严谨 — 注重逻辑链条和证据",
        "style_data": "数据驱动 — 追问数据来源和实验方法",
        "style_pressure": "压力测试 — 有罪推定式，默认你有问题",
        "style_behavior": "行为导向 — 关注STAR结构和个人行为",
        "progress_ready": "准备开始",
        "progress_ended": "面试结束",
        "progress_main": "{label} 主问题 / 共{total}板块",
        "progress_followup": "{label} 追问 / 共{total}板块",
    },
}

# JD text content per language
JD_TEXTS = {
    "en": {
        "ByteDance — AI PM (Search)": "Participate in Douyin e-commerce search AI product design, advance AI smart shopping guide scenarios. Understanding of LLM/Generative AI, Prompt/Retrieval-Augmented/Agent awareness preferred.",
        "Tencent — AI PM Trainee": "Drive AI product full lifecycle as project lead. Evaluate end-to-end loop, architecture judgment, AI taste, frontier sensitivity. Coding ability is a plus.",
        "Alibaba — AI PM (Tongyi/DAMO)": "AI application design, workflow orchestration and prompt design, translate LLM/Multimodal/Agent/RAG into product features. SQL/Python preferred.",
        "Meituan — AI PM (CatPaw Agent)": "Work with AI core team on projects, write demos and run experiments. AI coding ability + industrial practice + deep user.",
        "PDD — AI Product (E-commerce)": "C-end e-commerce AI application scenario design. Search/recommendation/smart customer service/marketing guide. Self-driven + logical expression + user thinking.",
    },
    "zh": {
        "字节跳动-AI产品经理（搜索方向）": "参与抖音电商搜索AI产品方案设计，推进AI智能导购场景。了解大模型/生成式AI，Prompt/检索增强/Agent认知优先。",
        "腾讯-AI产品经理培训生": "以项目负责人身份推进AI产品完整流程。考察端到端闭环、架构判断、AI品味、前沿敏感。代码能力加分。",
        "阿里巴巴-AI产品经理（通义/达摩院）": "AI应用方案设计，Workflow编排及Prompt设计，将LLM/多模态/Agent/RAG转化为产品功能。SQL/Python优先。",
        "美团-AI产品经理（CatPaw智能体）": "与AI核心团队参与项目，自己写Demo做实验。AI编程能力+工业实践+深度用户。",
        "拼多多-AI产品方向（C端电商）": "C端电商AI应用场景设计。搜索推荐/智能客服/营销导购。自驱力+逻辑表达+用户思维。",
    },
}


def t(key, **kwargs):
    lang = st.session_state.get("lang", "en")
    text = I18N.get(lang, I18N["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def get_lang():
    return st.session_state.get("lang", "en")


# ========== Page Config ==========
st.set_page_config(page_title="MianBa — AI PM Interview Simulator", page_icon="🎯", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {max-width: 1200px; padding-top: 1.5rem;}
.chat-message {padding: 0.5rem 0;}
.score-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.2rem;
}
.score-pass {background: #d1fae5; color: #065f46;}
.score-warn {background: #fef3c7; color: #92400e;}
.score-fail {background: #fee2e2; color: #991b1b;}
.section-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 4px;
}
.section-A {background: #dbeafe; color: #1e40af;}
.section-B {background: #ede9fe; color: #5b21b6;}
.section-C {background: #fce7f3; color: #9d174d;}
.section-D {background: #dcfce7; color: #166534;}
.risk-low {color: #059669; font-weight: 600;}
.risk-medium {color: #d97706; font-weight: 600;}
.risk-high {color: #dc2626; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


def init_state():
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "session" not in st.session_state:
        st.session_state.session = None
    if "review_data" not in st.session_state:
        st.session_state.review_data = None
    if "test_results" not in st.session_state:
        st.session_state.test_results = {}
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "prompts_config" not in st.session_state:
        st.session_state.prompts_config = {
            "module_a": MODULE_A_SYSTEM_PROMPT,
            "module_b": MODULE_B_SYSTEM_PROMPT,
            "few_shots": copy.deepcopy(FEW_SHOT_EXAMPLES),
        }
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False

    # 首次进入时从 .env / 环境变量预填 OpenRouter Key（只做一次，用户清空后不再回填）
    if not st.session_state.get("_env_prefilled"):
        st.session_state._env_prefilled = True
        env_key = load_env_api_key()
        if env_key:
            st.session_state.api_key = env_key
            default_provider, default_model = "openrouter", "openai/gpt-4o-mini"
            st.session_state.provider = default_provider
            st.session_state.model = default_model
            st.session_state.provider_label = PROVIDER_PRESETS[default_provider]["name"]
            lbl = provider_model_label(default_provider, default_model)
            if lbl:
                st.session_state.model_label = lbl
            st.session_state.env_key_loaded = True


def get_llm_client():
    if st.session_state.llm_client is not None:
        return st.session_state.llm_client
    api_key = st.session_state.get("api_key", "")
    model = st.session_state.get("model", "gpt-4o")
    provider = st.session_state.get("provider", "openai")
    base_url = st.session_state.get("base_url", "")
    if api_key:
        st.session_state.llm_client = LLMClient(
            api_key=api_key, model=model, base_url=base_url, provider=provider
        )
    else:
        st.session_state.llm_client = MockLLMClient()
    return st.session_state.llm_client


@st.cache_resource
def get_retriever():
    return HybridRetriever()


def section_badge(label):
    color_map = {"A": "section-A", "B": "section-B", "C": "section-C", "D": "section-D"}
    name_keys = {"A": "section_A", "B": "section_B", "C": "section_C", "D": "section_D"}
    css_class = color_map.get(label, "section-A")
    name = t(name_keys.get(label, "section_A"))
    return f'<span class="{css_class}">{name}</span>'


def risk_html(risk):
    css_class = f"risk-{risk}" if risk in ["low", "medium", "high"] else "risk-low"
    return f'<span class="{css_class}">{risk}</span>'


def render_doc_uploader(label, help_text, text_key, uploader_key):
    """渲染文件上传控件，解析 PDF/Word/TXT 后写入 text_key 对应的文本框。

    仅在每次上传「新文件」时解析一次（用 文件名+大小 去重），
    这样用户手动编辑后的内容不会被后续 rerun 覆盖。
    """
    uploaded = st.file_uploader(
        label,
        type=UPLOAD_EXTENSIONS,
        key=uploader_key,
        help=help_text,
    )
    if uploaded is None:
        return

    file_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get(f"_parsed_{uploader_key}") == file_id:
        st.caption(f"📎 {uploaded.name}")
        return

    with st.spinner(t("upload_reading")):
        text, err = extract_text_from_upload(uploaded)

    if err:
        st.error(f"❌ {err}")
        return

    st.session_state[text_key] = text
    st.session_state[f"_parsed_{uploader_key}"] = file_id
    st.success(f"✅ {t('upload_ok')}（{uploaded.name} · {len(text)} 字）")


# ========== Page: Interview Setup ==========
def page_setup():
    st.markdown(f"## {t('setup_title')}")
    st.markdown(t("setup_desc"))

    col_resume, col_jd = st.columns(2)

    with col_resume:
        st.markdown(f"### {t('resume_title')}")
        render_doc_uploader(
            t("resume_upload_label"),
            t("resume_upload_help"),
            text_key="resume_input",
            uploader_key="resume_file",
        )
        resume = st.text_area(
            t("resume_title"),
            height=200,
            key="resume_input",
            help=t("resume_help"),
            placeholder=t("resume_placeholder"),
        )

        st.markdown(f"### {t('style_title')}")
        style = st.selectbox(t("style_select"), [
            t("style_logic"),
            t("style_data"),
            t("style_pressure"),
            t("style_behavior"),
        ])

    with col_jd:
        st.markdown(f"### {t('jd_title')}")
        jd_preset_options = [
            t("jd_custom"),
            t("jd_bytedance"),
            t("jd_tencent"),
            t("jd_alibaba"),
            t("jd_meituan"),
            t("jd_pdd"),
        ]
        jd_preset = st.selectbox(t("jd_preset"), jd_preset_options)

        if jd_preset == t("jd_custom"):
            render_doc_uploader(
                t("jd_upload_label"),
                t("jd_upload_help"),
                text_key="jd_input",
                uploader_key="jd_file",
            )
            jd_text = st.text_area(t("jd_title"), height=160, key="jd_input",
                                   placeholder=t("jd_placeholder"))
        else:
            jd_texts = JD_TEXTS.get(get_lang(), JD_TEXTS["en"])
            jd_text = jd_texts.get(jd_preset, "")
            jd_text = st.text_area(t("jd_editable"), value=jd_text, height=160, key="jd_display")

        st.markdown(f"### {t('api_title')}")
        api_key = st.text_input(t("api_key_label"), type="password", key="api_key",
                                help=t("api_key_help"))
        if st.session_state.get("env_key_loaded"):
            st.caption(t("api_key_env"))

        provider_options = list(PROVIDER_PRESETS.keys())
        provider_labels = [f"{PROVIDER_PRESETS[k]['name']}" for k in provider_options]
        provider_label = st.selectbox(t("llm_provider"), provider_labels, key="provider_label")
        provider_key = provider_options[provider_labels.index(provider_label)]
        st.session_state.provider = provider_key

        provider_info = PROVIDER_PRESETS[provider_key]
        model_options = provider_info["models"]
        model_labels = [f"{m['name']} (${m['prompt_price']}/{m['completion_price']})" for m in model_options]
        model_label = st.selectbox(t("model"), model_labels, key="model_label")
        selected_model = model_options[model_labels.index(model_label)]
        st.session_state.model = selected_model["id"]

        if provider_key == "custom":
            base_url = st.text_input("API Base URL", placeholder="https://...", key="base_url")
        else:
            st.session_state.base_url = provider_info["base_url"]
            st.caption(f"API: {provider_info['base_url']}")

    total_rounds = len(SECTIONS) * (1 + FOLLOWUPS_PER_SECTION)
    st.markdown("---")
    cols_info = st.columns(4)
    with cols_info[0]:
        st.metric(t("stats_sections"), len(SECTIONS))
    with cols_info[1]:
        st.metric(t("stats_followups"), f"{FOLLOWUPS_PER_SECTION}")
    with cols_info[2]:
        st.metric(t("stats_rounds"), f"{total_rounds}")
    with cols_info[3]:
        st.metric(t("stats_kb"), "33")

    st.markdown("")

    if st.button(t("start_btn"), type="primary", use_container_width=True):
        if not resume:
            st.warning(t("warn_no_resume"))
            return
        if not jd_text:
            st.warning(t("warn_no_jd"))
            return

        st.session_state.llm_client = None
        llm = get_llm_client()
        retriever = get_retriever()

        st.session_state.session = InterviewSession(
            llm_client=llm,
            retriever=retriever,
            resume_context=resume,
            jd_context=jd_text,
            jd_tags=[],
            interviewer_style=style,
            lang=get_lang(),
        )
        st.session_state.interview_started = True

        with st.spinner(t("spinner_first_q")):
            result = st.session_state.session.start()

        st.session_state.page = t("page_interview")
        st.rerun()


# ========== Page: Mock Interview ==========
def page_interview():
    st.markdown(f"## {t('interview_title')}")

    if st.session_state.session is None:
        st.warning(t("warn_no_setup"))
        if st.button(t("go_setup")):
            st.session_state.page = t("page_setup")
            st.rerun()
        return

    session = st.session_state.session
    log = session.get_log()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("metric_progress"), session.progress_text)
    col2.metric(t("metric_style"), session.interviewer_style[:6])
    col3.metric(t("metric_api_calls"), session.llm.get_usage_summary().get("call_count", 0))
    usage = session.llm.get_usage_summary()
    col4.metric(t("metric_tokens"), f"{usage.get('total_tokens', 0)} (${usage.get('estimated_cost', 0):.4f})")

    st.markdown("---")

    main_col, side_col = st.columns([3, 1])

    with main_col:
        for entry in log:
            section = entry.get("section", "A")
            qtype = entry.get("question_type", "main")
            qtype_label = t("qtype_main") if qtype == "main" else t("qtype_followup")
            risk = entry.get("fabrication_risk", "low")
            risk_class = f"risk-{risk}" if risk in ["low", "medium", "high"] else "risk-low"

            with st.chat_message("assistant"):
                st.markdown(
                    f'{section_badge(section)} **{t("label_round", round=entry["round"])} · {qtype_label}** &nbsp;|&nbsp; '
                    f'{t("label_probing_layer")}: {entry.get("probing_layer", "N/A")} &nbsp;|&nbsp; '
                    f'{t("label_fabrication_risk")}: <span class="{risk_class}">{risk}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(entry["question"])

                intent = entry.get("interviewer_intent", "")
                if intent:
                    st.caption(f"🎯 {t('label_intent')}: {intent}")
                jd_kw = entry.get("jd_keywords", [])
                if jd_kw:
                    st.caption(f"🏷 {t('label_jd_kw')}: {', '.join(jd_kw)}")

                if qtype == "followup":
                    cot = entry.get("cot_analysis", {})
                    if cot and isinstance(cot, dict):
                        weaknesses = cot.get("weaknesses", [])
                        probing_angle = cot.get("probing_angle", "")
                        if weaknesses or probing_angle:
                            with st.expander(t("cot_header"), expanded=False):
                                if weaknesses:
                                    st.markdown(t("cot_weaknesses"))
                                    for w in weaknesses:
                                        st.markdown(f"- {w}")
                                if probing_angle:
                                    st.markdown(f"{t('cot_angle')} {probing_angle}")

            if entry.get("user_answer"):
                with st.chat_message("user"):
                    st.markdown(entry["user_answer"])

        if session.is_ended:
            st.success(t("interview_end_msg"))
            st.markdown(t("interview_end_hint"))
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(t("btn_review"), type="primary", use_container_width=True):
                    with st.spinner(t("review_spinner")):
                        review = session.generate_review()
                        st.session_state.review_data = review
                    st.session_state.page = t("page_review")
                    st.rerun()
            with col_btn2:
                if st.button(t("btn_restart"), use_container_width=True):
                    st.session_state.session = None
                    st.session_state.review_data = None
                    st.session_state.interview_started = False
                    st.session_state.page = t("page_setup")
                    st.rerun()
        else:
            user_input = st.chat_input(t("chat_placeholder"))
            if user_input:
                with st.spinner(t("spinner_analyzing")):
                    result = session.answer_and_next(user_input)
                st.rerun()

    with side_col:
        st.markdown(f"### {t('rag_title')}")

        current_entry = log[-1] if log else None
        if current_entry and current_entry.get("rag_results"):
            for r in current_entry["rag_results"]:
                retriever = get_retriever()
                item = retriever.get_by_id(r["id"])
                if item:
                    with st.container(border=True):
                        title = item.get("question") or item.get("title") or t("kb_no_title")
                        st.markdown(f"**{item['id']}** ({item['category']})")
                        st.markdown(f"{t('kb_relevance')}: {r['score']:.2f}")
                        st.caption(title[:50])
        else:
            st.info(t("rag_empty"))

        st.markdown("---")
        st.markdown(f"### {t('analysis_title')}")
        if current_entry:
            st.markdown(f"**{t('label_section')}:** {current_entry.get('section', 'N/A')}")
            st.markdown(f"**{t('label_type')}:** {t('qtype_main') if current_entry.get('question_type') == 'main' else t('qtype_followup')}")
            st.markdown(f"**{t('label_probing_layer')}:** {current_entry.get('probing_layer', 'N/A')}")
            st.markdown(f"**{t('label_intent')}:** {current_entry.get('interviewer_intent', 'N/A')}")
            risk = current_entry.get("fabrication_risk", "low")
            risk_class = f"risk-{risk}" if risk in ["low", "medium", "high"] else "risk-low"
            st.markdown(f"**{t('label_fabrication_risk')}:** <span class='{risk_class}'>{risk}</span>", unsafe_allow_html=True)


# ========== Page: Review Report ==========
def page_review():
    st.markdown(f"## {t('review_title')}")

    if not st.session_state.review_data:
        if st.session_state.session and st.session_state.session.is_ended:
            if st.button(t("review_generate_btn"), type="primary"):
                with st.spinner(t("review_spinner")):
                    st.session_state.review_data = st.session_state.session.generate_review()
                st.rerun()
        else:
            st.warning(t("warn_no_interview"))
            if st.button(t("go_setup")):
                st.session_state.page = t("page_setup")
                st.rerun()
        return

    review = st.session_state.review_data.get("review", {})
    if not review:
        st.error("Review generation failed")
        return

    overall = review.get("overall", {})
    total_score = overall.get("total_score", 0)
    score_class = "score-pass" if total_score >= 70 else ("score-warn" if total_score >= 50 else "score-fail")

    col_score, col_info = st.columns([1, 3])
    with col_score:
        st.markdown(f'<div class="{score_class}">{total_score}/100</div>', unsafe_allow_html=True)
        st.caption(t("review_total_score"))
    with col_info:
        st.markdown(f"{t('review_comment')} {overall.get('comment', 'N/A')}")

    st.markdown("---")

    profile = overall.get("capability_profile", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {t('review_advantages')}")
        for adv in profile.get("advantages", []):
            st.markdown(f"- {adv}")
    with col2:
        st.markdown(f"### {t('review_shortcomings')}")
        for short in profile.get("shortcomings", []):
            st.markdown(f"- {short}")
    with col3:
        st.markdown(f"### {t('review_style_risks')}")
        for risk in profile.get("style_risks", []):
            st.markdown(f"- {risk}")

    st.markdown("---")

    per_q = review.get("per_question", [])
    st.markdown(f"### {t('review_per_q')}")

    for i, q in enumerate(per_q):
        score = q.get("score", 0)
        score_class = "score-pass" if score >= 70 else ("score-warn" if score >= 50 else "score-fail")
        question_text = q.get("question", "")[:50]

        with st.expander(f"Q{i+1} | {t('review_total_score')}: {score} | {question_text}..."):
            st.markdown(f"{t('review_your_answer')} {q.get('answer', 'N/A')[:100]}...")
            st.markdown(f"{t('review_score_reason')} {q.get('score_reason', 'N/A')}")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(t("review_star"))
                star = q.get("star_diagnosis", {})
                star_labels = {"S": t("review_star_s"), "T": t("review_star_t"), "A": t("review_star_a"), "R": t("review_star_r")}
                for k in ["S", "T", "A", "R"]:
                    v = star.get(k, "N/A")
                    st.markdown(f"- {star_labels[k]} ({k}): {v}")

                st.markdown(t("review_cap_gap"))
                for gap in q.get("capability_gap", []):
                    st.markdown(f"- {gap}")

            with col_b:
                st.markdown(t("review_fab_risk"))
                for risk in q.get("fabrication_risk_warning", []):
                    st.markdown(f"- {risk}")

                st.markdown(t("review_calibration"))
                for cal in q.get("calibration", []):
                    st.markdown(f"- {cal}")

            st.markdown("---")
            st.markdown(t("review_reconstructed"))
            st.info(q.get("reconstructed_answer", "N/A"))

            st.markdown(t("review_improvement"))
            st.success(q.get("improvement_suggestion", "N/A"))

    st.markdown("---")

    st.markdown(f"### {t('review_growth')}")
    growth = overall.get("growth_plan", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(t("review_short_term"))
        st.write(growth.get("short_term", "N/A"))
    with col2:
        st.markdown(t("review_long_term"))
        st.write(growth.get("long_term", "N/A"))

    usage = st.session_state.review_data.get("usage", {})
    if usage:
        st.markdown("---")
        st.caption(f"{t('review_usage')}: {usage.get('total_tokens', 0)} | {t('review_cost')}: ${usage.get('estimated_cost_usd', 0)}")

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("btn_restart"), use_container_width=True):
            st.session_state.session = None
            st.session_state.review_data = None
            st.session_state.interview_started = False
            st.session_state.page = t("page_setup")
            st.rerun()
    with col2:
        if st.button(t("btn_abc"), use_container_width=True):
            st.session_state.page = t("page_comparison")
            st.rerun()


# ========== Page: A/B/C Comparison ==========
def page_comparison():
    st.markdown(f"## {t('cmp_title')}")

    st.markdown(f"""
    | Variant | RAG | Prompt | Few-shot | Description |
    |---------|-----|--------|----------|-------------|
    | **A — Base LLM** | ❌ | ❌ | ❌ | Base system prompt only |
    | **B — Simplified** | ❌ | ✅ | ✅ | Module A + few-shot |
    | **C — Full System** | ✅ | ✅ | ✅ | RAG + Module A + B |
    """)

    case_options = [f"{c['id']} - {c['category']}" for c in TEST_CASES]
    selected = st.selectbox(t("cmp_select_case"), case_options)
    case_idx = case_options.index(selected)
    test_case = TEST_CASES[case_idx]

    if st.button(t("cmp_run"), type="primary"):
        llm = get_llm_client()
        retriever = get_retriever()
        results = {}
        progress = st.progress(0, f"Running variant A...")

        for label, use_rag, use_prompt in [("A", False, False), ("B", False, True), ("C", True, True)]:
            with st.spinner(f"Running variant {label}..."):
                client = MockLLMClient() if isinstance(llm, MockLLMClient) else LLMClient(
                    api_key=st.session_state.get("api_key", ""),
                    model=st.session_state.get("model", "gpt-4o")
                )
                sess = InterviewSession(
                    llm_client=client,
                    retriever=retriever,
                    resume_context=test_case.get("resume", ""),
                    jd_context=test_case.get("jd", ""),
                    jd_tags=[],
                    interviewer_style=test_case.get("style", t("style_logic")),
                    lang=get_lang(),
                        )
                sess.start()
                for ans in test_case.get("answers", []):
                    r = sess.answer_and_next(ans)
                    if r.get("ended"):
                        break
                results[label] = {
                    "questions": [e["question"] for e in sess.get_log()],
                    "probing_layers": [e.get("probing_layer", "") for e in sess.get_log()],
                    "usage": sess.llm.get_usage_summary(),
                }
                progress.progress((["A", "B", "C"].index(label) + 1) * 33, f"Variant {label} done")

        st.session_state.test_results[case_idx] = results
        st.rerun()

    results = st.session_state.test_results.get(case_idx)
    if results:
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        for label, col, key in [(t("cmp_variant_a"), col_a, "A"), (t("cmp_variant_b"), col_b, "B"), (t("cmp_variant_c"), col_c, "C")]:
            with col:
                st.markdown(f"### {label}")
                data = results.get(key, {})
                st.markdown(t("cmp_questions"))
                for i, q in enumerate(data.get("questions", [])):
                    st.markdown(f"{i+1}. {q[:60]}...")
                st.markdown(t("cmp_probing"))
                layers = data.get("probing_layers", [])
                for layer in set(layers):
                    st.markdown(f"- {layer}: {layers.count(layer)} {t('cmp_times')}")
                usage = data.get("usage", {})
                st.metric(t("metric_tokens"), usage.get("total_tokens", 0))
                st.metric(t("metric_api_calls"), usage.get("call_count", 0))


# ========== Page: Prompt Config ==========
def page_prompt_editor():
    st.markdown(f"## {t('prompt_title')}")
    st.markdown(t("prompt_desc"))

    config = st.session_state.prompts_config
    tab_a, tab_b, tab_fs = st.tabs([t("prompt_tab_a"), t("prompt_tab_b"), t("prompt_tab_fs")])

    with tab_a:
        new_a = st.text_area("Module A System Prompt", value=config["module_a"], height=400, key="prompt_a")
        if st.button(t("prompt_save"), key="save_a"):
            config["module_a"] = new_a
            import prompts.module_a as ma
            ma.MODULE_A_SYSTEM_PROMPT = new_a
            st.success(t("prompt_saved"))

    with tab_b:
        new_b = st.text_area("Module B System Prompt", value=config["module_b"], height=400, key="prompt_b")
        if st.button(t("prompt_save"), key="save_b"):
            config["module_b"] = new_b
            import prompts.module_b as mb
            mb.MODULE_B_SYSTEM_PROMPT = new_b
            st.success(t("prompt_saved"))

    with tab_fs:
        few_shots = config["few_shots"]
        for i, ex in enumerate(few_shots):
            with st.expander(f"{t('prompt_fs_type')} {i+1}: {ex['type']}"):
                ex["type"] = st.text_input(t("prompt_fs_type"), value=ex["type"], key=f"fs_type_{i}")
                ex["user_answer"] = st.text_area(t("prompt_fs_answer"), value=ex["user_answer"], height=80, key=f"fs_user_{i}")
                ex["ideal_follow_up"] = st.text_area(t("prompt_fs_ideal"), value=ex["ideal_follow_up"], height=80, key=f"fs_ideal_{i}")
                ex["probing_layer"] = st.selectbox(t("prompt_fs_layer"), ["layer1", "layer2", "layer3"],
                    index=["layer1","layer2","layer3"].index(ex.get("probing_layer","layer1")), key=f"fs_layer_{i}")
                ex["fabrication_risk"] = st.selectbox(t("prompt_fs_risk"), ["low", "medium", "high"],
                    index=["low","medium","high"].index(ex.get("fabrication_risk","low")), key=f"fs_risk_{i}")
        if st.button(t("prompt_save_fs"), key="save_fs"):
            import prompts.module_a as ma
            ma.FEW_SHOT_EXAMPLES = copy.deepcopy(few_shots)
            st.success(t("prompt_saved"))

    st.markdown("---")
    st.markdown(f"### {t('prompt_usage_title')}")
    llm = get_llm_client()
    usage = llm.get_usage_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("prompt_total_calls"), usage.get("call_count", 0))
    c2.metric(t("prompt_total_tokens"), usage.get("total_tokens", 0))
    c3.metric(t("prompt_avg_tokens"), usage.get("avg_tokens_per_call", 0))
    c4.metric(t("prompt_cost"), f"${usage.get('estimated_cost', 0):.4f}")
    if st.button(t("prompt_reset")):
        llm.reset_usage()
        st.rerun()


# ========== Page: Knowledge Base ==========
def page_kb_manager():
    st.markdown(f"## {t('kb_title')}")

    retriever = get_retriever()
    kb = retriever.get_all()
    st.metric(t("kb_entries"), len(kb))

    categories = [t("kb_all")] + list(set(item["category"] for item in kb))
    selected_cat = st.selectbox(t("kb_filter"), categories)

    filtered = [item for item in kb if item["category"] == selected_cat] if selected_cat != t("kb_all") else kb

    for item in filtered:
        title = item.get("question") or item.get("title") or t("kb_no_title")
        with st.expander(f"[{item['id']}] {title[:60]} ({item['category']})"):
            st.json(item)

    st.markdown("---")
    st.markdown(f"### {t('kb_search_title')}")
    query = st.text_input(t("kb_search_title"), placeholder=t("kb_search_placeholder"))
    if query and st.button(t("kb_search_btn")):
        results = retriever.retrieve(query, top_k=5)
        for r in results:
            item = r["item"]
            title = item.get("question") or item.get("title") or t("kb_no_title")
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{item['id']}** — {title[:50]}")
                c1.caption(f"Category: {item['category']}")
                c2.metric(t("kb_relevance"), f"{r['score']:.3f}")

    st.markdown("---")
    if st.button(t("kb_reindex")):
        with st.spinner(t("kb_reindexing")):
            retriever.reload()
        st.success(t("kb_reindex_done"))


# ========== Main Navigation ==========
def main():
    init_state()

    # Language selector in sidebar
    st.sidebar.markdown("---")
    lang_col1, lang_col2 = st.sidebar.columns(2)
    with lang_col1:
        if st.button("🇬🇧 EN", use_container_width=True,
                      type="primary" if get_lang() == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with lang_col2:
        if st.button("🇨🇳 中文", use_container_width=True,
                      type="primary" if get_lang() == "zh" else "secondary"):
            st.session_state.lang = "zh"
            st.rerun()

    st.sidebar.markdown(f"## 🎯 {t('app_title')}")
    st.sidebar.caption(t("app_subtitle"))

    pages = {
        t("page_setup"): page_setup,
        t("page_interview"): page_interview,
        t("page_review"): page_review,
        t("page_comparison"): page_comparison,
        t("page_prompt"): page_prompt_editor,
        t("page_kb"): page_kb_manager,
    }

    if "page" not in st.session_state:
        st.session_state.page = t("page_setup")

    # If interview started, jump to interview page
    if st.session_state.interview_started and st.session_state.page == t("page_setup"):
        st.session_state.page = t("page_interview")

    page_keys = list(pages.keys())
    current_idx = page_keys.index(st.session_state.page) if st.session_state.page in page_keys else 0
    selected = st.sidebar.radio(t("nav"), page_keys, index=current_idx)
    st.session_state.page = selected

    st.sidebar.divider()
    with st.sidebar.expander(t("quick_start")):
        st.markdown(t("qs_1"))
        st.markdown(t("qs_2"))
        st.markdown(t("qs_3"))
        st.markdown(t("qs_4"))
        st.markdown(t("qs_5"))
        st.markdown(t("qs_6"))

    with st.sidebar.expander(t("api_notes")):
        st.markdown(t("api_note_1"))
        st.markdown(t("api_note_2"))
        st.markdown(t("api_note_3"))

    pages[selected]()


if __name__ == "__main__":
    main()
