"""LLM 客户端封装：支持任意 OpenAI 兼容 API（OpenAI/DeepSeek/智谱/月之暗面/Anthropic等），含 token 追踪"""

import json
import os
import time
from typing import Dict, List, Optional


# 厂商预设配置：base_url + 推荐模型 + 定价（美元/1K tokens）
PROVIDER_PRESETS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o", "prompt_price": 0.0025, "completion_price": 0.010},
            {"id": "gpt-4o-mini", "name": "GPT-4o mini", "prompt_price": 0.00015, "completion_price": 0.0006},
            {"id": "gpt-4.1", "name": "GPT-4.1", "prompt_price": 0.002, "completion_price": 0.008},
            {"id": "gpt-4.1-mini", "name": "GPT-4.1 mini", "prompt_price": 0.0004, "completion_price": 0.0016},
            {"id": "gpt-4.1-nano", "name": "GPT-4.1 nano", "prompt_price": 0.0001, "completion_price": 0.0004},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "prompt_price": 0.010, "completion_price": 0.030},
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": [
            {"id": "deepseek-chat", "name": "DeepSeek-V3", "prompt_price": 0.00014, "completion_price": 0.00028},
            {"id": "deepseek-reasoner", "name": "DeepSeek-R1", "prompt_price": 0.00055, "completion_price": 0.00219},
        ],
    },
    "zhipu": {
        "name": "智谱AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": [
            {"id": "glm-4-plus", "name": "GLM-4-Plus", "prompt_price": 0.00139, "completion_price": 0.00139},
            {"id": "glm-4-air", "name": "GLM-4-Air", "prompt_price": 0.000007, "completion_price": 0.000014},
            {"id": "glm-4-airx", "name": "GLM-4-AirX", "prompt_price": 0.000069, "completion_price": 0.000139},
            {"id": "glm-4-flash", "name": "GLM-4-Flash", "prompt_price": 0.00000014, "completion_price": 0.00000028},
        ],
    },
    "moonshot": {
        "name": "月之暗面 (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "models": [
            {"id": "moonshot-v1-8k", "name": "Moonshot-V1-8K", "prompt_price": 0.0017, "completion_price": 0.0034},
            {"id": "moonshot-v1-32k", "name": "Moonshot-V1-32K", "prompt_price": 0.0034, "completion_price": 0.0068},
            {"id": "moonshot-v1-128k", "name": "Moonshot-V1-128K", "prompt_price": 0.0068, "completion_price": 0.0136},
        ],
    },
    "qwen": {
        "name": "通义千问 (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            {"id": "qwen-plus", "name": "Qwen-Plus", "prompt_price": 0.00042, "completion_price": 0.0014},
            {"id": "qwen-turbo", "name": "Qwen-Turbo", "prompt_price": 0.000042, "completion_price": 0.00014},
            {"id": "qwen-max", "name": "Qwen-Max", "prompt_price": 0.0028, "completion_price": 0.0084},
        ],
    },
    "doubao": {
        "name": "豆包 (Doubao)",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "models": [
            {"id": "doubao-pro-4k", "name": "Doubao-Pro-4K", "prompt_price": 0.00035, "completion_price": 0.0007},
            {"id": "doubao-lite-4k", "name": "Doubao-Lite-4K", "prompt_price": 0.000035, "completion_price": 0.00007},
        ],
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com/v1",
        "models": [
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "prompt_price": 0.003, "completion_price": 0.015},
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "prompt_price": 0.015, "completion_price": 0.075},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "prompt_price": 0.00025, "completion_price": 0.00125},
        ],
    },
    # OpenRouter 统一网关（价格取 OpenRouter 公开报价，单位 $/1K token）
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini", "prompt_price": 0.00015, "completion_price": 0.0006},
            {"id": "deepseek/deepseek-v3.2", "name": "DeepSeek V3.2", "prompt_price": 0.00028, "completion_price": 0.00042},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "prompt_price": 0.0025, "completion_price": 0.01},
            {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "prompt_price": 0.0001, "completion_price": 0.0004},
            {"id": "qwen/qwen-2.5-72b-instruct", "name": "Qwen2.5 72B", "prompt_price": 0.00035, "completion_price": 0.0004},
            {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "prompt_price": 0.003, "completion_price": 0.015},
        ],
    },
    "custom": {
        "name": "自定义 (OpenAI兼容)",
        "base_url": "",
        "models": [
            {"id": "custom-model", "name": "自定义模型", "prompt_price": 0.001, "completion_price": 0.003},
        ],
    },
}


def get_model_pricing(model: str) -> Dict:
    """根据模型名称查找定价，找不到返回默认值"""
    for provider in PROVIDER_PRESETS.values():
        for m in provider["models"]:
            if m["id"].lower() == model.lower():
                return m
    return {"prompt_price": 0.001, "completion_price": 0.003}


def load_env_api_key(var: str = "OPENROUTER_API_KEY") -> str:
    """读取 API Key：优先环境变量，其次项目根目录的 .env。

    只读取、不做任何打印或外发；.env 不应提交到仓库。
    """
    key = (os.getenv(var) or "").strip()
    if key:
        return key
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == var:
                    return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def provider_model_label(provider: str, model_id: str) -> str:
    """把 provider + model_id 还原成界面上那个 "名称 ($p/$c)" 的下拉项文案。"""
    info = PROVIDER_PRESETS.get(provider, {})
    for m in info.get("models", []):
        if m["id"] == model_id:
            return f"{m['name']} (${m['prompt_price']}/{m['completion_price']})"
    return ""


class LLMClient:
    def __init__(self, api_key: str = "", model: str = "gpt-4o", base_url: str = "", provider: str = "openai"):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        if base_url:
            self.base_url = base_url
        elif provider in PROVIDER_PRESETS and PROVIDER_PRESETS[provider]["base_url"]:
            self.base_url = PROVIDER_PRESETS[provider]["base_url"]
        else:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.token_usage = {"total_prompt": 0, "total_completion": 0, "total_tokens": 0, "call_count": 0, "estimated_cost": 0.0}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("请安装 openai 库: pip install openai")
        return self._client

    def chat(self, messages: List[Dict], temperature: float = 0.7, response_format: Optional[Dict] = None) -> Dict:
        client = self._get_client()

        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format

        start_time = time.time()
        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start_time

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        self.token_usage["total_prompt"] += prompt_tokens
        self.token_usage["total_completion"] += completion_tokens
        self.token_usage["total_tokens"] += total_tokens
        self.token_usage["call_count"] += 1
        self.token_usage["estimated_cost"] += cost

        content = response.choices[0].message.content

        return {
            "content": content,
            "raw_response": response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "estimated_cost_usd": round(cost, 4),
            },
            "model": self.model,
        }

    def chat_json(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        result = self.chat(messages, temperature, response_format={"type": "json_object"})
        try:
            result["parsed"] = json.loads(result["content"])
        except json.JSONDecodeError:
            result["parsed"] = None
        return result

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """根据模型定价估算费用（美元/1K tokens）"""
        pricing = get_model_pricing(self.model)
        prompt_price = pricing.get("prompt_price", 0.001)
        completion_price = pricing.get("completion_price", 0.003)
        return (prompt_tokens / 1000 * prompt_price) + (completion_tokens / 1000 * completion_price)

    def get_usage_summary(self) -> Dict:
        return {
            **self.token_usage,
            "avg_tokens_per_call": round(self.token_usage["total_tokens"] / max(self.token_usage["call_count"], 1), 1),
        }

    def reset_usage(self):
        self.token_usage = {"total_prompt": 0, "total_completion": 0, "total_tokens": 0, "call_count": 0, "estimated_cost": 0.0}


class MockLLMClient:
    """Mock客户端：用消息内容判断调用类型，返回对应mock数据"""

    def __init__(self, model: str = "mock"):
        self.model = model
        self.token_usage = {"total_prompt": 0, "total_completion": 0, "total_tokens": 0, "call_count": 0, "estimated_cost": 0.0}

    def chat_json(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        self.token_usage["total_prompt"] += 500
        self.token_usage["total_completion"] += 300
        self.token_usage["total_tokens"] += 800
        self.token_usage["call_count"] += 1

        last_msg = messages[-1]["content"] if messages else ""
        result_type = self._detect_call_type(last_msg)

        if result_type == "outline":
            parsed = self._mock_outline()
        elif result_type == "querygen":
            parsed = self._mock_querygen(last_msg)
        elif result_type == "term_extract":
            parsed = self._mock_term_extract(last_msg)
        elif result_type == "review":
            parsed = self._mock_review_response()
        elif result_type == "module_a_main":
            parsed = self._mock_module_a_main(last_msg)
        elif result_type == "module_a_followup":
            parsed = self._mock_module_a_followup(last_msg)
        else:
            parsed = {"follow_up_question": "能展开讲讲吗？", "question_type": "followup", "section": "B",
                      "cot_analysis": {"weaknesses": ["回答简短"], "tech_terms_found": [], "probing_layer": "layer1", "probing_angle": "项目还原"},
                      "probing_layer": "layer1",
                      "jd_keywords_hit": ["AI产品"], "interviewer_intent": "了解更多", "fabrication_risk": "medium"}

        return {
            "content": json.dumps(parsed, ensure_ascii=False),
            "parsed": parsed,
            "usage": {"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800, "elapsed_seconds": 0.5, "estimated_cost_usd": 0.0},
            "model": "mock",
        }

    def _detect_call_type(self, last_msg: str) -> str:
        if "面试大纲" in last_msg and "生成" in last_msg:
            return "outline"
        if "RAG检索query" in last_msg and "生成" in last_msg:
            return "querygen"
        if "技术名词" in last_msg and "rag_query" in last_msg:
            return "term_extract"
        if "逐题诊断" in last_msg or "全局画像" in last_msg or "STAR重构" in last_msg:
            return "review"
        if "【主问题" in last_msg or "板块切换" in last_msg or "提出" in last_msg and "主问题" in last_msg:
            return "module_a_main"
        if "【追问" in last_msg:
            return "module_a_followup"
        return "unknown"

    def _mock_outline(self) -> Dict:
        return {
            "outline": {
                "A": {"topic": "自我介绍与实习经历", "focus": ["实习项目真实性", "AI产品理解", "表达逻辑性"], "question_directions": ["自我介绍", "最有成就感的项目", "为什么做AI产品"], "difficulty": "简单"},
                "B": {"topic": "简历深挖", "focus": ["项目细节还原", "AI技术理解深度", "个人贡献边界"], "question_directions": ["项目中具体做了什么", "遇到的最大困难", "技术方案选型理由"], "difficulty": "中等"},
                "C": {"topic": "案例设计", "focus": ["产品设计能力", "数据驱动思维", "优先级判断"], "question_directions": ["产品设计题", "费米估算", "数据分析题"], "difficulty": "中等"},
                "D": {"topic": "HR综合面", "focus": ["职业规划", "求职动机", "团队合作"], "question_directions": ["职业规划", "优缺点", "为什么选择我们"], "difficulty": "简单"},
            },
            "overall_difficulty": "中等",
            "key_risk_areas": ["实习项目贡献可能夸大", "AI技术理解可能停留在概念层面"],
        }

    def _mock_querygen(self, last_msg: str) -> Dict:
        if "板块A" in last_msg or "自我介绍" in last_msg:
            return {"rag_query": "AI产品经理 校招 自我介绍 实习经历 项目真实性"}
        elif "板块B" in last_msg or "简历深挖" in last_msg:
            return {"rag_query": "AI产品经理 校招 简历深挖 RAG技术 项目细节 数据溯源"}
        elif "板块C" in last_msg or "案例" in last_msg:
            return {"rag_query": "AI产品经理 校招 产品设计 费米估算 数据分析 MVP"}
        elif "板块D" in last_msg or "HR" in last_msg:
            return {"rag_query": "AI产品经理 HR面 职业规划 优缺点 校招"}
        else:
            return {"rag_query": "AI产品经理 校招 面试"}

    def _mock_term_extract(self, last_msg: str) -> Dict:
        tech_terms = []
        rag_query_parts = []
        keywords_map = {
            "RAG": "RAG检索增强生成",
            "embedding": "embedding向量嵌入",
            "向量": "向量检索 embedding",
            "MoE": "混合专家模型MoE",
            "Agent": "AI Agent智能体",
            "大模型": "大语言模型 LLM",
            "LLM": "大语言模型 LLM",
            "推荐": "推荐系统算法",
            "客服": "智能客服AI应用",
            "评测": "模型评测评估体系",
            "Prompt": "Prompt工程提示词优化",
        }
        for kw, full in keywords_map.items():
            if kw in last_msg:
                tech_terms.append({"term": kw, "reason": f"候选人提到了{kw}，可以深挖其理解深度"})
                rag_query_parts.append(full)
                if len(tech_terms) >= 3:
                    break
        if not tech_terms:
            tech_terms.append({"term": "AI产品", "reason": "泛泛提到AI，可以追问具体技术理解"})
            rag_query_parts.append("AI产品经理技术理解")
        return {"tech_terms": tech_terms, "rag_query": " ".join(rag_query_parts)}

    def _mock_module_a_main(self, last_msg: str) -> Dict:
        # 智能检测目标板块：优先找"切换到X板块"，再找"X板块"
        import re
        target_section = None
        # 优先匹配切换到X板块
        m = re.search(r'切换到([A-D])板块', last_msg)
        if m:
            target_section = m.group(1)
        # 再匹配【主问题 - X板块】
        elif re.search(r'主问题.*?([A-D])板块', last_msg):
            m2 = re.search(r'主问题.*?([A-D])板块', last_msg)
            target_section = m2.group(1)
        # 兜底：按关键词判断
        elif "自我介绍" in last_msg:
            target_section = "A"
        elif "简历深挖" in last_msg:
            target_section = "B"
        elif "案例" in last_msg:
            target_section = "C"
        elif "HR" in last_msg or "职业规划" in last_msg:
            target_section = "D"
        else:
            target_section = "A"

        if target_section == "A":
            return {
                "cot_analysis": {"weaknesses": [], "tech_terms_found": [], "probing_layer": "layer1", "probing_angle": "开场破冰"},
                "question_type": "main", "section": "A", "probing_layer": "layer1",
                "follow_up_question": "请做一下自我介绍，重点说说你在AI产品方向的实习经历，以及你为什么想做AI产品经理。",
                "jd_keywords_hit": ["AI产品经理", "实习经历"],
                "interviewer_intent": "开场破冰，了解候选人背景和动机",
                "fabrication_risk": "low",
            }
        elif target_section == "B":
            return {
                "cot_analysis": {"weaknesses": [], "tech_terms_found": [], "probing_layer": "layer1", "probing_angle": "项目还原"},
                "question_type": "main", "section": "B", "probing_layer": "layer1",
                "follow_up_question": "好的，接下来我们深入聊聊你的项目。请详细说说你在实习中做的那个AI相关的项目，你在其中具体负责了什么？",
                "jd_keywords_hit": ["项目经验", "个人贡献"],
                "interviewer_intent": "项目还原，建立项目基线",
                "fabrication_risk": "low",
            }
        elif target_section == "C":
            return {
                "cot_analysis": {"weaknesses": [], "tech_terms_found": [], "probing_layer": "layer1", "probing_angle": "产品设计能力考察"},
                "question_type": "main", "section": "C", "probing_layer": "layer1",
                "follow_up_question": "好的，我们换个话题。如果让你设计一款面向大学生的AI面试模拟产品，你会怎么设计它的核心功能？你认为衡量这款产品成功的最关键指标是什么？",
                "jd_keywords_hit": ["产品设计", "核心指标", "用户洞察"],
                "interviewer_intent": "考察产品设计能力和指标思维",
                "fabrication_risk": "low",
            }
        else:  # D
            return {
                "cot_analysis": {"weaknesses": [], "tech_terms_found": [], "probing_layer": "layer1", "probing_angle": "职业规划与动机"},
                "question_type": "main", "section": "D", "probing_layer": "layer1",
                "follow_up_question": "好的，最后我们聊聊个人发展。你的职业规划是什么？为什么选择AI产品经理这个方向？为什么想来我们公司？",
                "jd_keywords_hit": ["职业规划", "求职动机"],
                "interviewer_intent": "考察职业规划清晰度和文化匹配度",
                "fabrication_risk": "low",
            }


    def _mock_module_a_followup(self, last_msg: str) -> Dict:
        import re
        # 智能检测目标板块
        target_section = None
        m = re.search(r"追问.*?([A-D])板块", last_msg)
        if m:
            target_section = m.group(1)
        elif "自我介绍" in last_msg:
            target_section = "A"
        elif "简历深挖" in last_msg or "项目" in last_msg:
            target_section = "B"
        elif "案例" in last_msg or "设计" in last_msg:
            target_section = "C"
        elif "HR" in last_msg or "职业规划" in last_msg:
            target_section = "D"
        else:
            target_section = "B"

        if target_section == "A":
            return {
                "cot_analysis": {"weaknesses": ["实习项目细节模糊", "数据成果未说明"], "tech_terms_found": ["AI客服"], "probing_layer": "layer2", "probing_angle": "数据溯源+贡献边界"},
                "question_type": "followup", "section": "A",
                "probing_layer": "layer2",
                "follow_up_question": "你提到了实习项目，能具体说说这个项目的产出数据吗？作为实习生你是怎么拿到这些数据的？你个人具体做了哪部分工作？",
                "jd_keywords_hit": ["数据溯源", "贡献穿透"],
                "interviewer_intent": "数据溯源，验证经历真实性",
                "fabrication_risk": "medium",
            }
        elif target_section == "B":
            return {
                "cot_analysis": {"weaknesses": ["技术理解停留在概念层面", "可能没有实际做过"], "tech_terms_found": ["RAG", "embedding"], "probing_layer": "layer3", "probing_angle": "有罪推定式追问+技术深度验证"},
                "question_type": "followup", "section": "B",
                "probing_layer": "layer3",
                "follow_up_question": "你提到了RAG，能具体说说embedding模型你是怎么选的吗？你在实习中实际用过哪些RAG相关的技术或工具？还是说这些都是你从网上学到的概念？",
                "jd_keywords_hit": ["embedding", "技术选型", "实践经验"],
                "interviewer_intent": "贡献穿透+技术验证，检验真实深度",
                "fabrication_risk": "high",
            }
        elif target_section == "C":
            return {
                "cot_analysis": {"weaknesses": ["缺乏优先级思考", "指标定义模糊"], "tech_terms_found": ["MVP", "AB测试"], "probing_layer": "layer2", "probing_angle": "优先级判断+验证思维"},
                "question_type": "followup", "section": "C",
                "probing_layer": "layer2",
                "follow_up_question": "你列了不少功能，但如果只有2周开发时间，你会先做哪2-3个功能？为什么是这几个？你怎么验证它们是否有效？",
                "jd_keywords_hit": ["MVP", "优先级", "A/B测试"],
                "interviewer_intent": "考察优先级判断和数据驱动思维",
                "fabrication_risk": "medium",
            }
        elif target_section == "D":
            return {
                "cot_analysis": {"weaknesses": ["职业规划不够具体"], "tech_terms_found": [], "probing_layer": "layer1", "probing_angle": "深入了解动机"},
                "question_type": "followup", "section": "D",
                "probing_layer": "layer1",
                "follow_up_question": "你说想做AI产品经理，能具体说说你为这个方向做了哪些准备吗？你觉得你和其他候选人相比，你的优势和劣势分别是什么？",
                "jd_keywords_hit": ["自我认知", "求职准备"],
                "interviewer_intent": "考察自我认知和准备程度",
                "fabrication_risk": "low",
            }
        else:
            return {
                "cot_analysis": {"weaknesses": ["回答不够具体"], "tech_terms_found": [], "probing_layer": "layer2", "probing_angle": "深入了解"},
                "question_type": "followup", "section": "B",
                "probing_layer": "layer2",
                "follow_up_question": "能具体说说吗？",
                "jd_keywords_hit": [], "interviewer_intent": "深入了解", "fabrication_risk": "medium",
            }


    def _mock_review_response(self) -> Dict:
        return {
            "per_question": [
                {
                    "question": "请做一下自我介绍...",
                    "answer": "我叫张三，在字节AI产品部实习...",
                    "score": 60, "score_reason": "有实习经历但缺乏量化数据，贡献边界不清",
                    "star_diagnosis": {"S": "✅", "T": "⚠️", "A": "⚠️", "R": "❌"},
                    "capability_gap": ["数据驱动意识不足", "个人贡献表述模糊"],
                    "persona_check": {"resume_call_rate": "70%", "claim_inflation": ["参与→暗示负责"]},
                    "warning_signals": ["用我们代替我", "无量化数据"],
                    "evidence_chain": {"背景": "✅", "问题": "⚠️", "角色": "❌", "动作": "✅", "取舍": "❌", "结果": "❌"},
                    "reconstructed_answer": "我在字节跳动AI产品部实习3个月，参与了智能客服LLM升级项目。我负责TOP10高频问题的效果验证...",
                    "improvement_suggestion": "用实习公司+负责模块+量化结果+AI洞察的结构",
                    "fabrication_risk_warning": ["没有量化数据——面试官很可能让你报具体数字"],
                    "calibration": ["简历写参与，面试不要说负责"],
                },
                {
                    "question": "请详细说说你做的AI项目...",
                    "answer": "我做了智能客服的RAG功能...",
                    "score": 45, "score_reason": "技术理解停留在概念层，缺乏实践深度",
                    "star_diagnosis": {"S": "✅", "T": "✅", "A": "❌", "R": "❌"},
                    "capability_gap": ["AI技术理解不足", "缺乏实践经验"],
                    "persona_check": {"resume_call_rate": "40%", "claim_inflation": []},
                    "warning_signals": ["只讲概念不讲落地"],
                    "evidence_chain": {"背景": "✅", "问题": "✅", "角色": "❌", "动作": "❌", "取舍": "❌", "结果": "❌"},
                    "reconstructed_answer": "说实话，我实习时没有直接做RAG系统的开发，但我在做智能客服需求时接触到了RAG的整个流程...",
                    "improvement_suggestion": "遇到不熟悉的技术，诚实说明接触深度+展示学习能力比硬撑更安全",
                    "fabrication_risk_warning": ["声称了解RAG但讲不出细节——可能被技术追问穿帮"],
                    "calibration": [],
                },
                {
                    "question": "设计一款AI面试模拟产品...",
                    "answer": "核心功能是模拟面试和复盘...",
                    "score": 55, "score_reason": "有基本框架但缺乏优先级思考和验证方法",
                    "star_diagnosis": {"S": "⚠️", "T": "⚠️", "A": "⚠️", "R": "❌"},
                    "capability_gap": ["优先级判断能力待提升", "数据驱动思维不足"],
                    "persona_check": {"resume_call_rate": "30%", "claim_inflation": []},
                    "warning_signals": ["功能罗列无优先级"],
                    "evidence_chain": {"背景": "✅", "问题": "✅", "角色": "❌", "动作": "⚠️", "取舍": "❌", "结果": "❌"},
                    "reconstructed_answer": "面向大学生的AI面试模拟产品，我会聚焦简历面这个最痛的场景，MVP只做3件事...",
                    "improvement_suggestion": "先说聚焦什么场景不做什么，再讲MVP和怎么衡量成功",
                    "fabrication_risk_warning": ["指标定义太笼统——可能被追问具体怎么测"],
                    "calibration": [],
                },
                {
                    "question": "你的职业规划是什么？...",
                    "answer": "我想做AI产品经理...",
                    "score": 65, "score_reason": "有方向但不够具体，动机表述较泛",
                    "star_diagnosis": {"S": "✅", "T": "⚠️", "A": "⚠️", "R": "❌"},
                    "capability_gap": ["职业规划不够落地"],
                    "persona_check": {"resume_call_rate": "50%", "claim_inflation": []},
                    "warning_signals": ["回答模板化"],
                    "evidence_chain": {"背景": "✅", "问题": "❌", "角色": "❌", "动作": "❌", "取舍": "❌", "结果": "❌"},
                    "reconstructed_answer": "短期1-2年，我想在AI产品方向深入，把产品基本功打扎实...",
                    "improvement_suggestion": "职业规划分短期/中期/长期说，每个阶段说清楚具体目标和路径",
                    "fabrication_risk_warning": [],
                    "calibration": [],
                },
            ],
            "overall": {
                "total_score": 56,
                "capability_profile": {
                    "advantages": ["有实习经历", "对AI产品有基本认知", "表达清晰"],
                    "shortcomings": ["缺乏量化数据意识", "AI技术理解偏浅", "产品设计缺乏深度", "个人贡献表述模糊"],
                    "style_risks": ["习惯用我们代替我，贡献边界不清", "遇到不熟悉的技术容易硬撑"],
                },
                "growth_plan": {
                    "short_term": "准备3个STAR结构的实习故事，每个都要有量化数据和我的具体动作",
                    "long_term": "系统学习LLM/RAG/Agent的产品化知识，每个技术点都想如果我做产品怎么落地",
                },
            },
        }

    def chat(self, messages: List[Dict], temperature: float = 0.7, response_format=None) -> Dict:
        return self.chat_json(messages, temperature)

    def get_usage_summary(self) -> Dict:
        return {**self.token_usage, "avg_tokens_per_call": round(self.token_usage["total_tokens"] / max(self.token_usage["call_count"], 1), 1)}

    def reset_usage(self):
        self.token_usage = {"total_prompt": 0, "total_completion": 0, "total_tokens": 0, "call_count": 0, "estimated_cost": 0.0}
