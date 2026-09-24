"""20个测试用例 — 成员C主要修改这个文件
包含正常场景(12个) + 挑战场景(8个)
"""

SAMPLE_RESUME = """
教育背景：NTU MSc Enterprise AI
实习经历1：字节跳动-AI产品实习生（3个月），参与DeerFlow项目的文档优化和bad case标注
实习经历2：某创业公司-产品实习生（2个月），协助设计AI客服功能
技能：Python基础, SQL, Axure, 了解LLM/RAG/Agent概念
"""

TEST_CASES = [
    # === 正常场景 1-12 ===
    {
        "id": "TC-001", "type": "正常", "category": "自我介绍",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": SAMPLE_RESUME,
        "answers": [
            "我是NTU的AI硕士，在字节实习时参与了DeerFlow项目的文档优化和bad case标注，还帮创业公司设计了AI客服功能。对LLM和RAG有一定了解。",
            "在字节实习时，我负责了DeerFlow项目的文档模块，分析了约100条bad case，分类了3类问题模式，在mentor指导下完成了文档优化建议。",
        ],
        "eval_focus": "追问是否针对声明中的实习贡献和角色边界"
    },
    {
        "id": "TC-002", "type": "正常", "category": "STAR实习经历",
        "jd": "AI产品经理校招", "style": "行为导向",
        "resume": SAMPLE_RESUME,
        "answers": [
            "我在字节实习时参与了DeerFlow项目。当时项目文档质量参差不齐，用户反馈找不到想要的信息。我分析了100条bad case，发现3类问题：术语不一致、步骤缺失、示例过时。我重写了核心模块的文档，在mentor审核后合并。用户文档查找效率提升了约20%。",
        ],
        "eval_focus": "3层穿透触发（项目还原层）"
    },
    {
        "id": "TC-003", "type": "正常", "category": "用户调研方法",
        "jd": "AI产品经理校招", "style": "数据驱动",
        "resume": SAMPLE_RESUME,
        "answers": [
            "在创业公司实习时，我通过问卷和访谈收集了30个用户对AI客服的反馈，发现主要问题是回答不相关和语气生硬。我整理了反馈分类，和算法同学沟通后调整了Prompt模板。",
        ],
        "eval_focus": "追问调研方法执行细节"
    },
    {
        "id": "TC-004", "type": "正常", "category": "优先级划分",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": SAMPLE_RESUME,
        "answers": [
            "我会按用户影响面和实现成本两个维度来排优先级。比如AI客服的准确率修复影响所有用户且成本低，优先级最高。多轮对话功能影响部分用户但开发成本高，排第二。",
        ],
        "eval_focus": "追问优先级方法论"
    },
    {
        "id": "TC-005", "type": "正常", "category": "AI产品增长策略",
        "jd": "AI产品经理校招", "style": "数据驱动",
        "resume": SAMPLE_RESUME,
        "answers": [
            "在创业公司时，我们通过优化AI客服的回答质量来提升使用率。我分析了用户留存数据，发现回答准确率每提升5%，次日留存提升约2%。所以我们优先优化了高频问题的回答准确率。",
        ],
        "eval_focus": "数据溯源层追问"
    },
    {
        "id": "TC-006", "type": "正常", "category": "AI产品赋能场景",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": SAMPLE_RESUME,
        "answers": [
            "我认为AI可以赋能教育场景。比如用LLM+RAG做一个AI学习助手，根据学生的课程内容生成个性化练习题和解析。关键是确保答案准确，需要有评估机制。",
        ],
        "eval_focus": "追问AI技术理解深度"
    },
    {
        "id": "TC-007", "type": "正常", "category": "数据可视化平台设计",
        "jd": "AI产品经理校招", "style": "行为导向",
        "resume": SAMPLE_RESUME,
        "answers": [
            "如果要设计一个数据可视化平台，我会先定义核心用户是运营和产品经理，他们的需求是快速看趋势和异常。核心功能是自定义看板和异常告警。AI可以自动生成数据洞察文字。",
        ],
        "eval_focus": "追问B端产品设计逻辑"
    },
    {
        "id": "TC-008", "type": "正常", "category": "效率提升数据",
        "jd": "AI产品经理校招", "style": "压力测试",
        "resume": SAMPLE_RESUME,
        "answers": [
            "在创业公司实习时，我设计的AI客服功能上线后，工单处理效率提升了99%。主要是通过自动回复常见问题，减少了人工处理量。",
        ],
        "eval_focus": "质疑数据真实性（数据溯源层）"
    },
    {
        "id": "TC-009", "type": "正常", "category": "STAR失败经历",
        "jd": "AI产品经理校招", "style": "行为导向",
        "resume": SAMPLE_RESUME,
        "answers": [
            "实习时我曾尝试用LLM自动生成产品文档，但生成的文档经常出现幻觉和不准确的技术描述。我反思后发现是prompt太宽泛，后来改为分段生成+人工审核的方式，质量好多了。",
        ],
        "eval_focus": "追问是否深入到反思层面"
    },
    {
        "id": "TC-010", "type": "正常", "category": "费米估算",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": SAMPLE_RESUME,
        "answers": [
            "估算某AI产品日调用量：假设用户数100万，日活率20%即20万日活，平均每用户每天调用3次，所以日调用量约60万次。考虑到峰值和低谷，可能区间在40-80万次。",
        ],
        "eval_focus": "追问逻辑链拆解"
    },
    {
        "id": "TC-011", "type": "正常", "category": "AI产品设计题",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": SAMPLE_RESUME,
        "answers": [
            "为老年用户设计AI健康助手，核心场景是用药提醒和健康咨询。我会用语音交互降低使用门槛，AI回答需要有医学知识库支撑（RAG），并设置紧急联系功能。关键是确保医疗信息准确性。",
        ],
        "eval_focus": "追问用户细分和AI能力边界"
    },
    {
        "id": "TC-012", "type": "正常", "category": "AI产品留存策略",
        "jd": "AI产品经理校招", "style": "数据驱动",
        "resume": SAMPLE_RESUME,
        "answers": [
            "提升AI助手次日留存，我会做用户分群：新用户vs老用户，高频vs低频。对新用户优化首次体验，确保前3轮对话高质量。对低频用户做push召回，展示AI新功能。用A/B测试验证效果。",
        ],
        "eval_focus": "追问用户分群+A/B测试+AI指标"
    },
    # === 挑战场景 13-20 ===
    {
        "id": "TC-013", "type": "挑战-模糊", "category": "模糊回答",
        "jd": "AI产品经理校招", "style": "压力测试",
        "resume": SAMPLE_RESUME,
        "answers": ["实习时做了个AI项目，效果还不错。"],
        "eval_focus": "能否识别模糊并追问细节"
    },
    {
        "id": "TC-014", "type": "挑战-模糊", "category": "模糊数据",
        "jd": "AI产品经理校招", "style": "数据驱动",
        "resume": SAMPLE_RESUME,
        "answers": ["AI助手准确率提升了50%。"],
        "eval_focus": "追问AI技术细节和评估方法"
    },
    {
        "id": "TC-015", "type": "挑战-夸大", "category": "夸大经历",
        "jd": "AI产品经理校招", "style": "压力测试",
        "resume": SAMPLE_RESUME,
        "answers": ["我独立设计了整个AI推荐系统架构，从0到1负责了全部产品决策。"],
        "eval_focus": "贡献穿透层+实习生权限追问（阿酥式场景）"
    },
    {
        "id": "TC-016", "type": "挑战-夸大", "category": "夸大数据",
        "jd": "AI产品经理校招", "style": "压力测试",
        "resume": SAMPLE_RESUME,
        "answers": ["我负责的AI产品DAU过百万，是公司最成功的AI产品。"],
        "eval_focus": "数据溯源+人设矛盾检测"
    },
    {
        "id": "TC-017", "type": "挑战-缺失", "category": "数据缺失",
        "jd": "AI产品经理校招", "style": "数据驱动",
        "resume": SAMPLE_RESUME,
        "answers": ["实习时优化了AI客服的回答质量，用户反馈好了很多。"],
        "eval_focus": "数据驱动型是否追问量化"
    },
    {
        "id": "TC-018", "type": "挑战-误导", "category": "答非所问",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": SAMPLE_RESUME,
        "answers": ["（被问到用户调研方法）我觉得AI产品最重要的是技术能力，比如要理解Transformer的原理和attention机制。"],
        "eval_focus": "考官能否拉回正题"
    },
    {
        "id": "TC-019", "type": "挑战-缺失", "category": "能力缺失",
        "jd": "AI产品经理校招", "style": "逻辑严谨",
        "resume": "教育背景：某大学管理学本科\n实习经历：无AI相关实习\n技能：PPT, Excel",
        "answers": ["我对AI产品很感兴趣，虽然没有直接经验，但我平时经常用ChatGPT。"],
        "eval_focus": "Module B识别AI能力缺口"
    },
    {
        "id": "TC-020", "type": "挑战-矛盾", "category": "前后矛盾",
        "jd": "AI产品经理校招", "style": "压力测试",
        "resume": SAMPLE_RESUME,
        "answers": [
            "我在字节实习时负责了DeerFlow项目的核心模块设计。",
            "其实我在DeerFlow主要是做一些文档标注和辅助性工作，核心设计是mentor做的。",
        ],
        "eval_focus": "逻辑矛盾检测+话术校准"
    },
]


def get_all_cases():
    return TEST_CASES


def get_cases_by_type(case_type: str):
    return [c for c in TEST_CASES if c["type"] == case_type or c["type"].startswith(case_type)]


def get_case_by_id(case_id: str):
    for c in TEST_CASES:
        if c["id"] == case_id:
            return c
    return None
