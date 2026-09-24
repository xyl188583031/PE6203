"""Interview Orchestration: Agent-orchestrated 4-section semi-structured interview (AaBbCcDd)

Upgrades:
1. Main questions and follow-ups both use Advanced RAG pipeline
2. Adaptive retrieval: CoT decides if RAG is needed
3. Iterative retrieval: auto-rewrite query if first round results are poor
4. Structured injection: format_rag_structured replaces simple formatting
5. Bilingual support: lang parameter controls prompt and output language
"""

import json
from typing import Dict, List, Optional

from rag.retriever import HybridRetriever
from prompts.module_a import (
    build_outline_messages,
    build_querygen_messages,
    build_term_extraction_messages,
    build_module_a_messages,
)
from prompts.module_b import build_module_b_messages


SECTIONS_ZH = [
    {"label": "A", "topic": "自我介绍与实习经历", "topic_en": "self_intro"},
    {"label": "B", "topic": "简历深挖", "topic_en": "resume_deep_dive"},
    {"label": "C", "topic": "案例设计", "topic_en": "case_design"},
    {"label": "D", "topic": "HR综合面", "topic_en": "hr_interview"},
]

SECTIONS_EN = [
    {"label": "A", "topic": "Self-Introduction & Internship", "topic_en": "self_intro"},
    {"label": "B", "topic": "Resume Deep-Dive", "topic_en": "resume_deep_dive"},
    {"label": "C", "topic": "Case Design", "topic_en": "case_design"},
    {"label": "D", "topic": "HR Interview", "topic_en": "hr_interview"},
]

SECTIONS = SECTIONS_ZH

FOLLOWUPS_PER_SECTION = 1


class InterviewSession:
    def __init__(self, llm_client, retriever: HybridRetriever, resume_context: str,
                 jd_context: str, jd_tags: list, interviewer_style: str = "逻辑严谨",
                 lang: str = "zh"):
        self.llm = llm_client
        self.retriever = retriever
        if hasattr(self.retriever, 'llm') and self.retriever.llm is None:
            self.retriever.llm = llm_client
        self.resume_context = resume_context
        self.jd_context = jd_context
        self.jd_tags = jd_tags
        self.interviewer_style = interviewer_style
        self.lang = lang

        self.sections = SECTIONS_EN if lang == "en" else SECTIONS_ZH

        self.interview_outline: Dict = {}
        self.section_index = 0
        self.followup_round = 0
        self.current_round = 0
        self.total_rounds = len(self.sections) * (1 + FOLLOWUPS_PER_SECTION)

        self.interview_log: List[Dict] = []
        self.conversation_history: List[Dict] = []

        self._is_started = False
        self._is_ended = False

    @property
    def is_ended(self) -> bool:
        return self._is_ended

    @property
    def progress_text(self) -> str:
        if not self._is_started:
            return "Ready to start" if self.lang == "en" else "准备开始"
        if self._is_ended:
            return "Interview ended" if self.lang == "en" else "面试结束"
        section = self.sections[self.section_index]
        total = len(self.sections)
        if self.followup_round == 0:
            if self.lang == "en":
                return f"{section['label']} Main / {total} sections"
            return f"{section['label']} 主问题 / 共{total}板块"
        if self.lang == "en":
            return f"{section['label']} Follow-up / {total} sections"
        return f"{section['label']} 追问 / 共{total}板块"

    def start(self) -> Dict:
        outline_messages = build_outline_messages(
            resume_context=self.resume_context,
            jd_context=self.jd_context,
            interviewer_style=self.interviewer_style,
            lang=self.lang,
        )
        outline_result = self.llm.chat_json(outline_messages)
        self.interview_outline = outline_result.get("parsed", {})

        self._is_started = True
        self.section_index = 0
        self.followup_round = 0
        self.current_round = 1

        return self._ask_main_question(0, is_first=True)

    def answer_and_next(self, user_answer: str) -> Dict:
        if self.interview_log and self.interview_log[-1].get("user_answer", "") == "":
            self.interview_log[-1]["user_answer"] = user_answer
        self.conversation_history.append({"role": "user", "content": user_answer})

        if self.followup_round < FOLLOWUPS_PER_SECTION:
            self.followup_round += 1
            return self._ask_followup(user_answer)
        elif self.section_index < len(self.sections) - 1:
            self.section_index += 1
            self.followup_round = 0
            self.current_round += 1
            return self._ask_main_question(self.section_index, is_first=False)
        else:
            self._is_ended = True
            end_msg = "Interview ended" if self.lang == "en" else "面试结束"
            return {
                "question": end_msg,
                "round": self.current_round,
                "question_type": "end",
                "ended": True,
                "progress": end_msg,
            }

    def _ask_main_question(self, section_idx: int, is_first: bool = False) -> Dict:
        section = self.sections[section_idx]
        outline_section = self.interview_outline.get("outline", {}).get(section["label"], {})
        section_focus = outline_section.get("focus", [])
        focus_str = ", ".join(section_focus) if isinstance(section_focus, list) else str(section_focus)

        query_messages = build_querygen_messages(
            resume_context=self.resume_context,
            jd_context=self.jd_context,
            section_label=section["label"],
            section_topic=section["topic"],
            section_focus=focus_str,
            lang=self.lang,
        )
        query_result = self.llm.chat_json(query_messages)
        rag_query = query_result.get("parsed", {}).get("rag_query", section["topic"])

        rag_categories = self._get_rag_categories(section["label"])
        rag_results = self.retriever.retrieve(
            f"{rag_query} {self.jd_context}",
            top_k=3,
            categories=rag_categories,
        )

        rag_text = self.retriever.format_rag_structured(rag_results)

        previous_label = self.sections[section_idx - 1]["label"] if section_idx > 0 else ""
        is_switch = section_idx > 0

        messages = build_module_a_messages(
            resume_context=self.resume_context,
            jd_context=self.jd_context,
            jd_keywords=self.jd_tags,
            rag_results=rag_text,
            interviewer_style=self.interviewer_style,
            interview_outline=json.dumps(self.interview_outline, ensure_ascii=False, indent=2),
            question_type="main",
            section_label=section["label"],
            section_topic=section["topic"],
            conversation_history=self.conversation_history,
            user_answer="",
            is_section_switch=is_switch,
            previous_section=previous_label,
            lang=self.lang,
        )

        result = self.llm.chat_json(messages)
        parsed = result.get("parsed", {})
        fallback_q = f"Please share your understanding of {section['topic']}." if self.lang == "en" else f"请说说你对{section['topic']}的理解。"
        question = parsed.get("follow_up_question", fallback_q)

        log_entry = {
            "round": self.current_round,
            "question_type": "main",
            "section": section["label"],
            "section_topic": section["topic"],
            "question": question,
            "user_answer": "",
            "cot_analysis": parsed.get("cot_analysis", {}),
            "interviewer_intent": parsed.get("interviewer_intent", ""),
            "probing_layer": parsed.get("probing_layer", "layer1"),
            "jd_keywords": parsed.get("jd_keywords_hit", []),
            "fabrication_risk": parsed.get("fabrication_risk", "low"),
            "rag_query": rag_query,
            "rag_results": [{"id": r["item"]["id"], "score": round(r["score"], 3),
                             "category": r["item"].get("category", "")} for r in rag_results],
            "rag_insights": parsed.get("cot_analysis", {}).get("rag_insights", []),
        }
        self.interview_log.append(log_entry)
        self.conversation_history.append({"role": "assistant", "content": question})

        return {
            "question": question,
            "round": self.current_round,
            "question_type": "main",
            "section": section["label"],
            "section_topic": section["topic"],
            "rag_results": rag_results,
            "rag_query": rag_query,
            "rag_insights": parsed.get("cot_analysis", {}).get("rag_insights", []),
            "intent": parsed.get("interviewer_intent", ""),
            "probing_layer": parsed.get("probing_layer", ""),
            "fabrication_risk": parsed.get("fabrication_risk", ""),
            "cot_analysis": parsed.get("cot_analysis", {}),
            "usage": result.get("usage", {}),
            "ended": False,
            "progress": self.progress_text,
        }

    def _ask_followup(self, user_answer: str) -> Dict:
        section = self.sections[self.section_index]

        term_messages = build_term_extraction_messages(
            user_answer=user_answer,
            section_topic=section["topic"],
            lang=self.lang,
        )
        term_result = self.llm.chat_json(term_messages)
        term_parsed = term_result.get("parsed", {})
        tech_rag_query = term_parsed.get("rag_query", "")
        tech_terms = term_parsed.get("tech_terms", [])

        needs_rag = self._should_retrieve(user_answer, tech_terms, section)

        rag_results = []
        rag_text = ""
        rag_insights_used = []

        if needs_rag:
            rag_results = self.retriever.retrieve_advanced(
                user_answer=user_answer,
                section_label=section["label"],
                section_topic=section["topic"],
                jd_context=self.jd_context,
                question_type="followup",
            )

            if len(rag_results) < 2 and tech_rag_query:
                fallback_results = self.retriever.retrieve(
                    tech_rag_query,
                    top_k=3,
                    categories=["tech_docs", "interview_tips"],
                )
                existing_ids = {r["item"]["id"] for r in rag_results}
                for r in fallback_results:
                    if r["item"]["id"] not in existing_ids:
                        rag_results.append(r)

            rag_text = self.retriever.format_rag_structured(rag_results)

        no_rag_msg = (
            "[RAG Results]\n(RAG retrieval skipped this round. Please ask based on your judgment.)\n"
            if self.lang == "en"
            else "【RAG检索结果】\n（本次跳过RAG检索，请根据你的判断提问）\n"
        )

        messages = build_module_a_messages(
            resume_context=self.resume_context,
            jd_context=self.jd_context,
            jd_keywords=self.jd_tags,
            rag_results=rag_text if rag_text else no_rag_msg,
            interviewer_style=self.interviewer_style,
            interview_outline=json.dumps(self.interview_outline, ensure_ascii=False, indent=2),
            question_type="followup",
            section_label=section["label"],
            section_topic=section["topic"],
            conversation_history=self.conversation_history,
            user_answer=user_answer,
            lang=self.lang,
        )

        result = self.llm.chat_json(messages)
        parsed = result.get("parsed", {})
        fallback_q = "Could you elaborate on that?" if self.lang == "en" else "能展开讲讲吗？"
        question = parsed.get("follow_up_question", fallback_q)

        rag_insights_used = parsed.get("cot_analysis", {}).get("rag_insights", [])

        log_entry = {
            "round": self.current_round,
            "question_type": "followup",
            "section": section["label"],
            "section_topic": section["topic"],
            "followup_round": self.followup_round,
            "question": question,
            "user_answer": "",
            "cot_analysis": parsed.get("cot_analysis", {}),
            "tech_terms_found": tech_terms,
            "interviewer_intent": parsed.get("interviewer_intent", ""),
            "probing_layer": parsed.get("probing_layer", ""),
            "jd_keywords": parsed.get("jd_keywords_hit", []),
            "fabrication_risk": parsed.get("fabrication_risk", "low"),
            "rag_query": tech_rag_query,
            "rag_used": needs_rag,
            "rag_results": [{"id": r["item"]["id"], "score": round(r.get("rerank_score", r.get("score", 0)), 3),
                             "category": r["item"].get("category", "")} for r in rag_results],
            "rag_insights": rag_insights_used,
        }
        self.interview_log.append(log_entry)
        self.conversation_history.append({"role": "assistant", "content": question})

        return {
            "question": question,
            "round": self.current_round,
            "question_type": "followup",
            "section": section["label"],
            "section_topic": section["topic"],
            "followup_round": self.followup_round,
            "rag_results": rag_results,
            "rag_query": tech_rag_query,
            "rag_used": needs_rag,
            "tech_terms": tech_terms,
            "rag_insights": rag_insights_used,
            "intent": parsed.get("interviewer_intent", ""),
            "probing_layer": parsed.get("probing_layer", ""),
            "fabrication_risk": parsed.get("fabrication_risk", ""),
            "cot_analysis": parsed.get("cot_analysis", {}),
            "usage": result.get("usage", {}),
            "ended": False,
            "progress": self.progress_text,
        }

    def _should_retrieve(self, user_answer: str, tech_terms: list, section: dict) -> bool:
        if tech_terms:
            return True

        if section["label"] in ("B", "C"):
            return True

        vague_patterns = ["大概", "差不多", "约", "左右", "效果不错", "有提升", "还可以", "比较好",
                         "about", "around", "approximately", "pretty good", "improved", "decent"]
        if any(p in user_answer.lower() for p in vague_patterns):
            return True

        if len(user_answer) < 30:
            return False

        if section["label"] == "D":
            return False

        return True

    def generate_review(self) -> Dict:
        log_text = json.dumps(self.interview_log, ensure_ascii=False, indent=2)

        weakness_query = self._extract_weakness_query()

        tech_rag = self.retriever.retrieve(weakness_query, top_k=2, categories=["tech_docs"])
        tips_rag = self.retriever.retrieve(weakness_query, top_k=3,
                                            categories=["interview_tips", "STAR高分回答样例", "现代面试技巧方法论"])
        all_rag = tech_rag + tips_rag
        rag_text = self.retriever.format_rag_structured(all_rag)

        messages = build_module_b_messages(
            resume_context=self.resume_context,
            jd_context=self.jd_context,
            interviewer_style=self.interviewer_style,
            interview_log=log_text,
            rag_examples=rag_text,
            lang=self.lang,
        )

        result = self.llm.chat_json(messages)
        parsed = result.get("parsed", {})

        return {
            "review": parsed,
            "rag_query_used": weakness_query,
            "rag_results_count": len(all_rag),
            "usage": result.get("usage", {}),
        }

    def _extract_weakness_query(self) -> str:
        weaknesses = []
        for entry in self.interview_log:
            cot = entry.get("cot_analysis", {})
            ws = cot.get("weaknesses", [])
            if ws:
                weaknesses.extend(ws)

        unique_ws = list(dict.fromkeys(weaknesses))[:5]
        if unique_ws:
            return " ".join(unique_ws)
        return "AI product manager interview improvement tips" if self.lang == "en" else "AI产品经理 面试改进建议"

    def _get_rag_categories(self, section_label: str) -> List[str]:
        base_cats = ["AI PM校招高频题", "真实面试实录", "interview_tips"]
        if section_label == "A":
            return base_cats + ["企业真实JD"]
        elif section_label == "B":
            return base_cats + ["tech_docs", "行业评估量表"]
        elif section_label == "C":
            return base_cats + ["tech_docs", "行业评估量表"]
        elif section_label == "D":
            return ["interview_tips"]
        return base_cats

    def get_log(self) -> List[Dict]:
        return self.interview_log

    def get_conversation(self) -> List[Dict]:
        return self.conversation_history

    def get_outline(self) -> Dict:
        return self.interview_outline


def run_variant_test(llm_client, retriever: HybridRetriever, test_case: Dict, variant: str = "C",
                     lang: str = "zh") -> Dict:
    session = InterviewSession(
        llm_client=llm_client,
        retriever=retriever,
        resume_context=test_case.get("resume", ""),
        jd_context=test_case.get("jd", "AI产品经理校招"),
        jd_tags=test_case.get("jd_tags", []),
        interviewer_style=test_case.get("style", "逻辑严谨"),
        lang=lang,
    )

    session.start()
    for answer in test_case.get("answers", []):
        result = session.answer_and_next(answer)
        if result.get("ended"):
            break

    review = session.generate_review()
    log = session.get_log()

    return {
        "variant": variant,
        "test_case_id": test_case.get("id"),
        "questions": [entry["question"] for entry in log],
        "question_types": [entry.get("question_type", "") for entry in log],
        "sections": [entry.get("section", "") for entry in log],
        "probing_layers": [entry.get("probing_layer", "") for entry in log],
        "fabrication_risks": [entry.get("fabrication_risk", "") for entry in log],
        "cot_analyses": [entry.get("cot_analysis", {}) for entry in log],
        "rag_used": [entry.get("rag_used", True) for entry in log],
        "rag_insights": [entry.get("rag_insights", []) for entry in log],
        "review": review.get("review", {}),
        "usage": session.llm.get_usage_summary(),
    }
