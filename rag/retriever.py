"""RAG 混合检索系统：Advanced RAG 架构

升级内容：
- Layer 0: 元数据过滤（scenario_tags / strategy_type / difficulty）
- Pre-Retrieval: 场景路由 + 多视角Query改写 + HyDE
- Retrieval: 多路召回 + 动态权重 + 动态Top-K
- Post-Retrieval: LLM重排 + 阈值过滤 + 结构化注入
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi


KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# 场景路由映射
SCENARIO_ROUTE = {
    "A_main": ["self_intro"],
    "A_followup": ["self_intro", "project_dig"],
    "B_main": ["project_dig"],
    "B_followup": ["project_dig", "tech_probe"],
    "C_main": ["case_design", "tech_probe"],
    "C_followup": ["tech_probe", "case_design"],
    "D_main": ["hr_comprehensive"],
    "D_followup": ["hr_comprehensive"],
}

# 默认动态权重
DEFAULT_WEIGHTS = {"bm25": 0.4, "embedding": 0.6}

# 场景化权重
SCENARIO_WEIGHTS = {
    "tech_probe": {"bm25": 0.6, "embedding": 0.4},
    "case_design": {"bm25": 0.3, "embedding": 0.7},
    "hr_comprehensive": {"bm25": 0.3, "embedding": 0.7},
    "project_dig": {"bm25": 0.4, "embedding": 0.6},
    "self_intro": {"bm25": 0.3, "embedding": 0.7},
}

# 场景化Top-K
SCENARIO_TOP_K = {
    "tech_probe": 5,
    "case_design": 4,
    "hr_comprehensive": 2,
    "project_dig": 4,
    "self_intro": 2,
}


def load_knowledge_base() -> List[Dict]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    return [w for w in text.split() if len(w) > 0]


class HybridRetriever:
    def __init__(self, embedding_fn=None, llm_client=None):
        self.kb = load_knowledge_base()
        self.corpus = [item.get("embedding_text", item.get("question", "")) for item in self.kb]
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.corpus])
        self.embedding_fn = embedding_fn
        self.llm = llm_client
        self.chroma_client = None
        self.collection = None
        self._init_chroma()

    def _init_chroma(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self.collection = self.chroma_client.get_or_create_collection("mianba_kb")
        except Exception:
            self.collection = self.chroma_client.create_collection("mianba_kb")

        existing = self.collection.count()
        if existing < len(self.kb) or existing == 0:
            self._build_index()

    def _build_index(self):
        try:
            self.chroma_client.delete_collection("mianba_kb")
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection("mianba_kb")
        for item in self.kb:
            text = item.get("embedding_text", item.get("question", ""))
            self.collection.add(
                ids=[item["id"]],
                documents=[text],
                metadatas=[{
                    "id": item["id"],
                    "category": item.get("category", ""),
                    "title": item.get("title", item.get("question", ""))[:200],
                    "scenario_tags": ",".join(item.get("scenario_tags", [])),
                    "strategy_type": item.get("strategy_type") or "",
                    "difficulty": item.get("difficulty") or "",
                }],
            )

    # ===== Layer 0: 元数据过滤 =====

    def filter_by_scenario(self, scenario_tags: List[str]) -> List[int]:
        """按场景标签过滤知识库，返回匹配的索引"""
        if not scenario_tags:
            return list(range(len(self.kb)))
        matched = []
        for i, item in enumerate(self.kb):
            item_tags = item.get("scenario_tags", [])
            if any(t in item_tags for t in scenario_tags):
                matched.append(i)
        return matched if matched else list(range(len(self.kb)))

    # ===== Pre-Retrieval =====

    def route_scenario(self, section_label: str, question_type: str) -> List[str]:
        """场景路由：根据板块和问题类型返回场景标签"""
        key = f"{section_label}_{question_type}"
        return SCENARIO_ROUTE.get(key, ["project_dig"])

    def get_dynamic_weights(self, scenario_tags: List[str]) -> Tuple[float, float]:
        """动态权重：根据场景返回BM25和向量的权重"""
        for tag in scenario_tags:
            if tag in SCENARIO_WEIGHTS:
                w = SCENARIO_WEIGHTS[tag]
                return w["bm25"], w["embedding"]
        return DEFAULT_WEIGHTS["bm25"], DEFAULT_WEIGHTS["embedding"]

    def get_dynamic_top_k(self, scenario_tags: List[str]) -> int:
        """动态Top-K"""
        for tag in scenario_tags:
            if tag in SCENARIO_TOP_K:
                return SCENARIO_TOP_K[tag]
        return 3

    def multi_query_rewrite(self, user_answer: str, section_label: str,
                            section_topic: str, jd_context: str) -> List[str]:
        """多视角Query改写：生成3个不同视角的检索Query

        如果没有LLM，退化为简单的关键词拼接
        """
        if not self.llm:
            # 退化模式：生成3个简化query
            return [
                f"{section_topic} 技术追问 深度验证",
                f"AI产品经理 实习 追问 贡献验证",
                f"面试追问技巧 数据溯源 编造识别",
            ]

        prompt = f"""你是面试官的检索助手。请基于以下信息，生成3个不同视角的检索Query。

【候选人回答】{user_answer[:500]}
【当前板块】{section_label}: {section_topic}
【JD信息】{jd_context[:200]}

生成3个Query，分别从以下视角：
1. 技术知识视角：把回答翻译成技术术语，用于检索技术文档
2. 面试策略视角：把回答翻译成追问场景，用于检索面试经验
3. 风险探测视角：从"可能造假"的角度，用于检索编造识别策略

输出JSON格式：
{{"queries": ["query1", "query2", "query3"]}}
只输出JSON。"""

        try:
            result = self.llm.chat_json([{"role": "user", "content": prompt}])
            queries = result.get("parsed", {}).get("queries", [])
            if queries:
                return queries
        except Exception:
            pass

        # 退化
        return [
            f"{section_topic} 技术追问 深度验证",
            f"AI产品经理 实习 追问 贡献验证",
            f"面试追问技巧 数据溯源 编造识别",
        ]

    # ===== Retrieval =====

    def multi_recall(self, queries: List[str], scenario_tags: List[str],
                     recall_k: int = 10) -> List[Dict]:
        """多路召回：对多个Query分别检索，合并去重"""
        all_results = {}
        bm25_w, emb_w = self.get_dynamic_weights(scenario_tags)

        # 按场景过滤
        matched_indices = self.filter_by_scenario(scenario_tags)
        if not matched_indices:
            return []

        filtered_kb = [self.kb[i] for i in matched_indices]
        filtered_corpus = [self.corpus[i] for i in matched_indices]
        filtered_bm25 = BM25Okapi([tokenize(doc) for doc in filtered_corpus])

        # 向量检索的where条件
        where_clause = None
        if scenario_tags:
            # Chroma的$in逻辑不直接支持list in list，用category过滤兜底
            matched_ids = [item["id"] for item in filtered_kb]

        for query in queries:
            tokens = tokenize(query)

            # BM25
            bm25_scores = filtered_bm25.get_scores(tokens)

            # 向量检索
            emb_map = {}
            try:
                n_results = min(recall_k, len(filtered_kb))
                emb_results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                )
                if emb_results and emb_results.get("ids"):
                    for idx, doc_id in enumerate(emb_results["ids"][0]):
                        distance = emb_results["distances"][0][idx] if "distances" in emb_results else 1.0
                        emb_map[doc_id] = 1.0 - distance
            except Exception:
                pass

            max_bm25 = max(bm25_scores) if bm25_scores.size > 0 and max(bm25_scores) > 0 else 1.0
            max_emb = max(emb_map.values()) if emb_map else 1.0

            for i, item in enumerate(filtered_kb):
                item_id = item["id"]
                bm25_norm = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0
                emb_norm = emb_map.get(item_id, 0) / max_emb if max_emb > 0 else 0
                score = bm25_w * bm25_norm + emb_w * emb_norm

                if item_id not in all_results or score > all_results[item_id]["score"]:
                    all_results[item_id] = {
                        "item": item,
                        "score": score,
                        "bm25": bm25_norm,
                        "emb": emb_norm,
                    }

        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:recall_k]

    # ===== Post-Retrieval =====

    def llm_rerank(self, results: List[Dict], user_answer: str,
                  section_label: str, section_topic: str) -> List[Dict]:
        """LLM重排：对召回结果逐条打分重排

        如果没有LLM，退化为原始排序
        """
        if not self.llm or not results:
            return results

        # 准备打分Prompt
        items_text = []
        for i, r in enumerate(results):
            item = r["item"]
            title = item.get("title", item.get("question", ""))
            cat = item.get("category", "")
            strategy = item.get("strategy_type", "")
            triggers = ", ".join(item.get("trigger_patterns", [])[:3])
            templates = " | ".join(item.get("question_templates", [])[:2])
            reds = ", ".join(item.get("red_flags", [])[:2])
            fab = ", ".join(item.get("fabrication_patterns", [])[:2])
            tags = ", ".join(item.get("scenario_tags", []))

            items_text.append(
                f"[{i+1}] ID:{item['id']} 类别:{cat} 场景:{tags}\n"
                f"  标题:{title[:60]}\n"
                f"  策略类型:{strategy} 触发词:{triggers}\n"
                f"  追问话术:{templates}\n"
                f"  警惕信号:{reds} 编造识别:{fab}"
            )

        prompt = f"""你是面试官的检索质量评估助手。请对以下检索结果打分（1-5分）。

【候选人回答】{user_answer[:300]}
【当前板块】{section_label}: {section_topic}

【检索结果】
{chr(10).join(items_text)}

请从三个维度打分：
1. 内容相关性(Content Relevance)：内容与当前回答/场景的匹配程度
2. 策略可用性(Strategy Usability)：能否直接转化为追问角度或探测问题
3. 深度匹配度(Depth Match)：难度是否与当前场景匹配

总分 = 0.4*内容相关 + 0.4*策略可用 + 0.2*深度匹配

输出JSON：
{{"scores": [{{"index": 1, "total": 4.5, "relevance": 5, "usability": 4, "depth": 4}}, ...]}}
只输出JSON。"""

        try:
            result = self.llm.chat_json([{"role": "user", "content": prompt}])
            scores = result.get("parsed", {}).get("scores", [])

            # 创建索引到分数的映射
            score_map = {}
            for s in scores:
                idx = s.get("index", 0) - 1
                score_map[idx] = s.get("total", 0)

            # 更新结果的score并重排
            for i, r in enumerate(results):
                if i in score_map:
                    r["rerank_score"] = score_map[i]
                else:
                    r["rerank_score"] = r["score"]

            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception:
            pass

        return results

    def filter_by_threshold(self, results: List[Dict],
                             threshold: float = 2.5,
                             max_results: int = 3) -> List[Dict]:
        """阈值过滤：去掉低于阈值的结果"""
        if not results:
            return []

        # 优先用rerank_score，没有就用score
        filtered = []
        for r in results:
            score = r.get("rerank_score", r.get("score", 0))
            # rerank_score是1-5分，score是0-1分，需要归一化
            if score > 1.5:  # rerank_score范围
                normalized = score
            else:
                normalized = score * 5  # 把0-1映射到0-5

            if normalized >= threshold:
                filtered.append(r)

        # 如果过滤后为空但原始有结果，保留最高分的1条
        if not filtered and results:
            return [results[0]]

        return filtered[:max_results]

    # ===== 结构化注入格式化 =====

    def format_rag_structured(self, results: List[Dict]) -> str:
        """结构化注入：把RAG结果按结构化格式输出"""
        if not results:
            return "【RAG检索结果】\n（本次未检索到高相关度内容，请根据你的判断提问）\n"

        lines = ["【RAG检索结果 - 按相关度排序】\n"]

        # 分组：面试策略类 vs 技术参考类
        strategy_items = []
        tech_items = []
        other_items = []

        for r in results:
            item = r.get("item", r) if isinstance(r, dict) else r
            cat = item.get("category", "")

            if cat in ("interview_tips", "现代面试技巧方法论", "真实面试实录", "STAR高分回答样例"):
                strategy_items.append(r)
            elif cat == "tech_docs":
                tech_items.append(r)
            else:
                other_items.append(r)

        # 面试策略
        if strategy_items:
            lines.append(f"=== 面试策略（{len(strategy_items)}条）===")
            for i, r in enumerate(strategy_items):
                item = r.get("item", r) if isinstance(r, dict) else r
                score = r.get("rerank_score", r.get("score", 0))
                title = item.get("title", item.get("question", ""))
                strategy = item.get("strategy_type", "")
                triggers = ", ".join(item.get("trigger_patterns", [])[:3])
                templates = item.get("question_templates", [])
                reds = item.get("red_flags", [])

                lines.append(f"\n[{i+1} · 相关度{score:.1f}] {title}")
                if strategy:
                    lines.append(f"  策略类型:{strategy} 触发词:{triggers}")
                if templates:
                    lines.append(f"  追问话术: {templates[0]}")
                if reds:
                    lines.append(f"  警惕信号: {reds[0]}")

        # 技术参考
        if tech_items:
            lines.append(f"\n=== 技术参考（{len(tech_items)}条）===")
            for i, r in enumerate(tech_items):
                item = r.get("item", r) if isinstance(r, dict) else r
                score = r.get("rerank_score", r.get("score", 0))
                title = item.get("title", "")
                fab = item.get("fabrication_patterns", [])

                lines.append(f"\n[{i+1} · 相关度{score:.1f}] {title[:50]}")
                if fab:
                    lines.append(f"  编造识别: {fab[0]}")

        # 其他
        if other_items:
            lines.append(f"\n=== 其他参考（{len(other_items)}条）===")
            for i, r in enumerate(other_items):
                item = r.get("item", r) if isinstance(r, dict) else r
                title = item.get("title", item.get("question", ""))
                lines.append(f"\n[{i+1}] {title[:50]}")

        # 使用规则
        lines.append("\n【使用规则】")
        lines.append("1. 优先使用高相关度策略")
        lines.append("2. 结合候选人具体回答改写话术，不要直接抄")
        lines.append("3. 如果相关度都低于3分，请忽略RAG，按你的判断提问")
        lines.append("4. 每次追问选1-2个角度即可，不要堆砌")

        return "\n".join(lines)

    # ===== 完整的Advanced RAG检索流程 =====

    def retrieve_advanced(self, user_answer: str, section_label: str,
                          section_topic: str, jd_context: str,
                          question_type: str = "followup") -> List[Dict]:
        """Advanced RAG 完整检索流程

        Pre-Retrieval → Retrieval → Post-Retrieval
        """
        # Pre-Retrieval: 场景路由
        scenario_tags = self.route_scenario(section_label, question_type)

        # Pre-Retrieval: 多视角Query改写
        queries = self.multi_query_rewrite(
            user_answer, section_label, section_topic, jd_context
        )

        # Retrieval: 多路召回
        recall_k = self.get_dynamic_top_k(scenario_tags) * 3  # 召回更多
        recalled = self.multi_recall(queries, scenario_tags, recall_k=recall_k)

        if not recalled:
            return []

        # Post-Retrieval: LLM重排
        reranked = self.llm_rerank(recalled, user_answer, section_label, section_topic)

        # Post-Retrieval: 阈值过滤
        top_k = self.get_dynamic_top_k(scenario_tags)
        filtered = self.filter_by_threshold(reranked, threshold=2.5, max_results=top_k)

        return filtered

    # ===== 基础检索（向后兼容）=====

    def retrieve(self, query: str, top_k: int = 3, bm25_weight: float = 0.4,
                 embedding_weight: float = 0.6, categories: List[str] = None) -> List[Dict]:
        """基础混合检索（向后兼容旧调用）"""
        tokens = tokenize(query)

        if categories:
            filtered_indices = [i for i, item in enumerate(self.kb) if item.get("category") in categories]
            if not filtered_indices:
                return []
            filtered_kb = [self.kb[i] for i in filtered_indices]
            filtered_corpus = [self.corpus[i] for i in filtered_indices]
            bm25_filtered = BM25Okapi([tokenize(doc) for doc in filtered_corpus])
            bm25_scores = bm25_filtered.get_scores(tokens)
        else:
            filtered_indices = list(range(len(self.kb)))
            filtered_kb = self.kb
            bm25_scores = self.bm25.get_scores(tokens)

        try:
            where_clause = {"category": {"$in": categories}} if categories else None
            n_results = min(top_k * 3, len(filtered_kb))
            embedding_results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
            )
            emb_map = {}
            if embedding_results and embedding_results.get("ids"):
                for idx, doc_id in enumerate(embedding_results["ids"][0]):
                    distance = embedding_results["distances"][0][idx] if "distances" in embedding_results else 1.0
                    emb_map[doc_id] = 1.0 - distance
        except Exception:
            emb_map = {}

        max_bm25 = max(bm25_scores) if bm25_scores.size > 0 and max(bm25_scores) > 0 else 1.0
        max_emb = max(emb_map.values()) if emb_map else 1.0

        combined = {}
        for i, item in enumerate(filtered_kb):
            bm25_norm = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0
            emb_norm = emb_map.get(item["id"], 0) / max_emb if max_emb > 0 else 0
            score = bm25_weight * bm25_norm + embedding_weight * emb_norm
            combined[item["id"]] = {"item": item, "score": score, "bm25": bm25_norm, "emb": emb_norm}

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return ranked

    def get_by_id(self, item_id: str) -> Optional[Dict]:
        for item in self.kb:
            if item["id"] == item_id:
                return item
        return None

    def get_all(self) -> List[Dict]:
        return self.kb

    def reload(self):
        self.kb = load_knowledge_base()
        self.corpus = [item.get("embedding_text", item.get("question", "")) for item in self.kb]
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.corpus])
        try:
            self.chroma_client.delete_collection("mianba_kb")
        except Exception:
            pass
        self._init_chroma()
        self._build_index()


if __name__ == "__main__":
    retriever = HybridRetriever()

    # 测试基础检索
    print("=== 基础检索测试 ===")
    results = retriever.retrieve("实习期间负责了AI客服产品，准确率提升了", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['item']['id']}: {r['item'].get('title', r['item'].get('question', ''))[:50]}")

    # 测试Advanced RAG
    print("\n=== Advanced RAG 测试 ===")
    results = retriever.retrieve_advanced(
        user_answer="我在实习中负责了一个RAG系统的产品设计",
        section_label="B",
        section_topic="简历深挖",
        jd_context="AI产品经理校招",
        question_type="followup",
    )
    for r in results:
        item = r.get("item", r)
        score = r.get("rerank_score", r.get("score", 0))
        print(f"[{score:.3f}] {item['id']}: {item.get('title', item.get('question', ''))[:50]}")

    # 测试结构化注入
    print("\n=== 结构化注入测试 ===")
    print(retriever.format_rag_structured(results))
