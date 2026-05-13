from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "papers.json"
DATA_DIR = DATA_PATH.parent

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "what",
    "when",
    "why",
}

CN_QUERY_MAP = {
    "检索增强": "retrieval augmented generation rag",
    "增强检索": "retrieval augmented generation rag",
    "知识增强": "retrieval augmented generation rag",
    "幻觉": "hallucination faithfulness factuality grounded evidence",
    "缓解": "reduce mitigate",
    "证据": "evidence passage context citation",
    "引用": "citation evidence source",
    "问答": "question answering open-domain qa",
    "语义检索": "semantic dense retrieval embedding vector search",
    "向量检索": "dense retrieval embedding vector search",
    "关键词": "keyword sparse retrieval bm25",
    "混合检索": "hybrid retrieval bm25 dense semantic",
    "重排序": "reranking ranker ranking cross-encoder",
    "查询改写": "query rewriting query expansion",
    "查询扩展": "query expansion pseudo document",
    "评价": "evaluation benchmark precision recall mrr faithfulness",
    "长上下文": "long context passage ordering lost in the middle",
    "可解释": "explainable interpretable evidence transparent",
}

FIELD_WEIGHTS = {
    "title": 3.0,
    "topics": 2.6,
    "abstract": 1.4,
    "chunks": 1.0,
}

PHRASE_ALIASES = {
    "hybrid retrieval": ["hybrid retrieval", "bm25 dense", "sparse dense"],
    "query expansion": ["query expansion", "document expansion", "query rewriting", "pseudo document"],
    "hallucination": ["hallucination", "faithfulness", "factuality", "grounded"],
    "reranking": ["reranking", "rerank", "ranker", "cross-encoder"],
    "long context": ["long context", "evidence ordering", "passage ordering", "lost in the middle"],
    "question answering": ["question answering", "open-domain", "open domain", "qa", "reader"],
    "semantic search": ["semantic search", "vector search", "dense retrieval", "embedding"],
    "evaluation": ["evaluation", "benchmark", "precision", "recall", "mrr", "faithfulness"],
    "rag": ["retrieval-augmented", "retrieval augmented", "rag"],
}

TOPIC_INTENTS = {
    "bm25": {"BM25", "Sparse Retrieval"},
    "sparse": {"BM25", "Sparse Retrieval"},
    "dense": {"Dense Retrieval", "Embedding", "Vector Search"},
    "semantic": {"Dense Retrieval", "Embedding", "Semantic Search", "Vector Search"},
    "vector": {"Dense Retrieval", "Embedding", "Vector Search"},
    "embedding": {"Dense Retrieval", "Embedding", "Vector Search"},
    "hallucination": {"Hallucination", "Verification", "Faithfulness", "RAG"},
    "faithfulness": {"Hallucination", "Verification", "Faithfulness", "Evaluation"},
    "query": {"Query Expansion", "Query Rewriting"},
    "expansion": {"Query Expansion", "Query Rewriting"},
    "rewriting": {"Query Expansion", "Query Rewriting"},
    "reranking": {"Reranking", "Ranking", "Transformer"},
    "ranker": {"Reranking", "Ranking", "Transformer"},
    "long": {"Long Context", "Evaluation"},
    "context": {"Long Context", "RAG", "Evaluation"},
    "evaluation": {"Evaluation", "Benchmark"},
    "benchmark": {"Evaluation", "Benchmark"},
    "question": {"Question Answering", "Reader"},
    "answering": {"Question Answering", "Reader"},
}


def tokenize(text: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+|[0-9]{4}", text.lower())
    normalized: list[str] = []
    for term in terms:
        if term in STOPWORDS:
            continue
        if term.endswith("ies") and len(term) > 4:
            term = term[:-3] + "y"
        elif term.endswith("s") and len(term) > 4:
            term = term[:-1]
        normalized.append(term)
    return normalized


def expand_query(query: str) -> str:
    additions = []
    for cn, en in CN_QUERY_MAP.items():
        if cn in query:
            additions.append(en)
    return " ".join([query, *additions]).strip()


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: (1.0 if value > 0 else 0.0) for key, value in scores.items()}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "hybrid"
    year: int | None = None
    topic: str | None = None
    limit: int = 12


class AcademicSearchEngine:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.papers = self._load_papers()
        self.paper_by_id = {paper["id"]: paper for paper in self.papers}
        self.field_texts = {paper["id"]: self._field_texts(paper) for paper in self.papers}
        self.field_tokens = {
            paper_id: {field: tokenize(text) for field, text in fields.items()}
            for paper_id, fields in self.field_texts.items()
        }
        self.documents = {paper["id"]: self._document_text(paper) for paper in self.papers}
        self.tokens = {paper_id: tokenize(text) for paper_id, text in self.documents.items()}
        self.term_freqs = {paper_id: Counter(tokens) for paper_id, tokens in self.tokens.items()}
        self.doc_lengths = {paper_id: len(tokens) for paper_id, tokens in self.tokens.items()}
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(1, len(self.doc_lengths))
        self.document_frequency = self._document_frequency()
        self.idf = self._idf()
        self.tfidf_vectors = self._tfidf_vectors()
        self.vector_norms = {
            paper_id: math.sqrt(sum(weight * weight for weight in vector.values()))
            for paper_id, vector in self.tfidf_vectors.items()
        }

    def _load_papers(self) -> list[dict[str, Any]]:
        papers = []
        for path in sorted(DATA_DIR.glob("*.json")):
            with path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
                if isinstance(loaded, list):
                    papers.extend(loaded)
        deduped = {paper["id"]: paper for paper in papers}
        papers = list(deduped.values())
        return sorted(papers, key=lambda paper: (paper["year"], paper["title"]), reverse=True)

    def _document_text(self, paper: dict[str, Any]) -> str:
        return " ".join(
            [
                paper["title"],
                paper["abstract"],
                " ".join(paper["topics"]),
                " ".join(paper["chunks"]),
                " ".join(paper["authors"]),
                paper["venue"],
            ]
        )

    def _field_texts(self, paper: dict[str, Any]) -> dict[str, str]:
        return {
            "title": paper["title"],
            "topics": " ".join(paper["topics"]),
            "abstract": paper["abstract"],
            "chunks": " ".join(paper["chunks"]),
        }

    def _document_frequency(self) -> dict[str, int]:
        document_frequency: dict[str, int] = defaultdict(int)
        for tokens in self.tokens.values():
            for token in set(tokens):
                document_frequency[token] += 1
        return dict(document_frequency)

    def _idf(self) -> dict[str, float]:
        total_docs = len(self.papers)
        return {
            term: math.log(1 + (total_docs - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.document_frequency.items()
        }

    def _tfidf_vectors(self) -> dict[str, dict[str, float]]:
        vectors: dict[str, dict[str, float]] = {}
        for paper_id, frequencies in self.term_freqs.items():
            max_tf = max(frequencies.values(), default=1)
            vectors[paper_id] = {
                term: (0.5 + 0.5 * count / max_tf) * self.idf.get(term, 0.0)
                for term, count in frequencies.items()
            }
        return vectors

    def topics(self) -> list[str]:
        return sorted({topic for paper in self.papers for topic in paper["topics"]})

    def years(self) -> list[int]:
        return sorted({paper["year"] for paper in self.papers}, reverse=True)

    def paper_count(self) -> int:
        return len(self.papers)

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        paper = self.paper_by_id.get(paper_id)
        if not paper:
            return None
        return {
            **paper,
            "similar": [
                self._summary_result(item, score)
                for item, score in self._similar_papers(paper_id, limit=4)
            ],
            "citation_snippets": paper["chunks"][:2],
        }

    def search(self, query: str, config: SearchConfig | None = None) -> dict[str, Any]:
        config = config or SearchConfig()
        expanded = expand_query(query)
        query_terms = tokenize(expanded)
        query_phrases = self._query_phrases(expanded)
        candidates = [
            paper
            for paper in self.papers
            if (config.year is None or paper["year"] == config.year)
            and (not config.topic or config.topic in paper["topics"])
        ]
        candidate_ids = {paper["id"] for paper in candidates}
        bm25 = {
            paper_id: score
            for paper_id, score in self._bm25_scores(query_terms).items()
            if paper_id in candidate_ids
        }
        semantic = {
            paper_id: score
            for paper_id, score in self._semantic_scores(query_terms).items()
            if paper_id in candidate_ids
        }
        field_scores = {
            paper_id: score
            for paper_id, score in self._field_scores(query_terms).items()
            if paper_id in candidate_ids
        }
        bm25_norm = normalize_scores(bm25)
        semantic_norm = normalize_scores(semantic)
        field_norm = normalize_scores(field_scores)

        results = []
        for paper in candidates:
            paper_id = paper["id"]
            lexical_score = bm25_norm.get(paper_id, 0.0)
            semantic_score = semantic_norm.get(paper_id, 0.0)
            field_score = field_norm.get(paper_id, 0.0)
            phrase_signal = self._phrase_signal(paper, query_phrases)
            topic_signal = self._topic_signal(paper, query_terms)
            evidence = self._best_evidence(paper, query_terms)
            evidence_quality = min(1.0, len(evidence["matched_terms"]) / max(1, min(6, len(set(query_terms)))))
            if config.mode == "bm25":
                final_score = lexical_score
            elif config.mode == "semantic":
                final_score = semantic_score
            else:
                final_score = (
                    0.50 * lexical_score
                    + 0.25 * semantic_score
                    + 0.15 * field_score
                    + phrase_signal["score"]
                    + topic_signal["score"]
                    + 0.03 * evidence_quality
                )
            if final_score <= 0 and query_terms:
                continue
            results.append(
                {
                    **self._summary_result(paper, final_score),
                    "_raw_score": final_score,
                    "_topic_set": set(paper["topics"]),
                    "bm25_score": round(lexical_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "field_score": round(field_score, 4),
                    "evidence": evidence,
                    "matched_phrases": phrase_signal["matched_phrases"],
                    "topic_match": topic_signal["matched_topics"],
                    "ranking_reasons": self._ranking_reasons(
                        lexical_score,
                        semantic_score,
                        field_score,
                        phrase_signal,
                        topic_signal,
                        evidence_quality,
                    ),
                }
            )

        results.sort(key=lambda item: (item["_raw_score"], item["year"]), reverse=True)
        total_count = len(results)
        if config.mode == "hybrid":
            results = self._mmr_rerank(results, limit=config.limit)
        else:
            results = results[: config.limit]
        self._normalize_display_scores(results)
        return {
            "query": query,
            "expanded_query": expanded,
            "mode": config.mode,
            "count": total_count,
            "results": [self._public_result(result) for result in results],
            "facets": {"topics": self.topics(), "years": self.years(), "paper_count": self.paper_count()},
        }

    def _summary_result(self, paper: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "venue": paper["venue"],
            "topics": paper["topics"],
            "url": paper["url"],
            "abstract": paper["abstract"],
            "score": round(score, 4),
        }

    def _bm25_scores(self, query_terms: list[str]) -> dict[str, float]:
        k1 = 1.5
        b = 0.75
        scores: dict[str, float] = {}
        for paper_id, frequencies in self.term_freqs.items():
            score = 0.0
            doc_length = self.doc_lengths.get(paper_id, 0)
            for term in query_terms:
                freq = frequencies.get(term, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                score += idf * (freq * (k1 + 1)) / denom
            scores[paper_id] = score
        return scores

    def _semantic_scores(self, query_terms: list[str]) -> dict[str, float]:
        query_counts = Counter(query_terms)
        if not query_counts:
            return {paper["id"]: 0.0 for paper in self.papers}
        max_tf = max(query_counts.values(), default=1)
        query_vector = {
            term: (0.5 + 0.5 * count / max_tf) * self.idf.get(term, 0.0)
            for term, count in query_counts.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        if query_norm == 0:
            return {paper["id"]: 0.0 for paper in self.papers}
        scores: dict[str, float] = {}
        for paper_id, vector in self.tfidf_vectors.items():
            dot = sum(query_vector.get(term, 0.0) * weight for term, weight in vector.items())
            denom = query_norm * self.vector_norms.get(paper_id, 1.0)
            scores[paper_id] = dot / denom if denom else 0.0
        return scores

    def _field_scores(self, query_terms: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        unique_terms = set(query_terms)
        for paper_id, fields in self.field_tokens.items():
            score = 0.0
            for field, tokens in fields.items():
                token_counts = Counter(tokens)
                field_weight = FIELD_WEIGHTS[field]
                for term in unique_terms:
                    if term in token_counts:
                        score += field_weight * self.idf.get(term, 0.0) * (1 + math.log(token_counts[term]))
            scores[paper_id] = score
        return scores

    def _query_phrases(self, expanded_query: str) -> list[str]:
        query_text = expanded_query.lower()
        query_terms = set(tokenize(expanded_query))
        phrases = []
        for phrase, aliases in PHRASE_ALIASES.items():
            if any(alias in query_text for alias in aliases):
                phrases.append(phrase)
                continue
            phrase_terms = set(tokenize(phrase))
            if phrase_terms and phrase_terms.issubset(query_terms):
                phrases.append(phrase)
        return phrases

    def _phrase_signal(self, paper: dict[str, Any], query_phrases: list[str]) -> dict[str, Any]:
        paper_id = paper["id"]
        fields = {field: text.lower() for field, text in self.field_texts[paper_id].items()}
        score = 0.0
        matched: list[str] = []
        reasons: list[str] = []
        for phrase in query_phrases:
            aliases = PHRASE_ALIASES[phrase]
            if any(alias in fields["title"] for alias in aliases):
                score += 0.18
                matched.append(phrase)
                reasons.append(f"title phrase: {phrase}")
            elif any(alias in fields["topics"] for alias in aliases):
                score += 0.15
                matched.append(phrase)
                reasons.append(f"topic phrase: {phrase}")
            elif any(alias in fields["abstract"] for alias in aliases):
                score += 0.09
                matched.append(phrase)
                reasons.append(f"abstract phrase: {phrase}")
            elif any(alias in fields["chunks"] for alias in aliases):
                score += 0.06
                matched.append(phrase)
                reasons.append(f"evidence phrase: {phrase}")
        return {"score": min(score, 0.32), "matched_phrases": sorted(set(matched)), "reasons": reasons}

    def _topic_signal(self, paper: dict[str, Any], query_terms: list[str]) -> dict[str, Any]:
        desired_topics: set[str] = set()
        for term in set(query_terms):
            desired_topics.update(TOPIC_INTENTS.get(term, set()))
        paper_topics = set(paper["topics"])
        matched_topics = sorted(paper_topics & desired_topics)
        score = min(0.26, 0.075 * len(matched_topics))
        if {"BM25", "Dense Retrieval"} & paper_topics and {"bm25", "dense"} <= set(query_terms):
            score += 0.08
        return {"score": min(score, 0.30), "matched_topics": matched_topics}

    def _ranking_reasons(
        self,
        lexical_score: float,
        semantic_score: float,
        field_score: float,
        phrase_signal: dict[str, Any],
        topic_signal: dict[str, Any],
        evidence_quality: float,
    ) -> list[str]:
        reasons: list[str] = []
        if lexical_score >= 0.45:
            reasons.append("strong BM25 term match")
        if semantic_score >= 0.45:
            reasons.append("high TF-IDF semantic similarity")
        if field_score >= 0.45:
            reasons.append("query terms appear in title/topics")
        if topic_signal["matched_topics"]:
            reasons.append("topic match: " + ", ".join(topic_signal["matched_topics"][:3]))
        if phrase_signal["matched_phrases"]:
            reasons.append("phrase match: " + ", ".join(phrase_signal["matched_phrases"][:2]))
        if evidence_quality >= 0.35:
            reasons.append("evidence chunk covers query terms")
        return reasons[:4] or ["baseline lexical/semantic similarity"]

    def _mmr_rerank(self, results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        pool = results[: max(limit * 4, 20)]
        selected: list[dict[str, Any]] = []
        remaining = pool[:]
        max_raw = max((result["_raw_score"] for result in remaining), default=1.0)
        while remaining and len(selected) < limit:
            if not selected:
                best = max(remaining, key=lambda item: (item["_raw_score"], item["year"]))
            else:
                best = max(
                    remaining,
                    key=lambda item: (
                        item["_raw_score"] / max_raw
                        - 0.10 * max(self._topic_similarity(item, chosen) for chosen in selected)
                        - 0.05 * max(self._phrase_similarity(item, chosen) for chosen in selected),
                        item["year"],
                    ),
                )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _topic_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_topics = left["_topic_set"]
        right_topics = right["_topic_set"]
        if not left_topics or not right_topics:
            return 0.0
        return len(left_topics & right_topics) / len(left_topics | right_topics)

    def _phrase_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_phrases = set(left["matched_phrases"])
        right_phrases = set(right["matched_phrases"])
        if not left_phrases or not right_phrases:
            return 0.0
        return len(left_phrases & right_phrases) / len(left_phrases | right_phrases)

    def _normalize_display_scores(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        raw_scores = [result["_raw_score"] for result in results]
        high = max(raw_scores)
        for result in results:
            if math.isclose(high, 0.0):
                result["score"] = 1.0 if result["_raw_score"] > 0 else 0.0
            else:
                result["score"] = round(result["_raw_score"] / high, 4)

    def _public_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if not key.startswith("_")}

    def _best_evidence(self, paper: dict[str, Any], query_terms: list[str]) -> dict[str, Any]:
        chunks = [paper["abstract"], *paper["chunks"]]
        scored = []
        for chunk in chunks:
            chunk_terms = tokenize(chunk)
            overlap = sum(1 for term in query_terms if term in chunk_terms)
            semantic_overlap = len(set(query_terms) & set(chunk_terms))
            scored.append((overlap + 0.25 * semantic_overlap, chunk))
        _, best_chunk = max(scored, key=lambda item: item[0])
        return {
            "text": best_chunk,
            "highlighted": highlight_terms(best_chunk, query_terms),
            "matched_terms": sorted(set(query_terms) & set(tokenize(best_chunk))),
        }

    def _similar_papers(self, paper_id: str, limit: int = 4) -> list[tuple[dict[str, Any], float]]:
        source_vector = self.tfidf_vectors[paper_id]
        source_norm = self.vector_norms[paper_id]
        scored = []
        for other in self.papers:
            other_id = other["id"]
            if other_id == paper_id:
                continue
            vector = self.tfidf_vectors[other_id]
            dot = sum(source_vector.get(term, 0.0) * weight for term, weight in vector.items())
            denom = source_norm * self.vector_norms.get(other_id, 1.0)
            score = dot / denom if denom else 0.0
            scored.append((other, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


def highlight_terms(text: str, terms: list[str]) -> str:
    highlightable = [re.escape(term) for term in sorted(set(terms), key=len, reverse=True) if len(term) > 2]
    if not highlightable:
        return text
    pattern = re.compile(r"\b(" + "|".join(highlightable) + r")\b", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)


ENGINE = AcademicSearchEngine()
