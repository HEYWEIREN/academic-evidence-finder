from __future__ import annotations

from dataclasses import dataclass

try:
    from .search_engine import ENGINE, SearchConfig
except ImportError:  # pragma: no cover
    from search_engine import ENGINE, SearchConfig


@dataclass(frozen=True)
class QueryCase:
    query: str
    relevant: set[str]


QUERY_CASES = [
    QueryCase(
        "hybrid retrieval bm25 dense retrieval",
        {
            "lin-2021-pyserini",
            "thakur-2021-beir",
            "robertson-2009-bm25",
            "karpukhin-2020-dpr",
            "chen-2024-bge-m3",
            "xiong-2021-ance",
        },
    ),
    QueryCase(
        "RAG 如何缓解幻觉",
        {
            "asai-2023-self-rag",
            "shuster-2021-rag-hallucination",
            "mallen-2023-knowledge-conflicts",
            "yan-2024-crag",
            "zhao-2024-rag-survey",
        },
    ),
    QueryCase(
        "query expansion for sparse retrieval",
        {
            "ma-2023-query2doc",
            "nogueira-2019-doc2query",
            "gao-2022-hyde",
            "formal-2021-splade",
            "chen-2024-rq-rag",
            "ma-2024-corpus-steering",
        },
    ),
    QueryCase(
        "reranking transformer ranker",
        {
            "nogueira-2020-monot5",
            "nogueira-2019-bert-reranking",
            "lin-2021-pretrained-transformers-ir",
            "li-2023-prp",
            "burges-2005-ranknet",
        },
    ),
    QueryCase(
        "long context evidence ordering",
        {
            "liu-2023-lost-middle",
            "sarthi-2024-raptor",
            "ram-2023-in-context-rag",
            "zhao-2024-rag-survey",
            "edge-2024-graphrag",
        },
    ),
    QueryCase(
        "semantic vector search embedding",
        {
            "reimers-2019-sbert",
            "johnson-2019-faiss",
            "karpukhin-2020-dpr",
            "wang-2022-e5",
            "li-2023-bge",
            "chen-2024-bge-m3",
            "izacard-2021-contriever",
            "ni-2021-gtr",
        },
    ),
    QueryCase(
        "open domain question answering retriever reader",
        {
            "chen-2017-drqa",
            "karpukhin-2020-dpr",
            "izacard-2021-fid",
            "kwiatkowski-2019-nq",
            "joshi-2017-triviaqa",
            "yang-2018-hotpotqa",
        },
    ),
    QueryCase(
        "RAG evaluation faithfulness context relevance",
        {
            "saad-falcon-2023",
            "es-2023-ragas",
            "zhao-2024-rag-survey",
            "thakur-2021-beir",
            "chen-2024-blink",
        },
    ),
    QueryCase("graph rag summarization", {"edge-2024-graphrag", "sarthi-2024-raptor", "zhao-2024-rag-survey"}),
    QueryCase("SPLADE sparse retrieval", {"formal-2021-splade", "lin-2021-unicoil", "dai-2019-deepct", "robertson-2009-bm25"}),
]


def evaluate_mode(mode: str) -> dict[str, float]:
    precision_at_5 = []
    recall_at_10 = []
    reciprocal_ranks = []
    for case in QUERY_CASES:
        response = ENGINE.search(case.query, SearchConfig(mode=mode, limit=10))
        ranked_ids = [result["id"] for result in response["results"]]
        top_5 = ranked_ids[:5]
        hits_5 = len(set(top_5) & case.relevant)
        hits_10 = len(set(ranked_ids[:10]) & case.relevant)
        precision_at_5.append(hits_5 / 5)
        recall_at_10.append(hits_10 / len(case.relevant))
        reciprocal_rank = 0.0
        for index, paper_id in enumerate(ranked_ids, start=1):
            if paper_id in case.relevant:
                reciprocal_rank = 1 / index
                break
        reciprocal_ranks.append(reciprocal_rank)
    return {
        "precision_at_5": round(sum(precision_at_5) / len(precision_at_5), 4),
        "recall_at_10": round(sum(recall_at_10) / len(recall_at_10), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
    }


def evaluate_all() -> dict[str, dict[str, float]]:
    return {mode: evaluate_mode(mode) for mode in ["bm25", "semantic", "hybrid"]}


if __name__ == "__main__":
    results = evaluate_all()
    for mode, metrics in results.items():
        print(
            f"{mode:8s}  P@5={metrics['precision_at_5']:.4f}  "
            f"R@10={metrics['recall_at_10']:.4f}  MRR={metrics['mrr']:.4f}"
        )
