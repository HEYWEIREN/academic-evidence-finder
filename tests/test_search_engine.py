import unittest

from app.search_engine import ENGINE, SearchConfig, expand_query
from app.evaluate import evaluate_all


class SearchEngineTest(unittest.TestCase):
    def test_hybrid_retrieval_returns_relevant_results(self):
        response = ENGINE.search("hybrid retrieval", SearchConfig(mode="hybrid", limit=5))
        ids = {result["id"] for result in response["results"]}
        self.assertTrue({"thakur-2021-beir", "robertson-2009-bm25", "karpukhin-2020-dpr"} & ids)

    def test_hybrid_retrieval_query_contains_core_papers(self):
        response = ENGINE.search("hybrid retrieval bm25 dense retrieval", SearchConfig(mode="hybrid", limit=5))
        ids = {result["id"] for result in response["results"]}
        self.assertTrue({"thakur-2021-beir", "robertson-2009-bm25"} <= ids)
        self.assertTrue({"karpukhin-2020-dpr", "xiong-2021-ance", "chen-2024-bge-m3"} & ids)

    def test_corpus_is_large_enough_for_demo(self):
        self.assertGreaterEqual(ENGINE.paper_count(), 65)

    def test_chinese_query_expansion(self):
        expanded = expand_query("RAG 如何缓解幻觉")
        self.assertIn("hallucination", expanded)
        response = ENGINE.search("RAG 如何缓解幻觉", SearchConfig(mode="hybrid", limit=5))
        ids = [result["id"] for result in response["results"]]
        self.assertIn("asai-2023-self-rag", ids)

    def test_filters_keep_topic_and_year(self):
        response = ENGINE.search(
            "evaluation",
            SearchConfig(mode="hybrid", year=2023, topic="Evaluation", limit=10),
        )
        self.assertGreaterEqual(response["count"], 1)
        for result in response["results"]:
            self.assertEqual(result["year"], 2023)
            self.assertIn("Evaluation", result["topics"])

    def test_paper_detail_includes_similar_papers(self):
        detail = ENGINE.get_paper("lewis-2020-rag")
        self.assertIsNotNone(detail)
        self.assertIn("similar", detail)
        self.assertGreater(len(detail["similar"]), 0)

    def test_search_results_explain_ranking(self):
        response = ENGINE.search("reranking transformer ranker", SearchConfig(mode="hybrid", limit=3))
        first = response["results"][0]
        self.assertIn("matched_phrases", first)
        self.assertIn("topic_match", first)
        self.assertIn("ranking_reasons", first)
        self.assertGreater(len(first["ranking_reasons"]), 0)

    def test_hybrid_metrics_are_not_worse_than_single_modes(self):
        metrics = evaluate_all()
        self.assertGreaterEqual(metrics["hybrid"]["precision_at_5"], metrics["bm25"]["precision_at_5"])
        self.assertGreaterEqual(metrics["hybrid"]["precision_at_5"], metrics["semantic"]["precision_at_5"])


if __name__ == "__main__":
    unittest.main()
