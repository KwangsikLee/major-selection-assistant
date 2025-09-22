#!/usr/bin/env python3
"""KRAG retriever evaluation script."""

import argparse
import json
import os
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import StringIO
import math
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from krag.evaluators import RougeOfflineRetrievalEvaluators  # type: ignore
    from krag.document import KragDocument  # type: ignore
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)


def make_krag_document(content: str, doc_id: str | None = None, metadata: dict | None = None) -> KragDocument:
    """KRAG 문서 객체 생성 헬퍼."""
    meta: dict = {}
    if metadata:
        meta.update(metadata)
    if doc_id is not None:
        meta.setdefault("doc_id", doc_id)

    return KragDocument(page_content=content, metadata=meta)

import pandas as pd
from dotenv import load_dotenv


# Ensure shared KRAG modules are importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from a_my_rag_module.embedding import VectorStoreManager  # noqa: E402
from a_my_rag_module.evaluator import CustomEvaluatior  # noqa: E402
from a_my_rag_module.retriever import (  # noqa: E402
    AdvancedHybridRetriever,
    DEFAULT_RERANKER_MODEL,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate KRAG retriever performance.")
    parser.add_argument(
        "--dataset",
        default=str(project_root / "eval/krag_eval_set.csv"),
        help="Path to evaluation dataset CSV (question, ground_truth, contexts).",
    )
    parser.add_argument(
        "--vector-db",
        default=str(project_root / "vector_db"),
        help="Directory containing persisted FAISS indexes.",
    )
    parser.add_argument(
        "--index-name",
        default="college_guide",
        help="Vector store index name to load.",
    )
    parser.add_argument(
        "--model",
        default="embedding-gemma",
        help="Embedding model key registered in VectorStoreManager.",
    )
    parser.add_argument(
        "--method",
        choices=["similarity", "keyword", "hybrid", "ensemble", "advanced"],
        default="ensemble",
        help="Retriever search strategy to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of contexts to retrieve per question.",
    )
    parser.add_argument(
        "--reranker",
        default=DEFAULT_RERANKER_MODEL,
        help="Reranker key (use 'none' to disable).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking during retrieval.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for counting a context as relevant.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save per-question evaluation results as CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print retriever debug output (advanced search logs).",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_vector_store(args: argparse.Namespace) -> VectorStoreManager:
    hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""
    manager = VectorStoreManager(
        embedding_model_key=args.model,
        save_directory=args.vector_db,
        hf_api_token=hf_token,
    )

    if not manager.index_exists(args.index_name, args.model):
        raise FileNotFoundError(
            f"Vector store '{args.index_name}_{args.model}' not found under {args.vector_db}."
        )

    ok, message = manager.load_vector_store(args.index_name, args.model)
    if not ok:
        raise RuntimeError(message)
    print(message)

    if manager.embeddings is None:
        manager.lazy_init_embedding(manager.current_model_key)

    return manager


def build_retriever(manager: VectorStoreManager, args: argparse.Namespace) -> AdvancedHybridRetriever:
    vector_store = manager.current_vector_store
    if vector_store is None:
        raise RuntimeError("Vector store failed to load.")
    return AdvancedHybridRetriever(
        documents=None,
        vector_store=vector_store,
        reranker_model=args.reranker,
    )


def retrieve_documents(
    retriever: AdvancedHybridRetriever,
    question: str,
    args: argparse.Namespace,
) -> List:
    if args.method == "similarity":
        return retriever.search_by_similarity(question, k=args.top_k, use_rerank=not args.no_rerank)
    if args.method == "keyword":
        return retriever.search_by_keyword(question, k=args.top_k, use_rerank=not args.no_rerank)
    if args.method == "hybrid":
        return retriever.hybrid_search(question, k=args.top_k, use_rerank=not args.no_rerank)

    # advanced_search prints retrieval traces; silence unless verbose is requested
    target_method = "ensemble" if args.method == "ensemble" else "hybrid"

    print(f"retrieve docs : {target_method}, k:{args.top_k}, use_rerank:{not args.no_rerank}")

    if args.verbose:
        return retriever.advanced_search(
            question,
            method=target_method,
            k=args.top_k,
            use_rerank=not args.no_rerank,
        )

    with redirect_stdout(StringIO()):
        return retriever.advanced_search(
            question,
            method=target_method,
            k=args.top_k,
            use_rerank=not args.no_rerank,
        )


def parse_contexts(raw_contexts: str) -> List[str]:
    try:
        return json.loads(raw_contexts)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid contexts JSON: {exc}")


def evaluate_dataset(
    df: pd.DataFrame,
    retriever: AdvancedHybridRetriever,
    evaluator: CustomEvaluatior,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, List[List[KragDocument]], List[List[KragDocument]]]:
    rows = []
    actual_sets: List[List[KragDocument]] = []
    predicted_sets: List[List[KragDocument]] = []

    for idx, row in df.iterrows():
        question = row["question"]
        ground_truth = row.get("ground_truth", "")
        contexts = parse_contexts(row["contexts"])

        retrieved = retrieve_documents(retriever, question, args)
        retrieved_texts = [doc.page_content for doc in retrieved]
        context_norms = [normalize_text(ctx) for ctx in contexts]

        best_rank = None
        best_similarity = 0.0
        relevant_count = 0

        for rank, doc in enumerate(retrieved_texts):
            doc_norm = normalize_text(doc)
            doc_best = 0.0
            hit = False
            for ctx_norm in context_norms:
                score = SequenceMatcher(None, doc_norm, ctx_norm).ratio()
                doc_best = max(doc_best, score)
                if score >= args.threshold:
                    hit = True
                    break
            best_similarity = max(best_similarity, doc_best)
            if hit:
                relevant_count += 1
                if best_rank is None:
                    best_rank = rank

        precision = relevant_count / max(len(retrieved_texts), 1)
        recall = relevant_count / max(len(contexts), 1)
        hit = 1 if best_rank is not None else 0
        reciprocal_rank = 1.0 / (best_rank + 1) if best_rank is not None else 0.0
        context_precision = (
            evaluator.calculate_context_precision(question, retrieved_texts)
            if retrieved_texts
            else 0.0
        )

        actual_sets.append([
            make_krag_document(content=ctx, doc_id=f"q{idx}_gold_{i}")
            for i, ctx in enumerate(contexts)
        ])
        predicted_sets.append([
            make_krag_document(
                content=doc.page_content,
                doc_id=f"q{idx}_pred_{i}",
                metadata=getattr(doc, "metadata", None),
            )
            for i, doc in enumerate(retrieved)
        ])

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "hit": hit,
                "reciprocal_rank": reciprocal_rank,
                "precision@k": precision,
                "recall@k": recall,
                "best_similarity": best_similarity,
                "context_precision": context_precision,
                "retrieved_sources": [
                    f"{doc.metadata.get('source', 'unknown')}#p{doc.metadata.get('page', '?')}"
                    for doc in retrieved
                ],
            }
        )

    return pd.DataFrame(rows), actual_sets, predicted_sets


def compute_rouge_metrics_dataframe(
    actual_sets: Sequence[Sequence[KragDocument]],
    predicted_sets: Sequence[Sequence[KragDocument]],
    match_method: str = "rouge1",
    threshold: float = 0.5,
    k: int | None = None,
) -> tuple[RougeOfflineRetrievalEvaluators, pd.DataFrame]:
    """ROUGE 매칭 기반 검색 평가지표 계산."""

    evaluator = RougeOfflineRetrievalEvaluators(
        actual_sets,
        predicted_sets,
        match_method=match_method,
        threshold=threshold,
    )

    metrics = {
        "hit_rate": evaluator.calculate_hit_rate(k),
        "mrr": evaluator.calculate_mrr(k),
        "recall": evaluator.calculate_recall(k),
        "precision": evaluator.calculate_precision(k),
        "map": evaluator.calculate_map(k),
        "ndcg": evaluator.calculate_ndcg(k),
    }

    metrics_df = pd.DataFrame([metrics])
    return evaluator, metrics_df


def summarize(results: pd.DataFrame) -> None:
    if results.empty:
        print("No evaluation samples were processed.")
        return

    metrics = {
        "samples": len(results),
        "hit_rate@k": results["hit"].mean(),
        "mrr@k": results["reciprocal_rank"].mean(),
        "precision@k": results["precision@k"].mean(),
        "recall@k": results["recall@k"].mean(),
        "avg_context_precision": results["context_precision"].mean(),
        "avg_best_similarity": results["best_similarity"].mean(),
    }

    print("\n=== Retriever Evaluation Summary ===")
    for key, value in metrics.items():
        if key == "samples":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    


def main() -> None:
    args = parse_args()
    load_dotenv()

    # dataset_path = Path(args.dataset)
    dataset_path = Path("./eval/krag_eval_set_ex.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    manager = load_vector_store(args)
    evaluator = CustomEvaluatior(llm=None)
    evaluator.embeddings = manager.embeddings
    retriever = build_retriever(manager, args)

    description = """Hit Rate는 0~1 사이의 값으로 검색된 문서 중 실제 관련 있는 문서의 비율을 측정하며, 순서를 고려하지 않은 기본적인 평가 지표
    Mean Reciprocal Rank (MRR) 은 첫 번째 관련 문서가 등장하는 순위의 역수를 평균 내어 계산하며, 검색 결과의 순서를 고려한 평가가 가능
    Mean Average Precision (mAP@k) 는 상위 k개 문서 내에서 관련 문서 검색의 정확도를 평균화하여 산출
    NDCG@k는 문서의 관련성과 검색 순위를 동시에 고려하여 이상적인 순위와 비교한 정규화 점수를 제공하는 종합적인 평가 지표"""

    print(description)

    results, actual_sets, predicted_sets = evaluate_dataset(df, retriever, evaluator, args)
    summarize(results)

    rouge_evaluator, rouge_metrics = compute_rouge_metrics_dataframe(
        actual_sets,
        predicted_sets,
        match_method="rouge1",
        threshold=0.5,
        k=5,
    )

    print("\n=== ROUGE 기반 Retrieval Metrics (k=5) ===")
    for metric_name, value in rouge_metrics.iloc[0].items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            print(f"{metric_name}: {value}")
        else:
            print(f"{metric_name}: {numeric_value:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
