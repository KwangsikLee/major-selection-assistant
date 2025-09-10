
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CustomEvaluatior:    
    def __init__(self, llm):
        self.llm = llm

    def evaluate_response(self, query, response, ground_truth=None):
        """
        Evaluates the quality of a generated response based on the query and optionally ground truth.
        """
        prompt = f"""
        You are an AI assistant tasked with evaluating the quality of responses.
        Please assess the following response based on the given query and ground truth (if provided).

        Query: {query}
        Response: {response}
        """
        if ground_truth:
            prompt += f"Ground Truth: {ground_truth}\n"

        prompt += """
        Provide a score from 1 to 5 for the following criteria:
        1. Relevance (how well the response addresses the query)
        2. Accuracy (how factually correct the response is, especially compared to ground truth)
        3. Completeness (does the response provide all necessary information)
        4. Conciseness (is the response to the point without unnecessary verbosity)
        5. Coherence (is the response well-structured and easy to understand)

        Also, provide a brief explanation for each score and an overall assessment.

        Format your output as follows:
        Relevance: [Score]/5 - [Explanation]
        Accuracy: [Score]/5 - [Explanation]
        Completeness: [Score]/5 - [Explanation]
        Conciseness: [Score]/5 - [Explanation]
        Coherence: [Score]/5 - [Explanation]

        Overall Assessment: [Your overall assessment and any suggestions for improvement]
        """
        evaluation_result = self.llm.invoke(prompt)
        return evaluation_result.content

    def evaluate_retrieval(self, query, retrieved_documents):
        """
        Evaluates the relevance of retrieved documents to a given query.
        """
        if not retrieved_documents:
            return "No documents retrieved for evaluation."

        prompt = f"""
        You are an AI assistant tasked with evaluating the relevance of retrieved documents.
        Please assess the following documents based on the given query.

        Query: {query}

        Retrieved Documents:
        """
        for i, doc in enumerate(retrieved_documents):
            prompt += f"\n--- Document {i+1} ---\n{doc}\n"

        prompt += """
        For each document, provide a relevance score from 1 to 5 and a brief explanation.
        Format your output as follows:
        Document 1: [Score]/5 - [Explanation]
        Document 2: [Score]/5 - [Explanation]
        ...

        Also, provide an overall assessment of the retrieval quality and suggestions for improvement.
        """

        evaluation_result = self.llm.invoke(prompt)
        return evaluation_result.content
    
    def calculate_answer_similarity(self, answer: str, ground_truth: str) -> float:
        """답변 유사도 계산"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio()

    def calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """질문-답변 관련성 계산"""
        q_emb = self.embeddings.embed_query(question)
        a_emb = self.embeddings.embed_query(answer)
        similarity = cosine_similarity([q_emb], [a_emb])[0][0]
        return similarity

    def calculate_context_precision(self, question: str, contexts: List[str]) -> float:
        """컨텍스트 정확도 계산"""
        if not contexts:
            return 0.0

        q_emb = self.embeddings.embed_query(question)
        context_scores = []

        for context in contexts:
            c_emb = self.embeddings.embed_query(context)
            score = cosine_similarity([q_emb], [c_emb])[0][0]
            context_scores.append(score)

        return np.mean(context_scores)

    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """답변이 컨텍스트에 충실한지 평가"""
        if not contexts:
            return 0.0

        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words) / len(answer_words)
        return min(overlap * 1.5, 1.0)  # 스케일 조정

    def evaluate_response_all(self, eval_df: pd.DataFrame) -> Dict[str, float]:
        """모든 메트릭 계산"""
        metrics = defaultdict(list)

        for _, row in eval_df.iterrows():
            metrics['answer_similarity'].append(
                self.calculate_answer_similarity(row['answer'], row['ground_truth'])
            )
            metrics['answer_relevancy'].append(
                self.calculate_answer_relevancy(row['question'], row['answer'])
            )
            metrics['context_precision'].append(
                self.calculate_context_precision(row['question'], row['contexts'])
            )
            metrics['faithfulness'].append(
                self.calculate_faithfulness(row['answer'], row['contexts'])
            )

        ret = {
            metric: np.mean(scores)
            for metric, scores in metrics.items()
        }
        print("\n===== 평가 결과 =====")
        for metric, score in ret.items():
            print(f"{metric}: {score:.3f}")

        # 평균 계산
        return ret