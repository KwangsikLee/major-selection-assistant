#!/usr/bin/env python3
"""Build KRAG retriever evaluation dataset from temp_texts documents.

For each department under temp_texts, try to extract up to 4 QA pairs based on:
- 교육과정 (curriculum)
- 진로관련 (career)
- 학과의 특징 (features)
- 교육목표 (goals)

Outputs a CSV: eval/krag_eval_set.csv with columns: question, ground_truth, contexts
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TEMP_TEXTS = ROOT / "temp_texts"
OUT_CSV = ROOT / "eval/krag_eval_set.csv"


def split_sentences(text: str) -> List[str]:
    # Simple Korean-friendly splitting on newlines and sentence enders
    # Avoid heavy deps; keep deterministic
    text = text.replace("\r", "\n")
    parts = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        # Split on common Korean sentence enders (avoid lookbehind)
        segs = re.split(r"(?:[\.\?\!]|다\.|요\.|임\.|임\)|음\.)\s+", para)
        for s in segs:
            s = s.strip()
            if s:
                parts.append(s)
    return parts or [text]


def find_snippet(text: str, keywords: List[str], max_len: int = 400) -> Optional[str]:
    # Return the first sentence containing any of the keywords
    for sent in split_sentences(text):
        low = sent.lower()
        if any(kw.lower() in low for kw in keywords):
            return sent[:max_len]
    # fallback: line containing keyword
    for line in text.split("\n"):
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            line = line.strip()
            return (line[:max_len]) if line else None
    return None


Category = Dict[str, object]


CATEGORIES: Dict[str, Category] = {
    "교육과정": {
        "question": "이 학과의 교육과정(커리큘럼)은 어떻게 구성되어 있나요?",
        "keywords": [
            "교육과정",
            "커리큘럼",
            "과목",
            "학년",
            "전공",
            "필수",
            "선택",
            "캡스톤",
        ],
    },
    "진로관련": {
        "question": "졸업 후 진로와 관련 분야는 무엇인가요?",
        "keywords": [
            "진로",
            "졸업 후",
            "취업",
            "관련 자격",
            "자격증",
            "진출",
            "분야",
        ],
    },
    "학과의 특징": {
        "question": "이 학과의 주요 특징이나 강점은 무엇인가요?",
        "keywords": [
            "특징",
            "장점",
            "특성",
            "소개",
            "교육 자원",
            "우수",
            "강점",
        ],
    },
    "교육목표": {
        "question": "이 학과의 교육 목표는 무엇인가요?",
        "keywords": [
            "교육 목표",
            "목표",
            "지향",
            "인재",
            "양성",
            "비전",
            "도전",
        ],
    },
}


def derive_major_name(dir_name: str) -> str:
    # Use the trailing token as a rough major name
    major = dir_name
    if "." in dir_name:
        major = dir_name.split(".")[-1]
    elif "-" in dir_name:
        major = dir_name.split("-")[-1]
    return major.strip()


def iter_department_jsons(root: Path) -> Iterable[Tuple[Path, Path]]:
    for dept_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        docs_dir = dept_dir / "documents"
        if not docs_dir.exists():
            continue
        json_files = sorted(docs_dir.glob("*_documents.json"))
        if not json_files:
            continue
        yield dept_dir, json_files[0]


def personalize_question(major: str, base_question: str) -> str:
    # Try to make grammar natural by replacing placeholders
    q = base_question
    q = q.replace("이 학과의", f"{major}의")
    q = q.replace("이 학과는", f"{major}는")
    q = q.replace("이 학과", f"{major}")
    if not q.startswith(major):
        q = f"{major} {q}"
    return q


def build_rows_for_department(dept_dir: Path, json_path: Path, max_chunks: int = 12) -> List[dict]:
    rows: List[dict] = []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return rows

    sample_chunks = data[:max_chunks]
    covered = set()
    major = derive_major_name(dept_dir.name)

    # Greedy pick chunks per category
    for cat_name, spec in CATEGORIES.items():
        q = personalize_question(major, spec["question"])  # type: ignore[index]
        kws = spec["keywords"]  # type: ignore[index]
        found = None
        for chunk in sample_chunks:
            text = chunk.get("page_content", "")
            snippet = find_snippet(text, kws) if text else None
            if snippet:
                found = (text, snippet)
                break
        if found:
            ctxt, answer = found
            rows.append(
                {
                    "question": q,
                    "ground_truth": answer,
                    "contexts": json.dumps([ctxt], ensure_ascii=False),
                }
            )
            covered.add(cat_name)

    # Ensure at least one row exists; fallback to name question
    if not rows and sample_chunks:
        ctxt = sample_chunks[0].get("page_content", "")
        rows.append(
            {
                "question": f"{major} 안내 자료의 학과명은 무엇인가요? 학과명만 적으세요.",
                "ground_truth": major,
                "contexts": json.dumps([ctxt], ensure_ascii=False),
            }
        )

    return rows


def main() -> None:
    departments = list(iter_department_jsons(TEMP_TEXTS))
    if not departments:
        raise SystemExit("No departments with documents found under temp_texts")

    random.seed(42)
    sampled = random.sample(departments, k=min(20, len(departments)))

    all_rows: List[dict] = []
    for dept_dir, json_path in sampled:
        rows = build_rows_for_department(dept_dir, json_path)
        all_rows.extend(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
