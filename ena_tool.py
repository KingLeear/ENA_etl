# ena_tool.py
from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def detect_lang(text: str) -> str:
    zh_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en_chars = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def split_zh_sentences(
    text: Optional[str],
    min_len: int = 8,
) -> List[str]:
    if text is None:
        return []
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return []

    # Chinese sentence splitting
    sents = re.split(r"[。！？；…]\s*|\n+", text)
    sents = [s.strip() for s in sents if s and s.strip()]

    # adjustable, filter by min length
    if min_len and min_len > 0:
        sents = [s for s in sents if len(s) >= min_len]

    return sents


def split_en_sentences(text: Optional[str], min_len: int = 1) -> List[str]:
    # this is now the default simple English sentence splitter
    if text is None:
        return []
    t = str(text).strip()
    if not t or t.lower() == "nan":
        return []
    t = re.sub(r"\s+", " ", t)

    abbr = r"(Mr|Ms|Mrs|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\."
    t = re.sub(abbr, lambda m: m.group(0).replace(".", "<DOT>"), t, flags=re.IGNORECASE)

    parts = re.split(r"[.!?;]\s*", t)
    sents = [p.replace("<DOT>", ".").strip() for p in parts if p and p.strip()]

    if min_len and min_len > 0:
        sents = [s for s in sents if len(s) >= min_len]
    return sents

def segment_csv(
    in_csv: Path,
    out_csv: Path,
    text_col: str,
    id_cols: List[str],
    group_col: Optional[str],
    lang: str,
    min_len_zh: int,
    min_len_en: int,
) -> None:
    df = pd.read_csv(in_csv)

    # 1) handle duplicated columns
    if df.columns.duplicated().any():
        counts = {}
        new_cols = []
        for c in df.columns:
            if c not in counts:
                counts[c] = 0
                new_cols.append(c)
            else:
                counts[c] += 1
                new_cols.append(f"{c}__dup{counts[c]}")
        df.columns = new_cols

    # 2) use row_id as unique row identifier
    if "__row_id" in df.columns:

        df = df.rename(columns={"__row_id": "__row_id__orig"})
    df = df.reset_index(drop=False).rename(columns={"index": "__row_id"})
    row_id_col = "__row_id"

    # 3) check the required columns
    required_cols = [text_col, *id_cols]
    if group_col:
        required_cols.append(group_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}. Existing columns: {list(df.columns)}"
        )

    # 4) segmentation
    def _segment_row(x):
        t = x.get(text_col)
        if lang == "auto":
            l = detect_lang(str(t)) if pd.notna(t) else "en"
        else:
            l = lang

        if l == "zh":
            return split_zh_sentences(t, min_len=min_len_zh)
        else:
            return split_en_sentences(t, min_len=min_len_en)

    df["_segments"] = df.apply(_segment_row, axis=1)

    # 5) explode
    cols = []
    for c in (id_cols + ([group_col] if group_col else []) + [row_id_col, text_col, "_segments"]):
        if c and c not in cols:
            cols.append(c)

    df_seg = (
        df[cols]
        .explode("_segments")
        .dropna(subset=["_segments"])
        .reset_index(drop=True)
        .rename(columns={"_segments": "text"})
    )

    # 6) filter the length
    if lang == "auto":
        df_seg["lang"] = df_seg["text"].apply(detect_lang)
    else:
        df_seg["lang"] = lang

    # 7) segment_id, grouping by group_col if provided
    if group_col:
        df_seg["segment_id"] = df_seg.groupby(group_col).cumcount() + 1
    else:
        df_seg["segment_id"] = range(1, len(df_seg) + 1)

    # 8) unit_id 
    speaker_key = group_col if group_col else id_cols[0]
    df_seg["unit_id"] = (
        df_seg[speaker_key].astype(str)
        + "_"
        + df_seg[row_id_col].astype(str)
        + "_"
        + df_seg["segment_id"].astype(str)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Wrote segmented units: {out_csv} ({len(df_seg)} rows)")


def main():
    p = argparse.ArgumentParser(description="ENA local tool — CSV segmentation + GPT paradigms")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------- segment_csv ----------
    s = sub.add_parser("segment_csv", help="Segment a CSV text column into units (explode)")
    s.add_argument("--in_csv", required=True, type=Path)
    s.add_argument("--out_csv", required=True, type=Path)
    s.add_argument("--text_col", required=True, help="column name: the text to be segmented")
    s.add_argument("--id_cols", required=True, help="reserved id columns, e.g. _id,student_id")
    s.add_argument("--group_col", default=None, help="segment numbering group by this column")
    s.add_argument("--lang", default="auto", choices=["auto", "zh", "en"])
    s.add_argument("--min_len_zh", type=int, default=8)
    s.add_argument("--min_len_en", type=int, default=1)

    # ---------- paradigms ----------
    g = sub.add_parser("paradigms", help="Generate concept paradigm sentences via OpenAI API")
    g.add_argument("--concepts", required=True, type=Path, help="path to concepts.yaml")
    g.add_argument("--out_csv", required=True, type=Path, help="output paradigms csv")
    g.add_argument("--lang", required=True, choices=["zh", "en"], help="language for paradigms")
    g.add_argument("--n", type=int, default=20, help="number of paradigms per concept")
    g.add_argument("--model", type=str, default="gpt-5.2", help="OpenAI model name")

    args = p.parse_args()

    if args.cmd == "segment_csv":
        segment_csv(
            in_csv=args.in_csv,
            out_csv=args.out_csv,
            text_col=args.text_col,
            id_cols=[c.strip() for c in args.id_cols.split(",") if c.strip()],
            group_col=args.group_col,
            lang=args.lang,
            min_len_zh=args.min_len_zh,
            min_len_en=args.min_len_en,
        )

    elif args.cmd == "paradigms":
        step_paradigms(
            concepts_yaml=args.concepts,
            out_csv=args.out_csv,
            lang=args.lang,
            n=args.n,
            model=args.model,
        )
        

######Paradigms and API#####################################################

import json
from dataclasses import dataclass
from typing import Dict
import yaml
from openai import OpenAI

@dataclass
class Concept:
    code: str
    label: str
    definition: str

def load_concepts_yaml(path: Path) -> list[Concept]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    concepts = []
    for c in data.get("concepts", []):
        concepts.append(Concept(
            code=str(c["code"]).strip(),
            label=str(c.get("label", c["code"])).strip(),
            definition=str(c.get("definition", "")).strip(),
        ))
    if not concepts:
        raise ValueError("concepts.yaml contains no concepts.")
    return concepts

def generate_paradigms_openai(
    concept: Concept,
    lang: str,
    n: int,
    model: str,
) -> list[str]:
    client = OpenAI()

    if lang == "zh":
        lang_hint = "繁體中文"
        style_hint = "句子像學生反思或專案報告中的自然敘述。"
    else:
        lang_hint = "English"
        style_hint = "Sentences should read like natural student reflective writing or project reporting."

    prompt = f"""
You are helping create a set of paradigm example sentences for a conceptual category.

Concept code: {concept.code}
Concept label: {concept.label}
Concept definition: {concept.definition}

Generate {n} short, diverse, and representative sentences that clearly express this concept.

Language: {lang_hint}
Constraints:
- {style_hint}
- One sentence per line.
- Avoid duplicates and near-duplicates.
- Do not mention the concept code or label explicitly.
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    text = resp.output_text.strip()

    sents = [l.strip() for l in text.splitlines() if l.strip()]
    return sents


def step_paradigms(
    concepts_yaml: Path,
    out_csv: Path,
    lang: str,
    n: int,
    model: str,
) -> None:
    concepts = load_concepts_yaml(concepts_yaml)

    rows = []
    for c in concepts:
        sents = generate_paradigms_openai(
            concept=c,
            lang=lang,
            n=n,
            model=model,
        )

        for s in sents:
            rows.append({
                "concept_code": c.code,
                "concept_label": c.label,
                "lang": lang,
                "text": s.strip(),
                "source": "gpt_paradigm",
            })

    df = pd.DataFrame(rows).dropna()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Wrote paradigms dataset: {out_csv} ({len(df)} rows)")




if __name__ == "__main__":
    main()


