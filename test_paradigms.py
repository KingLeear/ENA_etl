# test_paradigms.py
from pathlib import Path
import pandas as pd
import yaml
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class Concept:
    code: str
    label: str
    definition: str

def load_concepts_yaml(path: Path) -> list[Concept]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Concept(c["code"], c.get("label", c["code"]), c.get("definition", ""))
        for c in data.get("concepts", [])
    ]

def generate_paradigms_openai(concept: Concept, lang: str, n: int, model: str) -> list[str]:
    client = OpenAI()

    if lang == "zh":
        lang_hint = "繁體中文"
    else:
        lang_hint = "English"

    prompt = f"""
Generate {n} representative example sentences for this concept.

Concept: {concept.label}
Definition: {concept.definition}

Language: {lang_hint}
One sentence per line.
""".strip()

    resp = client.responses.create(model=model, input=prompt)
    text = resp.output_text.strip()
    return [l.strip() for l in text.splitlines() if l.strip()]

def main():
    concepts = load_concepts_yaml(Path("schemas/concepts.yaml"))
    rows = []

    for c in concepts:
        sents = generate_paradigms_openai(c, lang="zh", n=5, model="gpt-5.2")
        for s in sents:
            rows.append({
                "concept_code": c.code,
                "concept_label": c.label,
                "lang": "zh",
                "text": s,
                "source": "gpt",
            })

    df = pd.DataFrame(rows)
    out = Path("data_out/paradigms_test.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("Wrote:", out)

if __name__ == "__main__":
    main()