"""
WebNLG data loading for Graph-LLaDA experiments.

Loads the GEM/web_nlg dataset from HuggingFace and formats it for
graph-to-text generation experiments.

Uses the parquet conversion branch since datasets v4+ dropped loading script
support. English examples are filtered by gem_id prefix.

Dataset fields:
    - input: List of RDF triples, e.g. ["Aarhus_Airport | cityServed | \"Aarhus, Denmark\""]
    - target: Reference natural language text
    - references: Additional reference texts
    - category: DBpedia category (e.g. "Airport", "SportsTeam")
"""

from datasets import load_dataset
from typing import Optional


PROMPT_TEMPLATE = (
    "Given the following knowledge graph triples, write a sentence that "
    "describes the information:\n\n{triples}\n\nSentence:"
)


def load_webnlg(split: str = "test", category: Optional[str] = None):
    """Load WebNLG English split from HuggingFace.

    Args:
        split: Dataset split ("train", "validation", "test")
        category: Filter by DBpedia category (e.g. "Airport"). None = all.

    Returns:
        List of dicts with keys: id, input, target, references, category, prompt
    """
    dataset = load_dataset(
        "GEM/web_nlg", "default", split=split,
        revision="refs/convert/parquet",
    )

    examples = []
    for row in dataset:
        # Filter to English only
        if not row["gem_id"].startswith("web_nlg_en"):
            continue

        if category and row["category"] != category:
            continue

        triples_str = "\n".join(row["input"])
        prompt = PROMPT_TEMPLATE.format(triples=triples_str)

        examples.append({
            "id": row["gem_id"],
            "input": row["input"],
            "target": row["target"],
            "references": row["references"],
            "category": row["category"],
            "prompt": prompt,
        })

    return examples


def format_triples(triples: list[str]) -> str:
    """Format a list of RDF triples as a readable string.

    Args:
        triples: List of "subject | predicate | object" strings

    Returns:
        Newline-joined triple string
    """
    return "\n".join(triples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview WebNLG data")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("-n", type=int, default=3, help="Number of examples to show")
    args = parser.parse_args()

    examples = load_webnlg(split=args.split, category=args.category)
    print(f"Loaded {len(examples)} examples from '{args.split}' split")

    if args.category:
        print(f"Filtered to category: {args.category}")

    # Show unique categories
    categories = sorted(set(ex["category"] for ex in examples))
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print()

    for ex in examples[:args.n]:
        print(f"--- {ex['id']} [{ex['category']}] ---")
        print(f"Triples: {ex['input']}")
        print(f"Target:  {ex['target']}")
        print()
        print("Prompt:")
        print(ex["prompt"])
        print()
