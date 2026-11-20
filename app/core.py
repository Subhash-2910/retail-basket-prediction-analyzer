"""The analyzer: basket data, an n-gram model, evaluation metrics and the comparison report.

A basket is a list of distinct item names. Shoppers' items come out in a random order (the generator shuffles
them), so "the next item" means: given the items seen so far in this basket, which item comes next?
The small language model lives in `app/lm.py`; it shares the evaluation code here.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BOS = "<s>"  # start-of-basket marker for the n-gram contexts


# --- Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
# Uses the generator from 'Teach a Language Model What Sells Together' (the `thilimarketbasket` library):
# a 1,000-SKU grocery catalogue, popularity-weighted background baskets, and 32 itemsets planted at known supports.

def to_sequence(basket: list[str]) -> list[str]:
    """Distinct items in first-seen order. Buying milk twice is one 'next item' event, not two."""
    return list(dict.fromkeys(basket))


def generate_baskets(n_baskets: int = 30_000, seed: int = 0, sample_size: int = 200_000) -> dict:
    """Simulated baskets with planted itemsets. Returns {"catalog", "planted", "baskets"} as plain JSON-able data."""
    from thilimarketbasket.generate import (background_baskets, default_planted, make_catalog, plant_all,
                                            popularity)

    catalog = make_catalog(seed=seed)
    shares = popularity(catalog)
    sample = background_baskets(sample_size, catalog, shares, seed=seed + 10_000)  # checks chance co-occurrence
    planted = default_planted(catalog, shares, sample, seed=seed)
    raw = plant_all(background_baskets(n_baskets, catalog, shares, seed=seed), planted, catalog, seed=seed)
    return {
        "catalog": [{"name": i.name, "base": i.base, "category": i.category} for i in catalog],
        "planted": [{"items": list(p.items), "support": p.support, "kind": p.kind} for p in planted],
        "baskets": [to_sequence(b) for b in raw],
    }


def support(baskets: list[list[str]], items) -> float:
    """Share of baskets that contain every one of `items`."""
    wanted = set(items)
    return sum(wanted <= set(b) for b in baskets) / len(baskets)


def save_dataset(data: dict, directory="data") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "catalog.json").write_text(json.dumps(data["catalog"], indent=1))
    planted = [{**p, "realised": support(data["baskets"], p["items"])} for p in data["planted"]]
    (directory / "planted.json").write_text(json.dumps(planted, indent=1))
    (directory / "baskets.jsonl").write_text("\n".join(json.dumps(b) for b in data["baskets"]) + "\n")
    return directory


def load_dataset(directory="data") -> dict:
    directory = Path(directory)
    if not (directory / "baskets.jsonl").exists():
        raise FileNotFoundError(f"no dataset in {directory}/ - run `python -m app.cli generate` first")
    return {
        "catalog": json.loads((directory / "catalog.json").read_text()),
        "planted": json.loads((directory / "planted.json").read_text()),
        "baskets": [json.loads(line) for line in (directory / "baskets.jsonl").read_text().splitlines() if line],
    }


def split(baskets: list, test_fraction: float = 0.2, seed: int = 0) -> tuple[list, list]:
    """The same train/test split every time for the same seed; both models use it."""
    order = np.random.default_rng(seed).permutation(len(baskets))
    n_test = int(round(test_fraction * len(baskets)))
    test = {int(i) for i in order[:n_test]}
    return ([b for i, b in enumerate(baskets) if i not in test], [b for i, b in enumerate(baskets) if i in test])


def shuffled(baskets: list, seed: int = 0) -> list:
    """Each basket in one fixed random order: the order the test set is read in, identical for both models."""
    rng = random.Random(seed)
    out = []
    for b in baskets:
        b = list(b)
        rng.shuffle(b)
        out.append(b)
    return out


def vocabulary(baskets: list) -> list[str]:
    return sorted({item for b in baskets for item in b})


# --- Milestone 2: Implement the n-gram model to predict next items in a basket.
# As in 'Build a Next-Word Predictor from Scratch': count which item follows each context of the last n-1
# items, and back off to shorter contexts (finally to plain popularity) when a context was never seen.

class NGramModel:
    """An order-n next-item model with 'stupid backoff'.

    Because item order inside a basket is random, each training basket is counted in `n_orders` random orders
    (the same augmentation the language model gets, one fresh order per epoch).
    """

    def __init__(self, n: int = 2, n_orders: int = 5, backoff: float = 0.4, seed: int = 0):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n, self.n_orders, self.backoff, self.seed = n, n_orders, backoff, seed

    def fit(self, baskets: list[list[str]]) -> "NGramModel":
        self.vocab = vocabulary(baskets)
        self.index = {item: i for i, item in enumerate(self.vocab)}
        counts = [defaultdict(Counter) for _ in range(self.n)]  # counts[k][context of k items][next item]
        rng = random.Random(self.seed)
        for basket in baskets:
            for _ in range(self.n_orders):
                seq = list(basket)
                rng.shuffle(seq)
                history = [BOS] + seq
                for i, item in enumerate(seq):
                    before = history[: i + 1]
                    for k in range(min(self.n - 1, len(before)) + 1):
                        counts[k][tuple(before[len(before) - k:])][self.index[item]] += 1
        # stored sparsely: context -> (item positions, probabilities)
        self.tables = []
        for table in counts:
            self.tables.append({ctx: (np.fromiter(c.keys(), int), np.fromiter(c.values(), float) / sum(c.values()))
                                for ctx, c in table.items()})
        unigram = np.ones(len(self.vocab))  # add-one so an item never seen still gets a (tiny) score
        idx, p = self.tables[0][()]
        unigram[idx] += p * sum(counts[0][()].values())
        self.unigram = unigram / unigram.sum()
        return self

    def scores(self, prefix: list[str]) -> np.ndarray:
        """A score for every vocabulary item as the next item after `prefix` (higher = more likely).

        Stupid backoff: each context length gives P(item | context), discounted by `backoff` for each step
        down from the full order, and an item keeps its best score. Items already in the basket are ruled out.
        """
        history = [BOS] + list(prefix)
        out = self.unigram * self.backoff ** (self.n - 1)
        for k in range(1, min(self.n - 1, len(history)) + 1):
            found = self.tables[k].get(tuple(history[len(history) - k:]))
            if found is not None:
                idx, p = found
                out[idx] = np.maximum(out[idx], p * self.backoff ** (self.n - 1 - k))
        for item in prefix:
            if item in self.index:
                out[self.index[item]] = -np.inf
        return out

    def predict(self, prefix: list[str], k: int = 5) -> list[str]:
        s = self.scores(prefix)
        return [self.vocab[i] for i in np.argsort(-s, kind="stable")[:k]]

    def ranks(self, sequences: list[list[str]]) -> list[np.ndarray]:
        """For each basket, the rank (1 = top guess) of the true item at positions 1..len-1."""
        out = []
        for seq in sequences:
            r = []
            for i in range(1, len(seq)):
                s = self.scores(seq[:i])
                target = self.index.get(seq[i])
                r.append(rank_of(s, target))
            out.append(np.array(r, dtype=int))
        return out


def rank_of(scores: np.ndarray, target: int | None) -> int:
    """1 + the number of items scored strictly above the target. An item never seen in training ranks last."""
    if target is None:
        return int(np.isfinite(scores).sum()) + 1
    return int((scores > scores[target]).sum()) + 1


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Adapted from 'Build a Next-Word Predictor from Scratch': top-k accuracy, plus the rank of the true next item.

def metrics(ranks) -> dict:
    """Top-1/5/10 accuracy, mean reciprocal rank, mean and median rank, from a flat array of ranks."""
    r = np.asarray(ranks, dtype=float)
    if r.size == 0:
        return {"n": 0}
    return {"n": int(r.size), "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "top10": float((r <= 10).mean()), "mrr": float((1 / r).mean()),
            "mean_rank": float(r.mean()), "median_rank": float(np.median(r))}


def position_labels(sequences: list[list[str]], planted: list[dict]) -> list[np.ndarray]:
    """Mark each predicted position 'planted' when the true next item completes a planted itemset that some
    earlier item in the basket already belongs to (the signal a model can learn), else 'other'."""
    partners = defaultdict(set)
    for p in planted:
        for a in p["items"]:
            partners[a] |= set(p["items"]) - {a}
    out = []
    for seq in sequences:
        labels, seen_partners = [], set()
        for i, item in enumerate(seq):
            if i > 0:
                labels.append("planted" if item in seen_partners else "other")
            seen_partners |= partners.get(item, set())
        out.append(np.array(labels, dtype=object))
    return out


def evaluate_ranks(ranks: list[np.ndarray], labels: list[np.ndarray], categories: list[np.ndarray]) -> dict:
    """Overall metrics, plus a breakdown by planted/other position and by the true item's category."""
    flat = np.concatenate(ranks) if ranks else np.array([])
    lab = np.concatenate(labels) if labels else np.array([])
    cat = np.concatenate(categories) if categories else np.array([])
    return {
        "overall": metrics(flat),
        "by_position": {name: metrics(flat[lab == name]) for name in ("planted", "other")},
        "by_category": {c: metrics(flat[cat == c]) for c in sorted(set(cat))},
    }


def target_categories(sequences: list[list[str]], catalog: list[dict]) -> list[np.ndarray]:
    category = {c["name"]: c["category"] for c in catalog}
    return [np.array([category.get(item, "?") for item in seq[1:]], dtype=object) for seq in sequences]


# --- Milestone 5: Present a comparison of the models' performance.

def comparison_table(results: dict) -> str:
    """A Markdown table of the headline metrics, one row per model."""
    lines = ["| Model | Top-1 | Top-5 | Top-10 | MRR | Mean rank | Median rank |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, res in results.items():
        m = res["overall"]
        lines.append(f"| {name} | {m['top1']:.1%} | {m['top5']:.1%} | {m['top10']:.1%} | {m['mrr']:.3f} "
                     f"| {m['mean_rank']:.1f} | {m['median_rank']:.0f} |")
    return "\n".join(lines)


def breakdown_table(results: dict, key: str, rows: list[str], metric: str = "top5") -> str:
    names = list(results)
    lines = [f"| {key} | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for row in rows:
        cells = []
        for name in names:
            m = results[name][key].get(row, {})
            cells.append(f"{m[metric]:.1%}" if m.get("n") else "-")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def insights(results: dict, lm_name: str = "Small LM", ngram_name: str = "Bigram (n=2)") -> list[str]:
    """Plain-language findings computed from the numbers (no hand-written claims)."""
    lm, ng = results[lm_name], results[ngram_name]
    out = []
    a, b = lm["overall"]["top5"], ng["overall"]["top5"]
    better = lm_name if a > b else ngram_name
    out.append(f"Overall, the {better} is better: top-5 accuracy {lm_name} {a:.1%} vs {ngram_name} {b:.1%} "
               f"(MRR {lm['overall']['mrr']:.3f} vs {ng['overall']['mrr']:.3f}).")
    lp, np_ = lm["by_position"]["planted"], ng["by_position"]["planted"]
    lo, no = lm["by_position"]["other"], ng["by_position"]["other"]
    if lp.get("n") and np_.get("n"):
        out.append(f"Where the next item completes a planted itemset ({lp['n']:,} positions), top-5 accuracy is "
                   f"{lp['top5']:.1%} for the {lm_name} and {np_['top5']:.1%} for the {ngram_name}: the LM reads the "
                   f"whole basket so far, the n-gram only the last item(s), and basket order is random.")
    if lo.get("n"):
        out.append(f"On ordinary positions ({lo['n']:,}) both models fall back to popularity: "
                   f"{lo['top5']:.1%} vs {no['top5']:.1%} top-5.")
    ngrams = sorted((k for k in results if k.startswith(("Unigram", "Bigram", "Trigram", "4-gram"))),
                    key=lambda k: -results[k]["overall"]["mrr"])
    if len(ngrams) > 1:
        out.append("N-gram orders ranked by MRR: " + ", ".join(
            f"{k} {results[k]['overall']['mrr']:.3f}" for k in ngrams)
            + ". Longer contexts get sparse quickly with 1,000 items, so higher orders gain little or lose.")
    return out
