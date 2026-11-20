"""One test (or more) per milestone. They use a small dataset so the whole suite runs in seconds on CI."""
import json

import numpy as np
import pytest

from app import core


@pytest.fixture(scope="module")
def data():
    return core.generate_baskets(n_baskets=4_000, seed=0, sample_size=50_000)


@pytest.fixture(scope="module")
def split(data):
    return core.split(data["baskets"], seed=0)


# Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
def test_milestone_1(data, tmp_path):
    baskets, planted = data["baskets"], data["planted"]
    assert len(baskets) == 4_000 and len(planted) == 32
    assert all(len(b) == len(set(b)) >= 1 for b in baskets)  # distinct items per basket
    # the commonest planted itemsets really are in about 5% of baskets
    for p in [p for p in planted if p["support"] == 0.05]:
        assert core.support(baskets, p["items"]) == pytest.approx(0.05, abs=0.01)
    core.save_dataset(data, tmp_path)
    again = core.load_dataset(tmp_path)
    assert again["baskets"] == baskets
    assert json.loads((tmp_path / "planted.json").read_text())[0]["realised"] > 0


def test_split_is_repeatable_and_disjoint(data):
    train, test = core.split(data["baskets"], seed=0)
    assert len(test) == 800 and len(train) == 3_200
    assert core.split(data["baskets"], seed=0) == (train, test)


# Milestone 2: Implement the n-gram model to predict next items in a basket.
def test_milestone_2(data, split):
    train, _ = split
    bigram = core.NGramModel(n=2).fit(train)
    bread, butter = data["planted"][0]["items"][:2]  # the strongest planted pair
    assert butter in bigram.predict([bread], k=5)
    scores = bigram.scores([bread])
    assert scores[bigram.index[bread]] == -np.inf  # never re-predicts what is already in the basket
    assert np.isfinite(scores).sum() == len(bigram.vocab) - 1


def test_ngram_orders_and_backoff(split):
    train, _ = split
    for n in (1, 2, 3, 4):
        model = core.NGramModel(n=n).fit(train)
        assert len(model.predict(["an item nobody ever bought"], k=3)) == 3  # unseen context backs off


# Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
def test_milestone_3(data, split, tmp_path):
    from app import lm

    train, test = split
    predictor, history = lm.pretrain(train, val=test[:300], epochs=3, seed=0, dim=64, log=lambda *_: None)
    assert history[-1]["val_loss"] < history[0]["val_loss"] - 0.5  # it learned something
    assert history[0]["val_loss"] == pytest.approx(np.log(len(predictor.vocab) + 2), abs=0.3)  # started random
    path = lm.save(predictor, tmp_path / "lm.pt")
    loaded = lm.load(path)
    basket = [data["planted"][0]["items"][0]]
    assert loaded.predict(basket) == predictor.predict(basket)


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
def test_milestone_4_metrics():
    m = core.metrics([1, 2, 5, 10, 100])
    assert m["top1"] == pytest.approx(0.2) and m["top5"] == pytest.approx(0.6) and m["top10"] == pytest.approx(0.8)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.2 + 0.1 + 0.01) / 5)
    assert m["median_rank"] == 5
    assert core.rank_of(np.array([0.1, 0.9, 0.5]), 2) == 2


def test_milestone_4_both_models_rank_the_same_positions(data, split):
    from app import lm

    train, test = split
    sequences = core.shuffled(test[:100], seed=1)
    ngram = core.NGramModel(n=2).fit(train)
    predictor, _ = lm.pretrain(train, epochs=1, dim=32, log=lambda *_: None)
    r1, r2 = ngram.ranks(sequences), predictor.ranks(sequences)
    assert [len(r) for r in r1] == [len(r) for r in r2] == [len(s) - 1 for s in sequences]
    # the batched LM ranks agree with ranking one prefix at a time
    seq = sequences[0]
    assert list(r2[0]) == [core.rank_of(predictor.scores(seq[:i]), predictor.index.get(seq[i]))
                           for i in range(1, len(seq))]
    labels = core.position_labels(sequences, data["planted"])
    result = core.evaluate_ranks(r1, labels, core.target_categories(sequences, data["catalog"]))
    assert result["overall"]["n"] == sum(len(s) - 1 for s in sequences)
    assert result["by_position"]["planted"]["n"] + result["by_position"]["other"]["n"] == result["overall"]["n"]


def test_position_labels():
    planted = [{"items": ["bread", "butter"]}]
    assert list(core.position_labels([["bread", "milk", "butter"]], planted)[0]) == ["other", "planted"]


# Milestone 5: Present a comparison of the models' performance.
def _fake(top5, planted_top5):
    m = {"n": 10, "top1": top5 / 2, "top5": top5, "top10": top5 * 1.5, "mrr": top5 / 2, "mean_rank": 50.0,
         "median_rank": 20.0}
    return {"overall": m, "by_position": {"planted": {**m, "top5": planted_top5}, "other": m},
            "by_category": {"Dairy": m}}


def test_milestone_5():
    results = {"Bigram (n=2)": _fake(0.11, 0.44), "Small LM": _fake(0.17, 0.86)}
    table = core.comparison_table(results)
    assert "| Small LM | 8.5% | 17.0% |" in table and "Bigram (n=2)" in table
    findings = core.insights(results)
    assert findings[0].startswith("Overall, the Small LM is better")
    assert "86.0%" in findings[1] and "44.0%" in findings[1]
    assert "| planted | 44.0% | 86.0% |" in core.breakdown_table(results, "by_position", ["planted"])


def test_cli_end_to_end(tmp_path, capsys):
    from app.cli import main

    common = ["--data", str(tmp_path / "data"), "--model", str(tmp_path / "lm.pt"), "--results", str(tmp_path / "out")]
    main(common + ["all", "--n-baskets", "10000", "--epochs", "1", "--orders", "1", "2"])
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Small LM" in report and "Bigram (n=2)" in report
    assert (tmp_path / "out" / "rank_distribution.png").exists()
