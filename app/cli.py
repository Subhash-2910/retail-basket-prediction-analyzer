"""Command-line entry point: `python -m app.cli --help`.

    generate  -> data/            simulated baskets with planted itemsets      (milestone 1)
    train     -> models/lm.pt     the small LM, pretrained on the baskets      (milestone 3)
    evaluate  -> results/metrics.json  n-gram (orders 1-4) vs the LM         (milestones 2 and 4)
    compare   -> results/report.md + charts                                   (milestone 5)
    all       -> every step above, in order
    predict   -> next-item guesses from both models for a basket you type
"""
import argparse
import json
from pathlib import Path

ORDER_NAMES = {1: "Unigram (n=1)", 2: "Bigram (n=2)", 3: "Trigram (n=3)", 4: "4-gram (n=4)"}
LM_NAME = "Small LM"
MAX_LEN = 32  # longest basket prefix the LM reads (0.02% of baskets are longer)


def cmd_generate(args):
    from app import core

    data = core.generate_baskets(n_baskets=args.n_baskets, seed=args.seed)
    core.save_dataset(data, args.data)
    planted = json.loads((Path(args.data) / "planted.json").read_text())
    lengths = [len(b) for b in data["baskets"]]
    print(f"{len(data['baskets']):,} baskets, {len(data['catalog']):,} items, {len(planted)} planted itemsets, "
          f"{sum(lengths) / len(lengths):.1f} distinct items per basket -> {args.data}/")
    for p in planted[:5]:
        print(f"  planted {p['kind']:<10} target {p['support']:.2%} realised {p['realised']:.2%}  {p['items']}")


def cmd_train(args):
    from app import core, lm

    data = core.load_dataset(args.data)
    train, test = core.split(data["baskets"], seed=args.seed)
    predictor, history = lm.pretrain(train, val=test[:2000], epochs=args.epochs, seed=args.seed)
    lm.save(predictor, args.model)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    (Path(args.results) / "train_history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {lm.n_parameters(predictor):,}-parameter model -> {args.model}")


def cmd_evaluate(args):
    from app import core, lm

    data = core.load_dataset(args.data)
    train, test = core.split(data["baskets"], seed=args.seed)
    # one fixed order, the same for every model; capped at the LM's context length so both see the same positions
    sequences = [s[:MAX_LEN] for s in core.shuffled(test, seed=args.seed + 1)]
    labels = core.position_labels(sequences, data["planted"])
    categories = core.target_categories(sequences, data["catalog"])
    models = {ORDER_NAMES[n]: core.NGramModel(n=n, seed=args.seed).fit(train) for n in args.orders}
    models[LM_NAME] = lm.load(args.model)
    results, rank_lists = {}, {}
    for name, model in models.items():
        print(f"evaluating {name} ...", flush=True)
        ranks = model.ranks(sequences)
        results[name] = core.evaluate_ranks(ranks, labels, categories)
        rank_lists[name] = [int(r) for row in ranks for r in row]
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    (out / "rank_lists.json").write_text(json.dumps(rank_lists))
    print(core.comparison_table(results))


def cmd_compare(args):
    from app import core

    out = Path(args.results)
    if not (out / "metrics.json").exists():
        raise SystemExit("no results yet - run `python -m app.cli evaluate` first")
    results = json.loads((out / "metrics.json").read_text())
    rank_lists = json.loads((out / "rank_lists.json").read_text())
    charts = _charts(results, rank_lists, out)
    findings = core.insights(results, LM_NAME, ORDER_NAMES[2])
    categories = sorted(results[LM_NAME]["by_category"], key=lambda c: -results[LM_NAME]["by_category"][c]["n"])
    report = [
        "# Retail basket next-item prediction: n-gram vs small LM", "",
        "Both models were trained on the same 80% of the simulated baskets and tested on the other 20%, read in "
        "the same random order. At each position the model ranks every catalogue item not yet in the basket; "
        "the rank of the true next item gives top-k accuracy, MRR and mean rank.", "",
        "## Headline results", "", core.comparison_table(results), "",
        "## Insights", "", *[f"- {line}" for line in findings], "",
        "## Planted itemsets vs ordinary positions (top-5 accuracy)", "",
        core.breakdown_table(results, "by_position", ["planted", "other"]), "",
        "## By category of the true next item (top-5 accuracy, 10 largest)", "",
        core.breakdown_table(results, "by_category", categories[:10]), "",
        "## Charts", "", *[f"![{c.stem}]({c.name})" for c in charts], "",
    ]
    if args.summary:
        report += ["## Plain-English summary (written by an LLM from the numbers above)", "", _llm_summary(report), ""]
    (out / "report.md").write_text("\n".join(report))
    print(core.comparison_table(results))
    print()
    print("\n".join(f"- {line}" for line in findings))
    print(f"\nreport -> {out / 'report.md'}")


def _charts(results, rank_lists, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    paths = []
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    for i, (metric, label) in enumerate([("top1", "Top-1"), ("top5", "Top-5"), ("top10", "Top-10")]):
        ax.bar(x + (i - 1) * 0.27, [results[n]["overall"][metric] * 100 for n in names], 0.27, label=label)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Next-item accuracy by model")
    ax.legend()
    fig.tight_layout()
    paths.append(out / "accuracy.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.unique(np.logspace(0, 3, 30).astype(int))
    for name in [ORDER_NAMES[2], LM_NAME]:
        if name in rank_lists:
            ax.hist(rank_lists[name], bins=bins, histtype="step", linewidth=2, density=True, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("rank of the true next item (1 = top guess, log scale)")
    ax.set_ylabel("share of predictions")
    ax.set_title("Where the true next item lands in each model's ranking")
    ax.legend()
    fig.tight_layout()
    paths.append(out / "rank_distribution.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def _llm_summary(report_lines) -> str:
    from app import llm

    try:
        return llm.chat([
            {"role": "system", "content": "You explain model evaluation results to retail analysts in 4-6 plain sentences."},
            {"role": "user", "content": "\n".join(report_lines)},
        ], temperature=0.2)
    except Exception as e:  # the report must not depend on an API key
        return f"(no summary: {e})"


def cmd_predict(args):
    from app import core, lm

    data = core.load_dataset(args.data)
    train, _ = core.split(data["baskets"], seed=args.seed)
    basket = [s.strip() for s in args.items.split(",") if s.strip()]
    for name, model in [(ORDER_NAMES[2], core.NGramModel(n=2).fit(train)), (LM_NAME, lm.load(args.model))]:
        print(f"{name}:")
        for item in model.predict(basket, k=5):
            print(f"  {item}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    parser.add_argument("--data", default="data", help="dataset folder")
    parser.add_argument("--model", default="models/lm.pt", help="where the small LM is saved")
    parser.add_argument("--results", default="results", help="where metrics and the report go")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    g = sub.add_parser("generate", help="milestone 1: simulate baskets with planted itemsets")
    g.add_argument("--n-baskets", type=int, default=30_000)
    t = sub.add_parser("train", help="milestone 3: pretrain the small LM")
    t.add_argument("--epochs", type=int, default=10)
    e = sub.add_parser("evaluate", help="milestones 2+4: score n-gram models and the LM on the test baskets")
    e.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    c = sub.add_parser("compare", help="milestone 5: write results/report.md and charts")
    c.add_argument("--summary", action="store_true", help="add a plain-English summary from the LLM in .env")
    a = sub.add_parser("all", help="generate, train, evaluate and compare in one go")
    a.add_argument("--n-baskets", type=int, default=30_000)
    a.add_argument("--epochs", type=int, default=10)
    a.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    a.add_argument("--summary", action="store_true")
    p = sub.add_parser("predict", help="top-5 next items from both models for a basket")
    p.add_argument("items", help='comma-separated item names, e.g. "Crunchy Crust Bread 400 g"')
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready. Open README.md for your milestones.")
    elif args.command == "all":
        for step in (cmd_generate, cmd_train, cmd_evaluate, cmd_compare):
            step(args)
    else:
        {"generate": cmd_generate, "train": cmd_train, "evaluate": cmd_evaluate,
         "compare": cmd_compare, "predict": cmd_predict}[args.command](args)


if __name__ == "__main__":
    main()
