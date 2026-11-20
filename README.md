# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/cwp76fetft/badge.svg)](https://ask.thili.ai/b/cwp76fetft)

This project develops a Retail Basket Prediction Analyzer, a tool for retail analysts to compare the effectiveness of different next-item prediction models. You will implement and evaluate both an n-gram model and a small language model on simulated shopping basket data to determine which approach offers better predictive accuracy for future purchases.

## What you are building

**Goal.** You will build a Retail Basket Prediction Analyzer that compares the next-item prediction accuracy of an n-gram model and a small language model, both trained on simulated shopping basket data.

**Done when.** You have trained both a next-item language model and an N-gram model on the same simulated retail basket data, evaluated their prediction performance using relevant metrics, and presented a comparison of their results.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**  
      Hint: Apply the data generation techniques from 'Teach a Language Model What Sells Together' to create your dataset.
- [ ] **2. Implement the n-gram model to predict next items in a basket.**  
      Hint: Implement the n-gram model as demonstrated in the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Pretrain a small language model (like in the drill) on the basket sequences.**  
      Hint: Use the pretraining steps for a small language model from 'Teach a Language Model What Sells Together' on your basket sequences.
- [ ] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**  
      Hint: Adapt the evaluation metrics, such as accuracy, from 'Build a Next-Word Predictor from Scratch' to assess both models.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Compile and present the evaluation results, highlighting the performance differences and insights gained from both models.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Explore how different n-gram orders affect the prediction accuracy of the n-gram model.
- Identify specific product categories or itemsets where one model significantly outperforms the other.
- Visualize the distribution of prediction ranks for correct next items for each model to understand their confidence.
