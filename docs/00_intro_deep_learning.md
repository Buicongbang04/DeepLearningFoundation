# Chapter 0 — Deep Learning là gì?

> Deep Learning is not a magic new field — it is the natural next step from classical Machine Learning once we let the model **learn the features itself**.

## Mục tiêu (Goal)

After this chapter you can:

- Place Deep Learning inside the bigger AI → ML → Representation Learning → DL diagram.
- Explain *why* learning representations matters and where feature engineering fails.
- Describe Deep Learning as the optimization of a multi-layer function over data.
- Write a 300-500-word note comparing classical ML and Deep Learning.

## Why this chapter

This chapter is the orientation map. There is no PyTorch yet, no code to run. We are answering three questions:

1. What problem does Deep Learning actually solve that classical ML cannot?
2. Why is "depth" valuable — what does it buy us?
3. What is the cost of going deep, and when is it the wrong tool?

- **Builds on:** any intro ML course that taught train/val/test split, classification vs regression, overfitting.
- **Sets up:** every later chapter — the rest of the course is the engineering of the answer to "depth is valuable, here is how to make it actually work in PyTorch".

## Key concepts

### AI vs. ML vs. DL

These three terms are nested:

```
                                  Artificial Intelligence (AI)
                                 /                              \
                          Machine Learning (ML)           Symbolic / Rule-based AI
                          /                  \
              Representation Learning      Classical ML (hand-crafted features)
                       |
                Deep Learning (multi-layer learned representations)
```

- **AI** — any system that performs tasks that "look intelligent" (rule-based chess engines belong here).
- **ML** — systems that learn from data instead of from hand-coded rules.
- **Representation Learning** — ML systems that learn *features* from data rather than relying on hand-crafted features.
- **Deep Learning** — a representation-learning approach where the features are organized in *multiple layers* of increasing abstraction.

Source: Goodfellow et al., *Deep Learning*, Chapter 1, Figure 1.4 (Venn diagram of AI subfields).

### Feature engineering vs. learned representations

In classical ML you spent days picking features:

- For images: edges, HoG, SIFT, color histograms.
- For text: n-grams, TF-IDF, hand-built dictionaries.
- For audio: MFCC, spectrogram bins.

Then you fed those features into a linear classifier (SVM, Logistic Regression). The model itself was simple — the *features were the model*.

Deep Learning flips this:

- You feed the raw input (pixels, tokens, audio samples) directly.
- The early layers learn low-level features (edges, n-gram-like patterns, frequency bands).
- The middle layers learn parts (corners, phrases, motifs).
- The top layers learn task-specific concepts (object identity, sentiment, intent).

This is the **hierarchy of concepts**:

```
pixels   →   edges   →   corners   →   parts   →   objects
```

The hierarchy is not enforced by you — it *emerges* from optimization, given enough data and depth.

### Deep Learning as multi-layer function optimization

A deep model is a function `f(x; θ)` made of many composed layers:

```
f(x; θ) = f_L( f_{L-1}( … f_1(x; θ_1) … ; θ_{L-1}); θ_L)
```

Training means picking the parameter vector `θ` that minimizes some loss `L(f(x; θ), y)` on a dataset `{(x_i, y_i)}`:

```
θ* = argmin_θ  (1/N) Σ_i  L(f(x_i; θ), y_i)
```

Three things have to go right:

1. **The architecture** has to be *able* to represent the right function (capacity).
2. **The optimization** has to actually *find* a good `θ*` (gradient descent, learning rate, initialization).
3. **The data** has to be enough to *constrain* `θ*` (otherwise we overfit).

This three-way tension is the spine of the entire course.

### When DL is and is not the right tool

Deep Learning is the right tool when:

- You have *lots* of data (tens of thousands of labeled examples or more).
- The input is high-dimensional and "raw" (images, audio, text, sensor streams).
- Hand-crafted features are hard or impossible to define.
- You can afford GPU time and engineering effort.

Classical ML is still the right tool when:

- You have a small dataset (a few hundred to a few thousand rows).
- The input is tabular with meaningful columns.
- You need interpretability (a regulator, a clinician, a customer must read the model).
- You need to ship in a few days with limited compute.

A common mistake is to start with DL because it is fashionable. The right baseline is *always* a classical ML model — Logistic Regression, KNN, Random Forest — and then justify the extra cost of going deep.

## Common pitfalls

- "Deep Learning replaces ML" → wrong. DL *is* a subset of ML. You still need train/val/test split, you still need a metric, you still worry about overfitting.
- "Deeper is always better" → wrong. Beyond a point, depth amplifies optimization problems (vanishing gradients, longer training, more compute) without buying accuracy.
- "I need raw data" → not enough. You need *enough* labeled data. A 200-row deep model is almost always worse than a 200-row Logistic Regression.

## Learning outcomes

- Draw the AI → ML → Representation Learning → DL diagram from memory.
- Explain in your own words why representation matters.
- Write a 300-500-word note comparing classical ML and Deep Learning, with one example where DL clearly wins and one example where it loses.

## Quick check (self-test)

<details>
<summary>Q1 — In one sentence, what is the difference between Representation Learning and classical Machine Learning?</summary>

Representation Learning *learns* the features from data; classical ML uses *hand-crafted* features and learns only the final decision.
</details>

<details>
<summary>Q2 — Give an example of a problem where classical ML beats Deep Learning.</summary>

Tabular data with a few thousand rows and meaningful columns (e.g., credit-risk scoring) — a gradient-boosted tree (XGBoost, LightGBM) usually wins.
</details>

<details>
<summary>Q3 — What does the "hierarchy of concepts" diagram (pixels → edges → corners → parts → objects) try to convey?</summary>

That a deep network organizes features in layers of increasing abstraction, and these layers *emerge* from optimization rather than being designed by hand.
</details>

## Further reading

- *Deep Learning* by Goodfellow, Bengio, Courville — Chapter 1.
- Ian Goodfellow's NIPS 2016 keynote, "Deep Learning, the Past and the Future" (YouTube).
- 3Blue1Brown — *But what is a neural network?* (YouTube).

## Companion artifact

`figures/ai_ml_dl_relationship.png`, `figures/hierarchy_of_concepts.png` (generated in the warm-up notebook of Chapter 1).
