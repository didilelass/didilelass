# 🧠 Cheat sheet: Bayes, ridge, lasso

One page with easy ways to remember it all. The hands-on version is in [`learn_bayes_ridge_lasso.ipynb`](learn_bayes_ridge_lasso.ipynb).

---

## 1. Bayes' rule
**Easy version:** *Start with a guess, then update it with evidence.*
**Picture:** a new café. You assume it's average (prior). One great coffee: "probably good". Ten great coffees: "definitely good". One bad cup after that barely changes your mind.

```
P(A|B) = P(B|A) · P(A) / P(B)

posterior ∝ likelihood × prior
new belief ∝ how well it explains the data × old belief
```

| Word | Means | Café |
|---|---|---|
| **Prior** P(A) | belief before data | "it's probably average" |
| **Likelihood** P(B\|A) | how well a belief explains the data | "if it's good, a great cup is likely" |
| **Posterior** P(A\|B) | belief after data | "it's good" |
| **Evidence** P(B) | just a constant that makes it add up to 1 | (ignore it) |

**Memory hook:** *"Posterior = prior updated by likelihood."* P-L-P.

---

## 2. Bayesian linear regression
**Easy version:** *Linear regression that also tells you how sure it is.*
Instead of one best weight, every weight is a bell curve: a best guess plus a ± range.

```
Model:      y = Xw + noise,  noise ~ N(0, σ²)
Prior:      w ~ N(μ₀, Σ₀)
Posterior:  Σₙ = (Σ₀⁻¹ + XᵀX/σ²)⁻¹
            μₙ = Σₙ (Σ₀⁻¹μ₀ + Xᵀy/σ²)
```

**Memory hook:** *"Precisions add."* Precision = 1/variance = how sure you are.
How sure after = how sure before + how much the data tells you.

**Use it when:** little data, you have prior knowledge, or you need a range (like the Run Coach app).

---

## 3. Ridge (L2)
**Easy version:** *"Don't let any weight get too big."* All weights shrink a little; none become 0.

```
minimise   ‖y − Xw‖²  +  λ Σ wⱼ²
solution   w = (XᵀX + λI)⁻¹ Xᵀy        ← exact formula
```

**The Bayes link:** ridge = Bayesian regression with a **Gaussian prior centred on 0**, keeping only the best guess.
`λ = σ² / τ²` (noise ÷ prior width). Narrow prior → big λ → more shrinkage.

**Memory hook:** *Ridge = **R**ound = squares = circle = smooth shrink.*
**Use it when:** many features that all matter a bit, or correlated features (ridge shares the weight between them).

---

## 4. Lasso (L1)
**Easy version:** *"Use as few features as possible."* Some weights go to exactly 0, so it picks features for you.

```
minimise   ‖y − Xw‖²  +  λ Σ |wⱼ|
solution   no exact formula → solved step by step (coordinate descent)
```

**The Bayes link:** lasso = **Laplace prior** (sharp peak at 0).
**Why exactly 0?** |w| has a corner at 0. The L1 region is a **diamond** with corners on the axes, and the best point usually lands on a corner, where some weights are 0. Ridge's region is a **circle**, which has no corners.

**Memory hook:** *Lasso **lassos** a few features and throws the rest away.*
**Weakness:** with correlated features it picks one more or less at random. → **Elastic net** = L1 + L2 fixes it.

---

## 5. λ, the one dial behind it all
**Easy version:** *λ = how strongly you believe "weights should be small".*

```
λ small → data rules        → complex model → risk: OVERFIT  (great on training data, bad on new)
λ big   → prior/penalty rules → simple model  → risk: UNDERFIT (bad everywhere)
```

That's the **bias–variance trade-off**. Choose λ with **cross-validation**. For data over time, use `TimeSeriesSplit` (train on the past, test on the future).

---

## 🗂️ Everything in one table

| | Plain linear | Ridge | Lasso | Elastic net | Bayesian |
|---|---|---|---|---|---|
| Penalty | none | λΣw² (L2) | λΣ\|w\| (L1) | both | prior (any) |
| Prior (Bayes view) | flat | Gaussian at 0 | Laplace at 0 | mix | Gaussian at μ₀ (or other) |
| Weights become 0? | no | no | **yes** | yes | no |
| Exact formula? | yes | yes | no | no | yes (Gaussian) |
| Gives uncertainty? | no | no | no | no | **yes** |
| Best for | lots of data | correlated features | few features matter | correlated + selection | little data, need a range |

---

## 💼 What interviews actually expect

**Short answer: no, they don't expect you to derive all this from memory.** For most data analyst / data scientist interviews, they want to see that you **understand the ideas and know when to use what**. Being able to derive or rewrite the formulas is a bonus that shows depth.

| Level | What they expect | You should be able to… |
|---|---|---|
| **Must know** (almost every DS interview) | Overfitting, regularisation, bias–variance, cross-validation | Explain ridge vs lasso in plain words and say when to use each |
| **Should know** | L1 → feature selection, L2 → shrinkage; why lasso gives zeros (diamond vs circle); Bayes' rule with a simple example | Draw the circle/diamond picture; do a P(A\|B) problem (the classic is the "medical test" question) |
| **Nice to have** (ML-heavy or research roles) | Ridge = Gaussian prior, lasso = Laplace prior; ridge closed form; posterior formula | Derive w = (XᵀX + λI)⁻¹Xᵀy in 5 lines |
| **Rarely asked** | Full Bayesian posterior derivation by hand | Recognise it, not reproduce it |

**What matters most in an interview:**
1. **Explain simply.** "Lasso sets some weights to zero, so it also picks features" beats a formula.
2. **Know the trade-offs.** When would you choose ridge? (Correlated features, all useful.) Lasso? (Want a simpler model.)
3. **Talk about a real project.** *"I built a 5K predictor on my own running data. With only a few races I used Bayesian regression with a prior, so it starts sensible and personalises as data comes in. In the bigger version I compared ridge, random forest and gradient boosting with time-series cross-validation."* That shows more than any formula.

### Practice questions (answer out loud, 30 seconds each)
1. What's the difference between L1 and L2 regularisation?
2. Why does lasso produce sparse models?
3. What happens if λ is too large? Too small?
4. How would you choose λ?
5. What's a prior? Give an example.
6. A test is 99% accurate and the disease affects 1 in 1,000 people. You test positive. Chance you have it? *(≈ 9%. Prior matters!)*
7. Why might you prefer a Bayesian model with very little data?
8. Two features are highly correlated. What happens with lasso vs ridge?

<details><summary>▶ Answer to Q6</summary>

P(sick | +) = P(+ | sick)·P(sick) / P(+) = 0.99 × 0.001 / (0.99 × 0.001 + 0.01 × 0.999) ≈ 0.00099 / 0.01098 ≈ **9%**.
Out of 1,000 people: 1 is sick and tests positive, and ~10 healthy people also test positive. So only 1 in 11 positives is real.
</details>
