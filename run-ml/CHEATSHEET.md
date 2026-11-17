# 🧠 Cheat sheet: the algorithms behind your 5K predictor

Every idea gets **an everyday explanation**, **math small enough for a phone calculator**, and **a hook to remember it**. Practice questions with answers are at the end.
The hands-on version is in [`learn_bayes_ridge_lasso.ipynb`](learn_bayes_ridge_lasso.ipynb).

**Contents**
1. [Linear regression](#1-linear-regression)
2. [Bayes' rule](#2-bayes-rule)
3. [Bayesian updating (what the app does)](#3-bayesian-updating-what-the-app-does)
4. [Ridge (L2)](#4-ridge-l2)
5. [Lasso (L1)](#5-lasso-l1)
6. [λ and the bias–variance trade-off](#6-λ-and-the-biasvariance-trade-off)
7. [Random forest](#7-random-forest)
8. [Gradient boosting](#8-gradient-boosting)
9. [Cross-validation](#9-cross-validation)
10. [Riegel's formula (5K-equivalent)](#10-riegels-formula-5k-equivalent)
11. [Everything in one table](#-everything-in-one-table)
12. [Interviews + practice questions with answers](#-what-interviews-actually-expect)

---

## 1. Linear regression
**Everyday version:** *Draw the straight line that best fits the dots.* "Best" means the total of the **squared** misses is as small as possible.

**Math:** `y = a + b·x`. With one feature:
```
b = Σ (x − x̄)(y − ȳ) / Σ (x − x̄)²        a = ȳ − b·x̄
```

**Tiny example.** Weekly runs x = 1, 2, 3; minutes saved y = 2, 4, 7.
- Means: x̄ = 2, ȳ = 4.33
- Top: (−1)(−2.33) + (0)(−0.33) + (1)(2.67) = 2.33 + 0 + 2.67 = **5**
- Bottom: (−1)² + 0² + 1² = **2**
- b = 5 / 2 = **2.5**, a = 4.33 − 2.5 × 2 = **−0.67**
- Line: *minutes saved ≈ −0.67 + 2.5 × runs per week*

**Why squares?** Misses in both directions count, and big misses count a lot more (a 2-minute miss counts 4×, not 2×).

**Hook:** *"Line of least squares."*

---

## 2. Bayes' rule
**Everyday version:** *Start with a guess, then update it with evidence.*
**Picture:** a new café. You assume it's average (prior). One great coffee: "probably good". Ten great coffees: "definitely good". One bad cup after that barely changes your mind.

**Math:**
```
P(A|B) = P(B|A) · P(A) / P(B)          posterior ∝ likelihood × prior
```

| Word | Means | Café |
|---|---|---|
| **Prior** P(A) | belief before data | "it's probably average" |
| **Likelihood** P(B\|A) | how well a belief explains the data | "if it's good, a great cup is likely" |
| **Posterior** P(A\|B) | belief after data | "it's good" |
| **Evidence** P(B) | a constant that makes it add up to 1 | (ignore it) |

**Tiny example: the medical test.** 1 in 1,000 people have a disease. The test catches 99% of sick people and wrongly flags 1% of healthy people. You test positive.
- Think of **1,000 people:** 1 is sick (tests positive), 999 are healthy, and ~**10** of them test positive by mistake
- Positives = 1 + 10 = 11, and only 1 is really sick → **1/11 ≈ 9%**
- Formula: 0.99 × 0.001 / (0.99 × 0.001 + 0.01 × 0.999) = 0.00099 / 0.01098 = **9%**

**Hook:** *"Posterior = prior updated by likelihood"* (P-L-P). **Think in counts of people, not percentages.**

---

## 3. Bayesian updating (what the app does)
**Everyday version:** *Your new belief is a **weighted average** of your old belief and what the data says. Whichever is more certain gets more weight.*

**Math (one weight, Gaussian everything):**
```
precision = 1 / variance              (how sure you are)
new precision = prior precision + data precision          ← precisions ADD
new mean = weighted average of prior mean and data, weighted by precision
```

**Tiny example: how many seconds does 1 extra hour of sleep save you?**
- **Prior:** 8 s, sd 12 → precision = 1/12² = **1/144**
- **One race:** you slept 2 h more and ran 40 s faster. The data alone says 40 / 2 = **20 s per hour**.
  A 5K varies ±40 s, so this race's precision = (hours)² / 40² = 4/1600 = **1/400**
- **New precision:** 1/144 + 1/400 = 0.0069 + 0.0025 = 0.0094 → new sd = 1/√0.0094 ≈ **10 s** (more sure than 12)
- **New mean:** (8 × 1/144 + 20 × 1/400) / 0.0094 = (0.056 + 0.050) / 0.0094 ≈ **11 s**

So one race moves you from 8 to 11 s, not all the way to 20, because one race is noisy. Ten races would land close to your true value.

**With many weights** (the app has 8: fitness, consistency, sleep, mood, stress…) it's the same thing with matrices:
```
Σₙ = (Σ₀⁻¹ + XᵀX/σ²)⁻¹                    ← precisions add
μₙ = Σₙ (Σ₀⁻¹μ₀ + Xᵀy/σ²)                 ← weighted average
prediction = μₙ·x ,  range = ± √(σ² + xᵀΣₙx)
```

**Hook:** *"Precisions add, means average."*

---

## 4. Ridge (L2)
**Everyday version:** *"Don't let any weight get too big."* It's a rubber band pulling every weight toward 0: all shrink, none reach 0.

**Math:**
```
minimise  Σ (y − w·x)²  +  λ Σ w²
one feature, no intercept:   w = Σxy / (Σx² + λ)
many features:               w = (XᵀX + λI)⁻¹ Xᵀy
```

**Tiny example** (same data as section 1, line through 0 to keep it simple): Σxy = 1·2 + 2·4 + 3·7 = **31**, Σx² = 1 + 4 + 9 = **14**

| λ | w = 31 / (14 + λ) | |
|---|---|---|
| 0 | 31/14 = **2.21** | plain regression |
| 1 | 31/15 = **2.07** | slight shrink |
| 14 | 31/28 = **1.11** | half |
| 1,000 | 31/1014 = **0.03** | almost 0, but never exactly 0 |

**The Bayes link:** ridge = Bayes with a **Gaussian prior centred on 0**, keeping only the peak. `λ = σ² / τ²` (noise² ÷ prior width²).

**Hook:** *Ridge = **R**ound = squares = circle = smooth shrink. λ sits in the **denominator**, so it divides and never reaches 0.*

---

## 5. Lasso (L1)
**Everyday version:** *"Use as few features as possible."* Weak features are set to **exactly 0**, so lasso also picks features.

**Math:**
```
minimise  Σ (y − w·x)²  +  λ Σ |w|
one feature, no intercept:   w = max(Σxy − λ/2, 0) / Σx²      ("soft threshold")
many features:               no formula → solved step by step (coordinate descent)
```

**Tiny example** (same numbers: Σxy = 31, Σx² = 14):

| λ | w = (31 − λ/2) / 14 | |
|---|---|---|
| 0 | 31/14 = **2.21** | plain regression |
| 10 | 26/14 = **1.86** | |
| 40 | 11/14 = **0.79** | |
| 62 or more | 31 − 31 = 0 → **0** | feature dropped! |

**Compare:** ridge **divides** (gets small, never 0). Lasso **subtracts** (hits 0 and stays there).

**Why the zeros, as a picture:** the L1 "budget" region is a **diamond** with corners on the axes. The best point usually lands on a corner, where some weights = 0. Ridge's region is a **circle**, which has no corners.

**The Bayes link:** lasso = **Laplace prior** (sharp peak at 0).
**Weakness:** with correlated features it picks one more or less at random. **Elastic net** (L1 + L2) fixes that.

**Hook:** *Lasso **lassos** a few features and throws the rest away. Ridge divides, lasso subtracts.*

---

## 6. λ and the bias–variance trade-off
**Everyday version:** *λ is the "trust the data" dial, turned backwards.*

| λ | Model | Risk | Analogy |
|---|---|---|---|
| tiny | follows the data closely | **overfit**: memorises noise, bad on new data | a student who memorised the answers |
| huge | ignores the data | **underfit**: too simple everywhere | a student who answers "C" to everything |
| just right | | lowest error on **new** data | understood the lesson |

**Math idea:** error on new data = bias² + variance + noise. Raising λ lowers variance and raises bias.

**How to choose λ:** try many values and keep the one with the lowest **cross-validation** error (section 9).

**Hook:** *"Big λ = simple. Small λ = wiggly."*

---

## 7. Random forest
**Everyday version:** *Ask 300 slightly different "experts" and average their answers.* Each expert is a **decision tree** (a flowchart of yes/no questions) trained on a random resample of the data, looking at random features. Their individual mistakes cancel out.

**A decision tree:**
```
km per week > 25?
├── yes → slept > 7 h?  ├── yes → 27:30
│                       └── no  → 28:20
└── no  → 30:10
```

**Tiny example:** three trees predict 28:10, 28:40 and 28:25 → forest = average = **28:25**.

**Pros:** handles non-linear patterns, little tuning. **Cons:** can't extrapolate beyond the data it has seen, needs more data than linear models.

**Hook:** *"Wisdom of the crowd of trees."* (Many trees, each trained on a **bag** of data: "bagging".)

---

## 8. Gradient boosting
**Everyday version:** *A team where each new member fixes the previous members' mistakes.* Start with a simple guess; each small tree learns what's still wrong; add a fraction of its fix; repeat.

**Tiny example:** true 5K = 28:00.

| Step | Prediction | Still wrong by | Next tree learns | Add 50% of it (learning rate 0.5) |
|---|---|---|---|---|
| start (average) | 29:00 | +60 s | −60 s | −30 s |
| after tree 1 | 28:30 | +30 s | −30 s | −15 s |
| after tree 2 | 28:15 | +15 s | −15 s | −7.5 s |
| after tree 3 | 28:07.5 | +7.5 s | … | … |

**Math:** `prediction = start + η·tree₁ + η·tree₂ + …`, where each tree fits the **residuals** (errors) left by the previous ones. η is the learning rate.

**Forest vs boosting:** forest = many trees **side by side**, averaged (reduces variance). Boosting = small trees **in a row**, each fixing the last (reduces bias).

**Hook:** *"Forest averages, boosting corrects."*

---

## 9. Cross-validation
**Everyday version:** *Never grade a student with the exam they studied from.* Hide part of the data, train on the rest, test on the hidden part, and repeat.

**K-fold (k = 5):** split into 5 parts → train on 4, test on 1 → repeat 5 times → average the 5 scores.

**Time-series split** (used here, because runs happen over time): always **train on the past, test on the future**.
```
fold 1:  [train ][test]
fold 2:  [train       ][test]
fold 3:  [train             ][test]
```
Shuffling time data lets the model "see the future", which makes it look better than it really is.

**Hook:** *"Train on yesterday, test on tomorrow."*

---

## 10. Riegel's formula (5K-equivalent)
**Everyday version:** *Convert any run to the 5K time it's "worth".* Longer races are run a bit slower per km, and the exponent 1.06 captures that.

```
T₅ₖ = T × (5 / distance)^1.06
```

**Tiny example:** a 10K in 60:00 → 60 × (5/10)^1.06 = 60 × 0.48 = **28:47**. (Not 30:00, because you'd run a 5K faster per km.)

**Hook:** *"Halve the distance, a bit less than halve the time."*

---

## 🗂️ Everything in one table

| | Plain linear | Ridge | Lasso | Bayesian | Random forest | Gradient boosting |
|---|---|---|---|---|---|---|
| Everyday version | best line | line with a rubber band | line that drops weak features | line + how sure | crowd of trees, averaged | trees fixing each other |
| Weights → 0? | no | no (shrinks) | **yes** | no | n/a | n/a |
| Exact formula? | yes | yes | no | yes (Gaussian) | no | no |
| Uncertainty? | no | no | no | **yes** | rough | no |
| Little data? | ok | good | good | **best** | weak | weak |
| Non-linear patterns? | no | no | no | no | **yes** | **yes** |
| Used in | – | `run-ml` | notebook | **phone app** | `run-ml` | `run-ml` |

---

## 💼 What interviews actually expect

**Short answer: no, they don't expect you to derive all this from memory.** For most data analyst / data scientist interviews, they want to see that you **understand the ideas and know when to use what**. Formulas are a bonus that shows depth.

| Level | What they expect |
|---|---|
| **Must know** | Overfitting, bias–variance, cross-validation, ridge vs lasso in plain words, forest vs boosting |
| **Should know** | Why lasso gives zeros (diamond vs circle), a simple Bayes problem (medical test), how to choose λ |
| **Nice to have** (ML-heavy roles) | Ridge = Gaussian prior, lasso = Laplace prior; ridge closed form |
| **Rarely asked** | Full Bayesian posterior derivation by hand |

**What matters most:**
1. **Explain simply.** "Lasso sets some weights to zero, so it also picks features" beats a formula.
2. **Know the trade-offs:** when to choose which.
3. **Talk about a real project:** *"I built a 5K predictor on my own running data. With only a few races I used Bayesian regression with a prior, so it starts sensible and personalises as data comes in. For a bigger dataset I compared ridge, random forest and gradient boosting with time-series cross-validation."*

---

## 📝 Practice questions with answers
Cover the answer, say yours out loud in about 30 seconds, then check.

**1. What's the difference between L1 and L2 regularisation?**
> L2 (ridge) adds λΣw², which shrinks all weights smoothly, none to zero. L1 (lasso) adds λΣ|w|, which pushes some weights to exactly zero, so it also does feature selection. Use L2 when many features each matter a bit, L1 when you think only a few matter or want a simpler model.

**2. Why does lasso produce sparse models (lots of zeros)?**
> Because |w| has a sharp corner at 0. In the picture, the L1 constraint is a diamond with corners on the axes, and the error contours usually touch it first at a corner, where some weights are 0. In one dimension: lasso *subtracts* λ/2 and stops at 0, while ridge *divides* by (Σx² + λ) and never reaches 0.

**3. What happens if λ is too large? Too small?**
> Too large: weights are crushed toward 0, so the model is too simple and underfits (high bias). Too small: it's basically plain regression, which can fit noise and overfit (high variance), especially with few rows or many features.

**4. How would you choose λ?**
> Cross-validation: try a range of λ values (usually on a log scale like 0.01 … 1000) and pick the one with the lowest error on held-out data. For data over time, use a time-series split so you always train on the past and test on the future. In scikit-learn: `RidgeCV`, `LassoCV`.

**5. What's a prior? Give an example.**
> What you believe before seeing the data, including how unsure you are. Example: before I have any of your race data, I assume an extra hour of sleep saves about 8 s on a 5K, give or take 12 s. As your races come in, that belief updates into a posterior.

**6. A test is 99% accurate and the disease affects 1 in 1,000 people. You test positive. What's the chance you have it?**
> About **9%**. Of 1,000 people, 1 is sick and tests positive, and about 10 healthy people also test positive by mistake (1% of 999). So 1 of ~11 positives is real. Formula: 0.99 × 0.001 / (0.99 × 0.001 + 0.01 × 0.999) ≈ 0.09. The lesson: when the prior is tiny, even a good test gives mostly false alarms.

**7. Why might you prefer a Bayesian model with very little data?**
> Because the prior keeps it sensible when data is scarce. Plain regression on 2 points can give wild weights, while a Bayesian model starts from reasonable values and moves toward the data as evidence grows. It also gives a range, which tells you how much to trust the prediction. That's why the phone app uses it.

**8. Two features are highly correlated. What happens with lasso vs ridge?**
> Ridge splits the weight between them fairly evenly and stays stable. Lasso tends to keep one and drop the other, somewhat arbitrarily, and which one it keeps can flip with small data changes. Elastic net (L1 + L2) gets lasso's selection with ridge's stability.

**9. Random forest vs gradient boosting?**
> Both use decision trees. A random forest trains many deep trees independently on random samples and **averages** them, which reduces variance; it's robust and easy to tune. Gradient boosting trains small trees **one after another**, each fitting the errors of the previous ones, which reduces bias; it's often more accurate but easier to overfit and needs tuning (learning rate, number of trees, depth).

**10. Why not just shuffle the data for cross-validation in your running project?**
> Because the runs are ordered in time. Shuffling would let the model train on future runs and test on past ones, so it would "know" fitness it hadn't built yet, and the score would look better than real life. A time-series split always trains on the past and tests on the future.

**11. In the medical test, what's the prior, likelihood and posterior?**
> Prior = P(sick) = 0.001. Likelihood = P(positive | sick) = 0.99. Posterior = P(sick | positive) ≈ 0.09.

**12. Your model has a low training error but a high test error. What's going on and what do you do?**
> Overfitting. Fixes: more regularisation (raise λ), fewer or simpler features, more data, a simpler model (shallower trees, fewer boosting rounds), and check with cross-validation that the test error actually comes down.
