Below is a **full, objective, third-person rewrite** that incorporates the clarified training / calibration regime and the LR meta-model details. Every claim is tied directly to the configuration and observed numbers; no speculative narratives are introduced.

---

## Overview of the Ensemble Architecture

The model is an **informed stacking ensemble** designed to predict NBA game outcomes. It consists of:

* **Six Gradient Boosting (GB) base classifiers**, each trained on a distinct conceptual slice of game information:

  1. Team season-long strength (B1)
  2. Recent team form (B2)
  3. Rest and travel (B3)
  4. Player talent (B4)
  5. Player injuries / availability (B5)
  6. Head-to-head matchup history (B6V2)

* A **logistic regression (LR) meta-model** with:

  * Regularization strength **C = 0.1** (strong L2 regularization),
  * Trained exclusively on **out-of-sample calibrated base predictions**.

This design enforces a clear separation between:

* *signal extraction* (handled by GB models), and
* *signal weighting and reconciliation* (handled by a constrained linear meta-model).

---

## Temporal Training, Calibration, and Evaluation Regime

The ensemble follows a strictly forward-looking, leakage-controlled timeline:

### Base Model Training

* Each GB base model is trained on historical NBA data from:

  * **2010–2011 through 2020–2021** seasons.

### Base Model Calibration

* Each base model's raw outputs are **time-based calibrated using sigmoid calibration** on:

  * **2021–2022**
  * **2022–2023**
  * **2023–2024**

This calibration step ensures:

* Output probabilities are aligned with observed frequencies,
* Systematic over- or under-confidence from tree ensembles is corrected,
* Base predictions become directly comparable across signal families.

### Meta-Model Training

* The LR meta-model is trained **only** on:

  * The calibrated outputs of the base models from the **2021–2022 through 2023–2024** seasons.
* No base model training data (2010–2021) is used at the meta level.

### Final Evaluation

* The complete stacked system is evaluated on:

  * **2024–2025**, which is fully unseen by both base training and meta training.

This pipeline ensures:

* The meta-model learns *how base signals behave in modern NBA contexts*,
* Without learning spurious patterns tied to older eras or raw model overconfidence.

---

## Role and Behavior of the Logistic Regression Meta-Model

### Structural Implications of LR (C = 0.1)

Using a strongly regularized LR as the meta-model imposes several constraints:

1. **No nonlinear feature interactions** are learned at the ensemble level.
2. Each base model contributes through a **single scalar coefficient**.
3. The meta-model cannot "memorize" contexts; it can only:

   * Reweight base probabilities,
   * Exploit systematic disagreement patterns across bases.

As a result:

* Any performance gain must come from **complementarity between base models**, not from overfitting.

---

## Meta-Model Coefficients and Their Implications

### Learned Meta Weights

| Base Signal               | Meta Weight |
| ------------------------- | ----------- |
| B1 – Team Season Strength | **1.785**   |
| B4 – Player Talent        | **1.645**   |
| B5 – Player Injuries      | **1.565**   |
| B2 – Team Form            | **1.453**   |
| B6V2 – Head-to-Head       | **1.036**   |
| B3 – Rest & Travel        | **0.780**   |
| pred_margin               | **0.005**   |

### Interpretation of Coefficient Magnitudes

Because:

* All base outputs are **sigmoid-calibrated probabilities**, and
* LR coefficients operate directly in log-odds space,

the relative coefficient magnitudes can be interpreted as **trust weights**.

#### Key Observations

1. **`pred_margin` is functionally unused**

   * Its coefficient (0.005) is orders of magnitude smaller than any base model.
   * The ensemble is therefore *not* hybridizing classification with a point-spread signal.
   * The final prediction is driven almost entirely by probabilistic base models.

2. **Team-level and roster-level signals dominate**

   * B1 (team season strength), B4 (player talent), and B5 (injuries) form the top tier.
   * Together they represent a majority of the ensemble's effective weight.

3. **Rest and travel are intentionally down-weighted**

   * B3's lower coefficient reflects limited global reliability,
   * Not irrelevance (see later discussion).

---

## Performance Outcomes and What They Imply

### Observed Metrics (2024–2025)

* **Accuracy:** 74.86%
* **Log Loss:** 0.516
* **Brier Score:** 0.170

### Comparison to Base Models

| Model                   | Standalone Accuracy |
| ----------------------- | ------------------- |
| Best Base (B4 – Talent) | ~65.8%              |
| Ensemble                | **74.9%**           |

The magnitude of this lift is significant and implies:

* Base models make **systematically different errors**,
* The LR meta-model is exploiting **conditional reliability**, not averaging noise.

Because LR cannot form nonlinear interactions, this improvement must come from:

* Certain bases being consistently more reliable in specific regions of the probability space learned during calibration seasons.

---

## Signal-Level Analysis (Grounded in Observed Importances)

### B1 — Team Season Strength

**Standalone Accuracy:** 64.7% | **40 features**

This base functions as the ensemble's **baseline prior**: a stable, slow-moving estimate of team quality. Its top meta weight indicates that even after accounting for form, injuries, and talent, season-long team strength remains the most reliable single anchor.

**Key Features (by importance):**

| Feature | Description |
|---------|-------------|
| `elo\|none\|raw\|home` | Elo rating: dynamic power ranking updated after each game. Higher = stronger team. (home team) |
| `margin\|season\|avg\|home` | Point differential (home_score - away_score). Key predictor of team strength. Full season average. (home team) |
| `elo\|none\|raw\|away` | Elo rating: dynamic power ranking updated after each game. Higher = stronger team. (away team) |
| `margin\|season\|avg\|away` | Point differential. Full season average. (away team) |
| `off_rtg_net\|season\|avg\|away` | Net offensive rating: team off_rtg - opponent def_rtg. Overall efficiency edge. Full season. (away team) |
| `margin\|season\|avg\|home\|side` | Point differential filtered to games where team played same home/away role. |
| `off_rtg_net\|season\|avg\|home` | Net offensive rating. Full season average. (home team) |
| `margin\|season\|avg\|away\|side` | Point differential filtered to same home/away role games. (away team) |
| `off_rtg_net\|season\|avg\|home\|side` | Net offensive rating filtered to same role games. (home team) |
| `wins\|season\|avg\|away` | Win rate. Full season. (away team) |

Home/away-filtered variants carry non-trivial importance, indicating role-specific performance adds information beyond raw season averages.

---

### B2 — Recent Team Form

**Standalone Accuracy:** 62.9% | **40 features**

The meta-model's substantial weight on B2 confirms recent performance adds independent information beyond season baselines.

**Key Features (by importance):**

| Feature | Description |
|---------|-------------|
| `points_net\|games_12\|avg\|home` | Point differential (team points - opponent points). Last 12 games rolling average. (home team) |
| `off_rtg_net\|games_12\|avg\|away` | Net offensive rating. Last 12 games rolling average. (away team) |
| `points_net\|games_12\|avg\|away` | Point differential. Last 12 games rolling average. (away team) |
| `off_rtg_net\|games_12\|avg\|home` | Net offensive rating. Last 12 games rolling average. (home team) |
| `off_rtg_net\|games_12\|avg\|home\|side` | Net offensive rating filtered to same role games. Last 12 games. (home team) |
| `off_rtg_net\|games_12\|avg\|away\|side` | Net offensive rating filtered to same role games. Last 12 games. (away team) |
| `close_win_pct\|season\|avg\|home` | Win percentage in games decided by ≤5 points. Clutch performance. Full season. (home team) |
| `close_win_pct\|season\|avg\|away` | Win percentage in games decided by ≤5 points. Full season. (away team) |
| `points_net\|games_12\|avg\|home\|side` | Point differential filtered to same role games. Last 12 games. (home team) |
| `turnovers\|games_12\|avg\|away` | Turnovers committed. Lower is better. Last 12 games. (away team) |

The model operationalizes "form" as **recent dominance**, not recent results. Margin-based features provide more predictive structure than win–loss records.

---

### B3 — Rest and Travel

**Standalone Accuracy:** 54.6% | **6 features**

Rest and travel effects are **rare but decisive**, producing large information gain in a small subset of games. The LR meta-model retains this base at a reduced weight, allowing these rare signals to influence predictions when they occur without degrading overall calibration.

**Features:**

| Feature | Description |
|---------|-------------|
| `travel\|days_5\|avg\|away` | Travel distance in miles to game location. Last 5 days. (away team) — **highest importance** |
| `travel\|days_5\|avg\|home` | Travel distance in miles to game location. Last 5 days. (home team) |
| `days_rest\|days_5\|raw\|away` | Days of rest before game. More rest generally helps. Last 5 days. (away team) |
| `days_rest\|days_5\|raw\|home` | Days of rest before game. Last 5 days. (home team) |
| `b2b\|none\|raw\|home` | Back-to-back indicator: 1 if playing second game in consecutive days. (home team) |
| `b2b\|none\|raw\|away` | Back-to-back indicator. (away team) |

Internal pattern: extremely high importance but extremely low split frequency for travel distance indicates these effects are rare but decisive.

---

### B4 — Player Talent

**Standalone Accuracy:** 65.8% | **36 features**

The ensemble's talent signal is fundamentally **rotation-centric**, not star-centric. The quality of minutes likely to be played matters more than top-end peak talent alone. The alignment between highest standalone accuracy and second-highest meta weight supports this signal as a core driver.

**Key Features (by importance):**

| Feature | Description |
|---------|-------------|
| `player_rotation_per\|season\|weighted_MIN_REC(k=35)\|home` | Minute-weighted PER for rotation players (top 10 active by MPG) with recency decay k=35. (home team) |
| `player_rotation_per\|season\|weighted_MIN_REC(k=35)\|away` | Minute-weighted PER for rotation players with recency decay k=35. (away team) |
| `player_team_per\|season\|weighted_MPG\|away` | Team-level PER weighted by minutes played per game. Overall roster talent. (away team) |
| `player_rotation_per\|season\|weighted_MIN_REC(k=30)\|away` | Rotation PER with recency decay k=30. (away team) |
| `player_starter_per\|season\|avg\|home` | Average PER for starting players. Full season. (home team) |
| `player_star_score\|season\|top3_sum\|home` | Star score (PER × MIN) sum of top 3. Weighted talent concentration. (home team) |
| `player_star_score\|season\|top3_sum\|away` | Star score sum of top 3. (away team) |
| `player_rotation_per\|season\|weighted_MIN_REC(k=20)\|home` | Rotation PER with recency decay k=20. (home team) |
| `player_star_score\|season\|top3_avg\|away` | Star score top 3 average. (away team) |
| `player_starters_per\|season\|avg\|away` | Starting lineup average PER (top 5 by starts/MPG). (away team) |

Single-star metrics exist but are secondary.

---

### B5 — Player Injuries and Availability

**Standalone Accuracy:** 61.7% | **22 features**

Despite middling standalone accuracy, B5 receives a high meta weight. This implies injury information is **not universally predictive**, but when relevant, it provides **non-redundant corrective power** relative to team strength and form. The LR meta-model has learned to systematically rely on this signal without allowing it to dominate indiscriminately.

**Key Features (by importance):**

| Feature | Description |
|---------|-------------|
| `inj_min_lost\|none\|raw\|home` | Total expected minutes lost to injury (sum of injured players' MPG). (home team) |
| `inj_severity\|season\|raw\|away` | Proportion of rotation minutes lost: inj_min_lost / team_rotation_mpg. Full season. (away team) |
| `inj_impact\|none\|blend:severity:0.45/top1_per:0.35/rotation:0.20\|away` | Injury impact blend: 0.45×severity + 0.35×top1_per + 0.20×rotation. Overall injury burden. (away team) |
| `inj_severity\|none\|raw\|away` | Proportion of rotation minutes lost. (away team) |
| `inj_severity\|none\|raw\|home` | Proportion of rotation minutes lost. (home team) |
| `inj_star_score_share\|none\|top3_sum\|away` | Top-3 injured star mass as share of team top-3. Clipped to [0, 1.5]. (away team) |
| `inj_impact\|none\|blend:severity:0.45/top1_per:0.35/rotation:0.20\|home` | Injury impact blend. (home team) |
| `inj_severity\|season\|raw\|home` | Proportion of rotation minutes lost. Full season. (home team) |
| `inj_per_weighted_share\|none\|weighted_MIN\|away` | Normalized weighted PER lost. (away team) |
| `inj_min_lost\|none\|raw\|away` | Total expected minutes lost to injury. (away team) |

Binary "top star out" indicators have negligible importance.

---

### B6V2 — Head-to-Head Matchups

**Standalone Accuracy:** 61.8% | **10 features**

The model treats matchup history as informative, but only insofar as dominance is demonstrated and sample size supports it. The moderate meta weight reflects controlled trust in matchup effects, consistent with their situational relevance in the NBA.

**Key Features (by importance):**

| Feature | Description |
|---------|-------------|
| `margin_h2h\|last_3\|eb\|diff` | Point margin in H2H games. Last 3 games (cross-season). Empirical Bayes shrinkage. (home - away) |
| `margin_h2h\|last_5\|eb\|diff` | Point margin in H2H games. Last 5 games. Empirical Bayes shrinkage. (home - away) |
| `margin_h2h\|last_5\|avg\|diff` | Point margin in H2H games. Last 5 games. Per-game average. (home - away) |
| `margin_h2h\|season\|logw\|diff` | Point margin in H2H games. Full season. Log-weighted recency. (home - away) |
| `margin_h2h\|last_5\|logw\|diff` | Point margin in H2H games. Last 5 games. Log-weighted recency. (home - away) |
| `margin_h2h\|last_5\|logw\|diff\|side` | Point margin in H2H filtered to same home/away role. Last 5 games. Log-weighted. |
| `margin_h2h\|season\|eb\|diff` | Point margin in H2H games. Full season. Empirical Bayes shrinkage. (home - away) |
| `margin_h2h\|last_3\|logw\|diff` | Point margin in H2H games. Last 3 games. Log-weighted recency. (home - away) |
| `h2h_win_pct\|last_5\|beta\|diff` | H2H win percentage. Last 5 games. Beta-prior-smoothed toward 0.5. (home - away) |
| `h2h_games_count\|season\|raw\|none` | Number of H2H games played. Sample size for H2H features. Full season. |

Dominated almost entirely by margin-based features with multiple smoothing schemes (raw, log-weighted, EB). Sample-size awareness is explicitly encoded.

---

## Integrated Interpretation

Given the architecture, calibration strategy, and learned coefficients, the ensemble behaves as follows:

1. **Establishes a baseline expectation** using season-long team quality.
2. **Adjusts for current reality** via recent form and rotation-level talent.
3. **Applies availability corrections** when injuries materially alter expected lineups.
4. **Incorporates matchup memory** when supported by sufficient historical dominance.
5. **Allows rare schedule stressors** (travel/rest) to influence outcomes selectively.
6. **Rejects point-margin shortcuts**, relying instead on probabilistic consensus.

All of these behaviors are directly evidenced by:

* The meta-model's constrained linear structure,
* The magnitude and ordering of coefficients,
* And the internal feature importance distributions of the base models.
