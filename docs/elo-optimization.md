## Plan to optimize Elo for NBA, CBB, WCBB, WNBA (neutral sites + MOV available)

### Objectives

1. Improve **probability calibration** (log loss / Brier) while maintaining accuracy.
2. Make Elo behavior **league-appropriate** via tuned **home advantage**, **dynamic K**, **margin-of-victory (MOV) scaling**, and **season-phase overrides** (neutral / postseason).
3. Keep YAML configs explicit so the Elo engine can be driven fully by league config.

---

## 1) Update YAML schema (apply to all leagues)

Replace your current `elo:` block with this richer structure:

```yaml
elo:
  starting_rating: 1500

  # Win probability mapping: expected win prob from Elo diff.
  # Typical Elo formula uses diff/scale (scale ~ 400).
  scale: 400

  # Home advantage (in Elo points) applied only when NOT neutral.
  home_advantage: 100
  neutral_site_home_advantage: 0

  # K strategy: static uses k_factor; dynamic uses k_schedule by games played.
  strategy: dynamic
  k_factor: 20
  k_schedule:
    - {max_games: 10, k: 35}
    - {max_games: 20, k: 25}
    - {default: 18}

  # Margin of Victory scaling (enabled since you compute MOV).
  # Update magnitude is multiplied by a factor derived from MOV; cap limits blowout leverage.
  use_margin: true
  margin_multiplier:
    type: mov_log          # recommended: log-based multiplier
    cap: 12                # cap points used for multiplier input
    scale: 2.2             # controls how much MOV amplifies updates (tune per league)

  # Season phase adjustments (assumes you can tag phase: regular/postseason).
  postseason:
    enabled: true
    home_advantage: 80     # optional override vs regular season
    k_multiplier: 0.85     # damp updates in postseason if desired

  # Neutral-site handling (assumes you have neutral flag).
  neutral_site:
    enabled: true
    home_advantage: 0      # neutral-site HA override

  # Optional: offseason regression toward mean at season boundary (high ROI for CBB/WCBB).
  offseason_regression:
    enabled: true
    carryover_weight: 0.75   # retain this portion of prior rating
    to_mean_weight: 0.25     # remainder pulls toward league mean (1500)
```

**Notes on interpretation (for implementation):**

* **Expected probability** uses Elo diff adjusted by home advantage (or neutral override) and mapped via `scale`.
* **Update size** uses K (from `k_schedule`) × (result − expected) × MOV multiplier (if enabled).
* **Neutral site**: if neutral flag true, apply `neutral_site.home_advantage` (typically 0).
* **Postseason**: if phase is postseason, optionally override home advantage and multiply K by `postseason.k_multiplier`.
* **Offseason regression**: at season start, set `rating = carryover_weight * prior_rating + to_mean_weight * 1500`.

---

## 2) League-specific YAML values (drop-in configs)

### NBA (more stable; smaller HA; lower K; MOV influence modest)

```yaml
elo:
  starting_rating: 1500
  scale: 400

  home_advantage: 60
  neutral_site_home_advantage: 0

  strategy: dynamic
  k_schedule:
    - {max_games: 20, k: 18}
    - {max_games: 50, k: 15}
    - {default: 12}

  use_margin: true
  margin_multiplier:
    type: mov_log
    cap: 15
    scale: 1.8

  postseason:
    enabled: true
    home_advantage: 55
    k_multiplier: 0.85

  neutral_site:
    enabled: true
    home_advantage: 0

  offseason_regression:
    enabled: true
    carryover_weight: 0.85
    to_mean_weight: 0.15
```

### WNBA (shorter season; slightly higher K than NBA; HA similar/moderate; MOV moderate)

```yaml
elo:
  starting_rating: 1500
  scale: 400

  home_advantage: 65
  neutral_site_home_advantage: 0

  strategy: dynamic
  k_schedule:
    - {max_games: 12, k: 24}
    - {max_games: 24, k: 18}
    - {default: 14}

  use_margin: true
  margin_multiplier:
    type: mov_log
    cap: 15
    scale: 2.0

  postseason:
    enabled: true
    home_advantage: 60
    k_multiplier: 0.85

  neutral_site:
    enabled: true
    home_advantage: 0

  offseason_regression:
    enabled: true
    carryover_weight: 0.80
    to_mean_weight: 0.20
```

### CBB (high variance; strong HA; high early K; MOV influence stronger; postseason/neutral important)

```yaml
elo:
  starting_rating: 1500
  scale: 400

  home_advantage: 105
  neutral_site_home_advantage: 0

  strategy: dynamic
  k_schedule:
    - {max_games: 10, k: 38}
    - {max_games: 20, k: 26}
    - {default: 18}

  use_margin: true
  margin_multiplier:
    type: mov_log
    cap: 12
    scale: 2.4

  postseason:
    enabled: true
    home_advantage: 80
    k_multiplier: 0.90

  neutral_site:
    enabled: true
    home_advantage: 0

  offseason_regression:
    enabled: true
    carryover_weight: 0.65
    to_mean_weight: 0.35
```

### WCBB (even more top-heavy/volatile; strong HA; very high early K; MOV influence strong; offseason regression strongest)

```yaml
elo:
  starting_rating: 1500
  scale: 400

  home_advantage: 110
  neutral_site_home_advantage: 0

  strategy: dynamic
  k_schedule:
    - {max_games: 10, k: 42}
    - {max_games: 20, k: 28}
    - {default: 18}

  use_margin: true
  margin_multiplier:
    type: mov_log
    cap: 14
    scale: 2.6

  postseason:
    enabled: true
    home_advantage: 85
    k_multiplier: 0.90

  neutral_site:
    enabled: true
    home_advantage: 0

  offseason_regression:
    enabled: true
    carryover_weight: 0.60
    to_mean_weight: 0.40
```

---

## 3) Elo calculation explanation (what these YAML keys mean operationally)

For a game between home team H and away team A:

1. **Compute effective Elo diff**

* Start with `diff = Elo_H − Elo_A`.
* If `neutral_site.enabled` and game is neutral: add `neutral_site.home_advantage` (usually 0).
* Else add `home_advantage` (regular season) or `postseason.home_advantage` (postseason override).

2. **Convert Elo diff to expected win probability**

* Use `scale` (e.g., 400): larger scale makes probabilities less extreme.
* Expected: `p_home = f(diff / scale)` (standard Elo logistic mapping).

3. **Compute result term**

* `result = 1` if home wins, `0` if home loses.

4. **Determine K for this team/game**

* If `strategy: static`, use `k_factor`.
* If `strategy: dynamic`, choose K using `k_schedule` based on games played so far for the team (or an agreed convention like min games played between teams; whichever your system standardizes).
* If postseason and `postseason.enabled`, multiply by `postseason.k_multiplier`.

5. **Apply MOV multiplier (since MOV is computed)**

* If `use_margin: true`:

  * Clamp margin input at `cap`.
  * Compute a multiplier using `type: mov_log` and `scale`.
  * This multiplier increases update size for larger, more informative wins (bounded by cap).

6. **Update ratings**

* `delta = K * MOV_multiplier * (result − p_home)`
* `Elo_H += delta`
* `Elo_A -= delta`

7. **Offseason regression** (at season boundary)

* If enabled:

  * `Elo_new = carryover_weight * Elo_old + to_mean_weight * 1500`

---

## 4) Tuning workflow (per league, using your existing time-split)

For each league independently:

1. Tune **home_advantage** around the suggested default (±10 for NBA/WNBA; ±15–20 for CBB/WCBB).
2. Tune **early K** and **default K** in `k_schedule`.
3. Tune MOV `cap` and `scale`.
4. Tune offseason `carryover_weight` (biggest lever for CBB/WCBB).
   Measure by **eval-year log loss / Brier** and optionally early-season vs late-season splits.

This plan is fully YAML-driven and should be directly implementable by a codebase agent.
