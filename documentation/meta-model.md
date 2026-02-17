# TEAM + FORM META-MODEL PLAN

## Contents
  - Base Model 1: LR1 - team-season
  - Base Model 2: GB2 - team-form
  - Experiments

## LR1 - team-season
elo|none|raw|diff
margin|season|avg|diff
off_rtg_net|season|avg|diff
def_rtg|season|avg|diff
efg_net|season|avg|diff
turnovers|season|avg|diff
margin|season|std|diff
efg_net|season|std|diff
turnovers|season|std|diff

-- can try with just home/away and just diff --
pace|season|raw|home
pace|season|raw|away

-- try with and without these |side  (NO OTHERS) --
margin|season|avg|diff|side
off_rtg_net|season|avg|diff|side

## GB2 - team-form
### Block A — Current form level (diff)
Short window:
margin|games_5|avg|diff
off_rtg_net|games_5|avg|diff
efg_net|games_5|avg|diff
turnovers|games_5|avg|diff
def_rtg|games_5|avg|diff
pace|games_5|raw|diff

Medium window:
margin|games_10|avg|diff
off_rtg_net|games_10|avg|diff
efg_net|games_10|avg|diff
turnovers|games_10|avg|diff
def_rtg|games_10|avg|diff
pace|games_10|raw|diff

### Block B — Blended form level (diff)
(Blend to reduce “hard cutoff” noise.)

margin|blend:games_5:0.70/games_10:0.30|avg|diff
off_rtg_net|blend:games_5:0.70/games_10:0.30|avg|diff
efg_net|blend:games_5:0.70/games_10:0.30|avg|diff
turnovers|blend:games_5:0.70/games_10:0.30|avg|diff
def_rtg|blend:games_5:0.70/games_10:0.30|avg|diff
pace|blend:games_5:0.70/games_10:0.30|raw|diff


### Block C — Change vs baseline (delta features)
Use “recent minus season” delta (diff):

margin|delta:games_5-season|avg|diff
off_rtg_net|delta:games_5-season|avg|diff
efg_net|delta:games_5-season|avg|diff
turnovers|delta:games_5-season|avg|diff
def_rtg|delta:games_5-season|avg|diff
pace|delta:games_5-season|raw|diff

“blend minus season”:
margin|delta:blend:games_5:0.70/games_10:0.30-season|avg|diff
efg_net|delta:blend:games_5:0.70/games_10:0.30-season|avg|diff
turnovers|delta:blend:games_5:0.70/games_10:0.30-season|avg|diff

### Block D — Volatility / instability (std dev)

Short window volatility:
margin|games_5|std|diff
efg_net|games_5|std|diff
turnovers|games_5|std|diff

Medium window volatility:
margin|games_10|std|diff
efg_net|games_10|std|diff
turnovers|games_10|std|diff

Blended volatility (optional; GB can learn from both short+medium without explicit blend, but this can help).

Run the entire ensemble with and without these to see which is better
margin|blend:games_5:0.70/games_10:0.30|std|diff
efg_net|blend:games_5:0.70/games_10:0.30|std|diff
turnovers|blend:games_5:0.70/games_10:0.30|std|diff

### Block E — Minimal absolute context (home/away)
pace|blend:games_5:0.70/games_10:0.30|raw|home
pace|blend:games_5:0.70/games_10:0.30|raw|away
efg_net|blend:games_5:0.70/games_10:0.30|avg|home
efg_net|blend:games_5:0.70/games_10:0.30|avg|away


## Experiments
Train these 4 versions and compare not just standalone logloss/Brier, but **stacking lift**:
### B2-A (small + strong)
* blended avg diffs for: margin/off_rtg/efg/to/def/pace
* delta vs season for: margin/efg/to/off_rtg
* std diffs for: margin/efg/to (games_5)

### B2-B (add medium window stability)
B2-A plus:
* games_10 std diffs for margin/efg/to

### B2-C (add limited absolutes)
B2-B plus:
* blended pace home/away
* blended efg home/away

### B2-D (optional venue split test)
B2-C plus:
* `margin|games_5|avg|diff|side`
* `efg_net|games_5|avg|diff|side`

