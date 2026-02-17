# Pre-Game NBA Statistical Narratives

### (Stat-Only, Computable, No Hidden Concepts)

Each section includes:

* **Narrative question**
* **Why it matters**
* **Concrete statistical signals**
* **Example computable metrics / equations**

---

## 1. Scoring Path Concentration

**Narrative:**
*How narrow are each team’s paths to points?*

**Why it matters:**
Teams that rely on one scoring source (one player, one shot type) are fragile.

**Signals to look for:**

* High share of points from top scorer
* High share of shots from one zone
* Low secondary usage

**Metrics:**

* **Top Scorer Point Share**
  [
  \text{TopScorerPts%} = \frac{\max_i(\text{PTS}_i)}{\text{Team PTS}}
  ]

* **Shot Source Concentration**
  [
  \text{3PA%} = \frac{\text{Team 3PA}}{\text{Team FGA}}
  \quad
  \text{Rim%} = \frac{\text{Rim FGA}}{\text{Team FGA}}
  ]

* **Usage Drop-Off**
  [
  \Delta USG = USG_1 - USG_3
  ]

**Pre-game indicator:**
Teams with **high concentration + high variance shot mix** need things to go right.

---

## 2. Shooting Variance Exposure

**Narrative:**
*How sensitive is this matchup to shooting noise?*

**Why it matters:**
Variance-exposed teams swing outcomes disproportionately.

**Signals:**

* High 3PA volume
* High historical shooting variance
* Opponent allows high 3PA

**Metrics:**

* **3P Variance**
  [
  \sigma_{3P} = \text{StdDev}(3P%)_{last\ N}
  ]

* **Effective Volatility Index**
  [
  \text{EVI} = 3PA% \times \sigma_{3P}
  ]

* **Opponent Allowed 3PA**
  [
  \text{Opp 3PA Rate} = \frac{\text{Opp 3PA}}{\text{Opp FGA}}
  ]

**Pre-game indicator:**
High EVI → upset probability increases even if mean efficiency is lower.

---

## 3. Possession Advantage Potential

**Narrative:**
*Who is more likely to create extra possessions?*

**Why it matters:**
Possessions compound quietly and reduce reliance on shooting.

**Signals:**

* Turnover gap
* Offensive rebounding gap
* Free throw attempts per FGA

**Metrics:**

* **Turnover Differential**
  [
  \Delta TO = TO_{opp} - TO_{team}
  ]

* **Offensive Rebound Rate**
  [
  ORB% = \frac{ORB}{ORB + OppDRB}
  ]

* **Possession Edge Score**
  [
  \text{PES} = w_1 \cdot \Delta TO + w_2 \cdot \Delta ORB + w_3 \cdot \Delta FTA/FGA
  ]

**Pre-game indicator:**
Teams with a positive PES can survive poor shooting nights.

---

## 4. Transition Reliance vs Stability

**Narrative:**
*How much scoring requires opponent mistakes?*

**Why it matters:**
Scoring tied to turnovers is volatile and opponent-dependent.

**Signals:**

* Points off turnovers per game
* Turnover-to-points conversion rate
* Opponent turnover rate allowed

**Metrics:**

* **Transition Dependency**
  [
  \text{POT Ratio} = \frac{\text{Points Off TO}}{\text{Total PTS}}
  ]

* **Conversion Efficiency**
  [
  \text{POT / TO} = \frac{\text{Points Off TO}}{TO_{forced}}
  ]

**Pre-game indicator:**
High POT ratio teams have higher ceilings but lower floors.

---

## 5. Free Throw Pressure Asymmetry

**Narrative:**
*Which team generates points without the clock moving?*

**Why it matters:**
FTs stabilize scoring and distort rotations.

**Signals:**

* FTA per FGA
* Opponent foul rate
* Starter foul frequency

**Metrics:**

* **FT Pressure**
  [
  FT/FGA = \frac{FTA}{FGA}
  ]

* **Starter Foul Risk**
  [
  \text{Foul Rate}_{starter} = \frac{PF}{MIN}
  ]

**Pre-game indicator:**
Large FT asymmetries create slow, structural edges.

---

## 6. Star Load & Failure Sensitivity

**Narrative:**
*What happens if the best player is inefficient?*

**Why it matters:**
Usage concentration magnifies efficiency swings.

**Signals:**

* High usage + low assist share
* Large efficiency drop without free throws
* Limited secondary creators

**Metrics:**

* **Star Load Index**
  [
  SLI = USG \times \frac{PTS}{TeamPTS}
  ]

* **Efficiency Dependence**
  [
  \text{Non-FT TS%} = \frac{PTS - FTM}{2(FGA)}
  ]

**Pre-game indicator:**
High SLI teams lose disproportionately when the star is off.

---

## 7. Lineup Stability & Replacement Quality

**Narrative:**
*How robust is the rotation?*

**Why it matters:**
Injuries and foul trouble propagate through lineups.

**Signals:**

* Minutes continuity
* Bench net rating
* Replacement usage shifts

**Metrics:**

* **Bench Net Rating**
  [
  NetRtg_{bench}
  ]

* **Usage Redistribution**
  [
  \Delta USG_{bench} = USG_{bench,new} - USG_{bench,avg}
  ]

* **Lineup Volatility**
  [
  \sigma(\text{NetRtg}_{top\ 5\ lineups})
  ]

**Pre-game indicator:**
High lineup volatility = unpredictable swings.

---

## 8. Rebounding & Physical Edge (Stat-Only)

**Narrative:**
*Who is more likely to extend possessions?*

**Why it matters:**
Second chances reduce efficiency dependence.

**Signals:**

* ORB%
* Defensive rebound leakage
* Size proxy via rebound share

**Metrics:**

* **Rebound Share**
  [
  \text{Reb Share}_i = \frac{REB_i}{Team REB}
  ]

* **Opponent DRB Failure**
  [
  1 - DRB%
  ]

**Pre-game indicator:**
Rebounding edges compound over 90–100 possessions.

---

## 9. Fatigue & Usage Strain

**Narrative:**
*Who is more likely to fade late?*

**Why it matters:**
Fatigue shows up in efficiency, not minutes.

**Signals:**

* Usage × minutes load
* Recent games played
* Efficiency decay over time windows

**Metrics:**

* **Player Load**
  [
  Load_i = USG_i \times MIN_i
  ]

* **Team Load Index**
  [
  TLI = \sum_{top\ 3} Load_i
  ]

* **Recent Efficiency Trend**
  [
  \Delta TS%_{last\ 3 \rightarrow season}
  ]

**Pre-game indicator:**
High TLI teams struggle late unless pace is low.

---

## 10. Late-Game Scoring Profile

**Narrative:**
*Who scores efficiently when possessions matter most?*

**Why it matters:**
Close games hinge on repeatable scoring.

**Signals:**

* Clutch TS%
* Turnovers in final 5 minutes
* Shot quality proxies (FTA, rim attempts)

**Metrics:**

* **Clutch Efficiency**
  [
  TS%_{clutch}
  ]

* **Late TO Rate**
  [
  TO / Possession_{last\ 5}
  ]

**Pre-game indicator:**
Disparities here explain spread vs ML mispricing.

---

## 11. Outcome Fragility Index (Composite)

**Narrative:**
*How many things must go right?*

**Construct:**
[
Fragility =

* w_1(\text{Scoring Concentration})
* w_2(\text{EVI})
* w_3(\text{Star Load})

- w_4(\text{PES})
- w_5(\text{Bench Stability})
  ]

**Interpretation:**

* High fragility = volatile outcomes
* Low fragility = multiple win paths

---

## 12. Model–Market Sanity Signals

**Narrative:**
*Is this disagreement structural or noisy?*

**Signals:**

* Model edge aligns with possession edge → structural
* Model edge relies on shooting → fragile
* Market shaded without stat confirmation → investigate

---

## How to Use This Before Every Game

Ask:

1. Which team has **more ways to score**?
2. Which team needs **variance to win**?
3. Where do **extra possessions** come from?
4. Which side collapses first if efficiency drops?
5. Is the edge **structural or conditional**?

---

### Final Note (Implicit but Critical)

Everything above is:

* Derivable from **raw game + player stats**
* Computable at scale

# Player/personnel specific Pre-Game NBA Statistical Narratives
Absolutely — this is the **player-interaction layer** that most models never fully exploit, and it’s where *lineups, injuries, and usage redistribution* quietly decide games.

Below is a **stat-only, player-level interaction narrative framework**, written so **every concept is computable from raw game logs, player box scores, lineup minutes, and on/off splits**. No scheme labels, no film concepts — just personnel math.

Think of these as **interaction hypotheses** you can test pre-game or post-game.

---

# Player-Level Statistical Narratives

### (Interaction Effects, On/Off Dynamics, Personnel Fragility)

---

## 1. Pair Synergy vs Pair Dependence

**Narrative:**
*Are two players additive, multiplicative, or mutually dependent?*

**Why it matters:**
Some players boost each other. Others only function together.

**Signals:**

* Net rating when both play vs solo
* Usage and efficiency shifts when separated

**Metrics:**

* **Pair Net Rating Lift**
  [
  \Delta Net_{A+B} =
  Net(A \cap B) - \max(Net(A), Net(B))
  ]

* **Dependence Ratio**
  [
  DR_A = \frac{Net(A \cap B)}{Net(A \setminus B)}
  ]

**Interpretation:**

* High lift → synergy
* High dependence → fragility if one misses

---

## 2. Primary–Secondary Creator Interactions

**Narrative:**
*Who benefits from having a creator next to them?*

**Why it matters:**
Efficiency is often borrowed, not intrinsic.

**Signals:**

* TS%, eFG% change when creator present
* Assist share concentration

**Metrics:**

* **Efficiency Borrowing**
  [
  \Delta TS%_B =
  TS%(B | A\ on) - TS%(B | A\ off)
  ]

* **Shot Quality Proxy**
  [
  AST%_B(on/off)
  ]

**Interpretation:**
Large deltas imply player B is context-dependent.

---

## 3. Usage Redistribution Chains (Injury-Critical)

**Narrative:**
*When Player A is out, who absorbs the burden — and at what cost?*

**Why it matters:**
Minutes replacement ≠ role replacement.

**Signals:**

* Usage spikes
* Efficiency decay
* Turnover rate increase

**Metrics:**

* **Usage Absorption**
  [
  \Delta USG_i =
  USG_i(A\ out) - USG_i(A\ in)
  ]

* **Cost of Absorption**
  [
  \Delta TS%_i, \quad \Delta TO%_i
  ]

**Interpretation:**
Teams break when usage flows to inefficient absorbers.

---

## 4. Lineup Anchor Effects

**Narrative:**
*Which player stabilizes lineups regardless of who they play with?*

**Why it matters:**
Anchors reduce variance and protect benches.

**Signals:**

* Low net rating variance across lineups
* Positive on/off regardless of partner quality

**Metrics:**

* **Lineup Stability Index**
  [
  LSI_A = -\sigma(NetRtg_{lineups\ with\ A})
  ]

* **Universal On/Off**
  [
  OnOff_A = Net(A\ on) - Net(A\ off)
  ]

**Interpretation:**
High LSI players smooth rotations and foul trouble.

---

## 5. Offensive Gravity Without Touches

**Narrative:**
*Which players improve teammates without scoring?*

**Why it matters:**
Spacing and attention effects show up indirectly.

**Signals:**

* Teammate TS% increase
* Teammate rim or 3PA increase

**Metrics:**

* **Teammate Efficiency Lift**
  [
  \Delta TS%_{others | A\ on}
  ]

* **Shot Mix Shift**
  [
  \Delta 3PA%, \Delta Rim%
  ]

**Interpretation:**
Gravity players matter even in low-usage games.

---

## 6. Defensive Liability Interactions (Stat-Only)

**Narrative:**
*Which players drag down lineups regardless of scoring?*

**Why it matters:**
Some players’ negative impact is interaction-based.

**Signals:**

* Opponent TS% spike when player is on
* Foul and rebound leakage

**Metrics:**

* **Opponent Efficiency On/Off**
  [
  \Delta TS%_{opp | A\ on/off}
  ]

* **Opponent ORB% Swing**
  [
  \Delta ORB%_{opp}
  ]

**Interpretation:**
Defensive weaknesses amplify when paired poorly.

---

## 7. Bench Connector Effects

**Narrative:**
*Who keeps second units functional?*

**Why it matters:**
Bench collapses lose games quietly.

**Signals:**

* Bench net rating with vs without player
* Assist and turnover stabilization

**Metrics:**

* **Bench On/Off**
  [
  Net_{bench | A\ on} - Net_{bench | A\ off}
  ]

* **Bench Ball Security**
  [
  \Delta TO%_{bench}
  ]

**Interpretation:**
Connector players prevent negative runs.

---

## 8. Foul Magnet & Foul Shield Interactions

**Narrative:**
*Which players create or absorb foul pressure for others?*

**Why it matters:**
Foul trouble propagates across rotations.

**Signals:**

* FTA rate changes
* Starter foul rate changes

**Metrics:**

* **Foul Draw Lift**
  [
  \Delta FTA_{team | A\ on}
  ]

* **Foul Shield Effect**
  [
  \Delta PF_{teammates | A\ on}
  ]

**Interpretation:**
Some players protect teammates by absorbing contact.

---

## 9. Rebounding Complementarity

**Narrative:**
*Do players overlap or complement on the glass?*

**Why it matters:**
Redundant rebounders waste value.

**Signals:**

* Rebound share overlap
* Team ORB% with pair vs solo

**Metrics:**

* **Rebound Overlap**
  [
  Overlap_{A,B} =
  RebShare_A + RebShare_B - RebShare_{A+B}
  ]

* **Incremental ORB Gain**
  [
  \Delta ORB%_{A+B}
  ]

**Interpretation:**
Complementary rebounders unlock possession edges.

---

## 10. Late-Game Trust Networks

**Narrative:**
*Who is on the floor together when games tighten?*

**Why it matters:**
Clutch minutes reveal coaching confidence.

**Signals:**

* Co-minutes in last 5 minutes
* Usage consolidation

**Metrics:**

* **Clutch Pair Minutes**
  [
  MIN_{A+B}^{clutch}
  ]

* **Clutch Usage Share**
  [
  USG_{A+B}^{clutch}
  ]

**Interpretation:**
High trust pairs drive late outcomes.

---

## 11. Replacement Quality Asymmetry (Injury Edge)

**Narrative:**
*Is the drop-off linear or catastrophic?*

**Why it matters:**
Two teams can lose equal minutes with unequal damage.

**Signals:**

* Replacement net rating
* Efficiency drop per 10 minutes lost

**Metrics:**

* **Replacement Gap**
  [
  RG_A =
  Net(A\ on) - Net(replacement)
  ]

* **Damage Rate**
  [
  RG_A / MIN_A
  ]

**Interpretation:**
High RG players are non-linear injury risks.

---

## 12. Player Interaction Volatility Index

**Narrative:**
*How unstable is this team based on who’s available?*

**Composite:**
[
Volatility =
\sigma(NetRtg_{top\ lineups})

* \sum DR_{critical\ pairs}
* \sum RG_{star}
  ]

**Interpretation:**

* High volatility → outcomes swing on availability
* Low volatility → resilient roster

---

## 13. Model-Facing Interaction Narratives (For You)

High-value diagnostics:

* Model likes team, but **pair dependence is extreme**
* Injury flagged neutral, but **usage absorption is inefficient**
* Market ignores **bench connector absence**
* Spread inflated despite **high interaction volatility**

---

## How to Use This Practically

Before a game:

1. Identify **2–3 critical pairs**
2. Check **on/off net, usage, efficiency**
3. Ask: *If one piece fails, what breaks first?*

After a game:

* Attribute swings to **interaction failure**, not player failure

---

### Final Thought

Player interaction narratives are where:

* Injuries become nonlinear
* Bench minutes decide spreads
* “Same roster” ≠ same team