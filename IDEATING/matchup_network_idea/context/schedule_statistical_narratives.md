# Rest/Schedule/Fatigue stats
Below is a **stat-only, computable narrative framework** for **travel, rest, density, and road-sequence effects**, written so *everything* can be derived from:

* game dates
* locations (home/away)
* basic geography (team city lat/lon)
* minutes, usage, efficiency splits

No qualitative assumptions, no scheme talk — just calendar + miles + box scores.

---

# Travel / Schedule / Rest Statistical Narratives

### (Stat-Only, Computable, Compounding Fatigue Signals)

---

## 1. Schedule Density Load

**Narrative:**
*How compressed has the calendar been?*

**Why it matters:**
Fatigue accumulates non-linearly with density.

**Signals:**

* Games played in last X days
* Rest days between games
* Clusters (B2B, 3-in-4, 5-in-7)

**Metrics:**

* **Games-in-Days**
  [
  G_{X} = #\text{games in last } X \text{ days}
  ]

* **Rest Days**
  [
  Rest = Date_{current} - Date_{previous\ game}
  ]

* **Density Index**
  [
  DI = \sum_{i=1}^{G_X} \frac{1}{Rest_i + 1}
  ]

**Interpretation:**
Higher DI → higher late-game and shooting-efficiency decay risk.

---

## 2. Back-to-Back (B2B) Stress Profiles

**Narrative:**
*Is this a “hard” or “soft” B2B?*

**Why it matters:**
Not all B2Bs are equal.

**Signals:**

* Home→Away vs Away→Away
* Travel miles between games
* Minutes played night 1

**Metrics:**

* **B2B Flag**
  [
  B2B = \mathbb{1}(Rest = 0)
  ]

* **B2B Severity**
  [
  B2B_{sev} = Miles_{prev \rightarrow current} \times AvgMIN_{top3}
  ]

**Interpretation:**
High-severity B2Bs correlate with Q4 efficiency drops.

---

## 3. Travel Distance Accumulation

**Narrative:**
*How far has this team actually moved recently?*

**Why it matters:**
Travel fatigue compounds independently of rest.

**Signals:**

* Miles traveled in last X games
* Direction changes
* Consecutive flights

**Metrics:**

* **Game-to-Game Distance**
  [
  Miles_i = Haversine(City_{i-1}, City_i)
  ]

* **Rolling Travel Load**
  [
  TTL_X = \sum_{i=1}^{X} Miles_i
  ]

* **Travel Intensity**
  [
  TI = \frac{TTL_X}{G_X}
  ]

**Interpretation:**
High TI → sluggish starts, late fouls, reduced rebounding.

---

## 4. Road Trip Compression

**Narrative:**
*Is this a long road stretch without recovery?*

**Why it matters:**
Road fatigue is cumulative even without B2Bs.

**Signals:**

* Consecutive road games
* Time since last home game
* Road net rating decay

**Metrics:**

* **Road Streak Length**
  [
  RS = #\text{consecutive away games}
  ]

* **Days Away From Home**
  [
  DaysAway = Date_{current} - Date_{last\ home}
  ]

* **Road Performance Drift**
  [
  \Delta NetRtg_{road} = NetRtg_{recent\ road} - NetRtg_{season\ road}
  ]

**Interpretation:**
Long road stretches create hidden fragility even with “rest”.

---

## 5. Rest Asymmetry (Opponent Comparison)

**Narrative:**
*Who is coming in fresher — quantitatively?*

**Why it matters:**
Relative fatigue matters more than absolute fatigue.

**Signals:**

* Rest day differential
* Travel load differential
* Minutes load differential

**Metrics:**

* **Rest Differential**
  [
  \Delta Rest = Rest_{team} - Rest_{opp}
  ]

* **Travel Differential**
  [
  \Delta TTL = TTL_{team} - TTL_{opp}
  ]

* **Fatigue Edge Score**
  [
  FES = w_1\Delta Rest - w_2\Delta TTL - w_3\Delta Load
  ]

**Interpretation:**
Negative FES teams underperform spreads late.

---

## 6. Minutes × Schedule Interaction

**Narrative:**
*Who actually absorbs the fatigue?*

**Why it matters:**
Fatigue shows up through *usage-weighted minutes*.

**Signals:**

* High-usage players with heavy recent minutes
* Short rotations

**Metrics:**

* **Player Load**
  [
  Load_i = USG_i \times MIN_i
  ]

* **Team Fatigue Load**
  [
  TFL = \sum_{top3} Load_i \times DI
  ]

**Interpretation:**
Teams with high TFL fade faster than teams with equal minutes but lower usage.

---

## 7. Performance Decay Patterns

**Narrative:**
*Does this team historically decay under stress?*

**Why it matters:**
Some teams manage fatigue better.

**Signals:**

* Efficiency drop on short rest
* Shooting splits by rest category

**Metrics:**

* **TS% Decay**
  [
  \Delta TS%_{short\ rest} = TS%(Rest \le 1) - TS%(Rest \ge 2)
  ]

* **Rebound/Foul Drift**
  [
  \Delta ORB%, \Delta PF
  ]

**Interpretation:**
Decay-prone teams are exploitable in dense stretches.

---

## 8. Start vs Finish Asymmetry

**Narrative:**
*Where does fatigue show up — early or late?*

**Why it matters:**
Different betting angles (1H vs full game).

**Signals:**

* Q1 vs Q4 net rating splits
* Late turnover and foul rate

**Metrics:**

* **Late Fade Index**
  [
  LFI = NetRtg_{Q4} - NetRtg_{Q1}
  ]

* **Clutch Efficiency Drop**
  [
  \Delta TS%_{clutch}
  ]

**Interpretation:**
High LFI teams are poor late-game closers under fatigue.

---

## 9. Altitude / Return-to-Home Effects (Stat-Only)

**Narrative:**
*Is this a physiological bounce or drain spot?*

**Why it matters:**
Certain travel sequences distort performance.

**Signals:**

* Road → home return
* Long trip end points
* Home efficiency spikes after travel

**Metrics:**

* **Return Game Flag**
  [
  ReturnHome = \mathbb{1}(Away_{prev}=1 \land Home_{current}=1)
  ]

* **Return Boost**
  [
  \Delta TS%_{home\ after\ road}
  ]

**Interpretation:**
Some teams rebound sharply after extended travel.

---

## 10. Schedule Fragility Index (Composite)

**Narrative:**
*How breakable is this team today due to the calendar?*

**Composite Example:**
[
SchedFrag =

* DI
* TTL_X
* RS
* TFL

- Rest
  ]

**Interpretation:**

* High fragility → variance + late collapse risk
* Low fragility → stable execution

---

## 11. Market-Facing Schedule Narratives

**Narrative:**
*Is the market fully pricing this stress?*

**Signals:**

* Line unchanged despite large fatigue asymmetry
* Totals misaligned with pace decay
* Model edge aligns with rest/travel gap

**Interpretation:**
Schedule edges are often **undervalued unless extreme**.

---

## 12. Practical Pre-Game Questions

Before every game:

1. Who has traveled farther in the last 5–7 days?
2. Who has absorbed the minutes under that travel?
3. Is fatigue **absolute** or **relative**?
4. Does this team historically decay under stress?
5. Will fatigue show early or late?

---

### Final Thought

Travel and schedule effects are:

* Structural
* Compounding
* Under-modeled
* And **most visible in Q4 efficiency, fouls, and rebounding**