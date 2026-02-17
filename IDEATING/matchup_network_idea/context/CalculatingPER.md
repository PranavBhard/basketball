# Calculating PER
**Calculating PER** — an explanation of the Player Efficiency Rating. ([Basketball Reference][1])

---

### **2. Introduction**

A description of **Player Efficiency Rating (PER)**:

* PER is a **per-minute player rating** developed by **John Hollinger**.
* It aims to **summarize positive and negative box score contributions** into one number.
* League-average PER is adjusted to **15.00** for comparability. ([Basketball Reference][1])

---

### **3. Definition & Purpose of PER**

This section explains what PER attempts to measure:

* Combines positive stats (points, assists, rebounds, steals, blocks, made shots) and negative stats (missed shots, turnovers, fouls).
* Produces a per-minute efficiency value.
* Mentions its strengths (compares across minutes, combines contributions) and limitations (heavier offensive weight, defensive nuance issues). ([Basketball Reference][1])

---

### **4. The PER Formula**

The page shows the **unadjusted PER (uPER)** formula — the core computation before league and pace adjustments:

```
uPER = (1 / MP) *
     [ 3P
     + (2/3) * AST
     + (2 - factor * (team_AST / team_FG)) * FG
     + (FT *0.5 * (1 + (1 - (team_AST / team_FG)) + (2/3) * (team_AST / team_FG)))
     - VOP * TOV
     - VOP * DRB% * (FGA - FG)
     - VOP * 0.44 * (0.44 + (0.56 * DRB%)) * (FTA - FT)
     + VOP * (1 - DRB%) * (TRB - ORB)
     + VOP * DRB% * ORB
     + VOP * STL
     + VOP * DRB% * BLK
     - PF * ( … league PF adjustment … )
     ]
```

Where components include:

* **factor**: team AST/FG adjustment
* **VOP**: value of possession (league pace factor)
* **DRB%**: defensive rebound percentage
* Terms reflect aggregated counting stats normalized per minute. ([Basketball Reference][1])

---

### **5. Explanation of Terms**

* Definitions for:

  * **factor**
  * **VOP**
  * **DRB%**
  * Player stats (3P, FG, AST, FT, etc.)
  * Team and league totals (tmAST, lgFG, lgFT, etc.)
* This section defines each component so you can compute them from box score data. ([Basketball Reference][1])

---

### **6. Pace Adjustment & Normalization**

After uPER is computed:

```
PER = (uPER × (league pace / team pace)) × (15 / league uPER average)
```

This step:

* **adjusts for pace** (faster teams give more opportunities),
* and **normalizes the rating** so that league average remains ~15.00. ([Wikipedia][2])

---

### **7. Limitations & Notes**

The page may include notes such as:

* **Historical limitations** — e.g., prior to full stat tracking (turnovers, steals, etc.) earlier seasons lack complete data.
* Cautions on interpreting PER defensively. ([Basketball Reference][1])

---

### **8. (Possible) Examples or Historical Context**

Some context on how far back Basketball-Reference has expanded PER beyond Hollinger’s original database (back to early 1950s). ([Basketball Reference][1])

---

## **Statistical Notation Covered**

The page defines the following *raw data components* that you could compute directly from raw box scores:

| Abbreviation | Meaning                                                   |
| ------------ | --------------------------------------------------------- |
| MP           | Minutes Played                                            |
| 3P           | 3-point field goals made                                  |
| FG           | Field goals made                                          |
| AST          | Assists                                                   |
| FT           | Free throws made                                          |
| TOV          | Turnovers                                                 |
| TRB          | Total rebounds                                            |
| ORB          | Offensive rebounds                                        |
| STL          | Steals                                                    |
| BLK          | Blocks                                                    |
| PF           | Personal fouls                                            |
| team_/lg_    | Team or League aggregated totals                          |
| VOP          | Value of possession based on league offense/defense stats |
| DRB%         | Defensive rebound percentage                              |

All of these are computable from standard box score aggregates. ([Basketball Reference][1])

---

## **Summary of What You’d See on the Page**

1. **Intro/Definition**
2. **PER concept explanation**
3. **Full uPER formula**
4. **Term definitions**
5. **Pace adjustment & normalization**
6. **Historical/limitations notes**
7. **Examples or context for computation**

---

If you want, I can rewrite the full PER formula in a *step-by-step computational guide* (with pseudocode or spreadsheet formulas) so you can compute it from raw box score data yourself. Just let me know!

[1]: https://www.basketball-reference.com/about/per.html?utm_source=chatgpt.com "Calculating PER"
[2]: https://en.wikipedia.org/wiki/Player_efficiency_rating?utm_source=chatgpt.com "Player efficiency rating"
