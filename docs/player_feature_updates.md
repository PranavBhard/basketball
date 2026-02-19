# Player Filter Factory for Feature Calcs:

## DEFAULT CONSTANTS
    MPG_THRESH = 10
    GP_FLOOR = 10
    GP_RATIO = .15
    GP_THRESH = max(
        GP_FLOOR,
        ceil(GP_RATIO * team_games_played_to_date)
    )
    EPS = 1e-6
    ROTATION_SIZE = 10

## PLAYER SUBSET DEFINITION (FOR A GIVEN TEAM (home or away) FOR A GIVEN GAME):
    {ROSTER} =
        mode="train":
            stats.min>0 for (game_id, player_id) instance in nba's player stats collection
        mode="predict":
            (team, season) instance in nba_rosters

    {LOW_MPG} =
        set of player_ids who have played < `MPG_THRESH` MPG this season to date (not including game date)

    {LOW_GP} =
        p for p in {ROSTER} if p.games_played_this_season < `GP_THRESH`
    
    {LOW_USAGE} =
        LOW_MPG ∪ LOW_GP

    {USAGE_PLAYERS} =
        {ROSTER} - {LOW_USAGE}

    {INJ_PLAYERS} =
        mode="train":
            player_ids in stats_nba instance's "{home/away}Team.injured_players" list
        mode="predict":
            set of player_ids in (team, season) instance in nba_rosters with "injured"=true

    {ACTIVE_PLAYERS} =
        {ROSTER} - {INJ_PLAYERS}

    {STARTERS} =
        mode="train":
            player_ids in stats_nba_players instance's where "starter"=True
        mode="predict":
            set of player_ids in (team, season) instance in nba_rosters with "starter"=true

    index = min ( len({ACTIVE_PLAYERS}-{LOW_MPG}-{LOW_GP}), ROTATION_SIZE )
    {ROTATION} =
        [{ACTIVE_PLAYERS}-{LOW_MPG}-{LOW_GP}].sort(player.MPG, -1)[0:index]

    {BENCH} =
        {ROTATION} - {STARTERS}

    star_score = player.MPG * player.PER
    {STAR} = 
        {USAGE_PLAYERS}.sort(star_score, -1)[0]

SIGNALS I WANT TO CAPTURE:

## Active Player Talent

**Starters Talent**
- player_starter_per|season|avg|*
- players involved in calcs: {STARTERS}

**Bench Talent**
- player_bench_per|season|weighted_MPG|* (not recency weighted -- straight MPG weight)
  player_bench_per|season|weighted_MIN_REC(k=50)|*
  player_bench_per|season|weighted_MIN_REC(k=45)|*
  player_bench_per|season|weighted_MIN_REC(k=40)|*
  player_bench_per|season|weighted_MIN_REC(k=35)|*
- players involved in calcs: {BENCH}
- calc (weighted_MPG):

    Σ_p [ PER(p) × MIN(p) ]
    --------------------------------
    Σ_p [ MIN(p) ]

- calc (weighted_MIN_REC):
    nums = []
    denoms = []
    For each player in BENCH:
        For each game by player this season (stats.min>0):
            weight_g = exp(-days_since_game / k)
            nums.append(PER_g(p) * MIN_g(p) * weight_g)
            denoms.append(MIN_g(p) * weight_g)
    final = sum(nums)/[sum(denoms) + EPS]
    
    Σ_p Σ_g [ PER_g(p) × MIN_g(p) × weight_g ]
    --------------------------------
    Σ_p Σ_g [ MIN_g(p) × weight_g ] + EPS

**Rotation Talent -- Weighted by Usage (MPG)**
- player_rotation_per|season|weighted_MPG|* (not recency weighted -- straight MPG weight)
  player_rotation_per|season|weighted_MIN_REC(k=20)|*
  player_rotation_per|season|weighted_MIN_REC(k=25)|*
  player_rotation_per|season|weighted_MIN_REC(k=30)|*
  player_rotation_per|season|weighted_MIN_REC(k=35)|*
- players involved in calcs: {ROTATION}
- calc (weighted_MPG):

    Σ_p [ PER(p) × MIN(p) ]
    --------------------------------
    Σ_p [ MIN(p) ]

- calc (weighted_MIN_REC):
    nums = []
    denoms = []
    For each player in ROTATION:
        For each game by player this season (stats.min>0):
            weight_g = exp(-days_since_game / k)
            nums.append(PER_g(p) * MIN_g(p) * weight_g)
            denoms.append(MIN_g(p) * weight_g)
    final = sum(nums)/[sum(denoms) + EPS]

    Σ_p Σ_g [ PER_g(p) × MIN_g(p) × weight_g ]
    --------------------------------
    Σ_p Σ_g [ MIN_g(p) × weight_g ] + EPS

**Starter/Bench Gap**
- player_starter_bench_per_gap|season|derived(k=50)|*
- players involved: {ROTATION}
- calc: player_starter_per|season|avg|* - player_bench_per|season|weighted_MIN_REC(k=50)|*
    *note: include k=45,40,35 versions -- k matches weighted_MIN_REC derived from*

**Top Active Star**
- player_star_score|season|top1|* (not recency weighted -- straight season PER x MPG)
  player_star_score|season|weighted_MIN_REC(k=20)|*
- players involved in calcs: {ROTATION}
- calc:
    player_scores = []
    for player in ROTATION:
        nums = []
        denoms = []
        for game in player_games_this_season:  # MIN_g > 0
            STAR_g = PER_g * MIN_g
            w = exp(-days_since_game / k)
            nums.append(STAR_g * w)
            denoms.append(w)
        star_p = sum(nums) / (sum(denoms) + EPS)   # recency-weighted STAR per game
        player_scores.append({"player_id": player_id, "star_p": star_p})
    top1_player = sorted(player_scores, key=lambda x: x["star_p"], reverse=True)[0]

    *note: include k=25,30,35 versions of weighted_MIN_REC*

**Top 3 Active Stars**
- player_star_score|season|top3_avg|* (not recency weighted -- straight season PER x MPG average)
  player_star_score|season|top3_sum|* (not recency weighted -- straight season PER x MPG sum)
  player_star_score|season|top3_weighted_MIN_REC(k=20)|*
  player_star_score|season|top3_weighted_MIN_REC(k=25)|*
  player_star_score|season|top3_weighted_MIN_REC(k=30)|*
  player_star_score|season|top3_weighted_MIN_REC(k=35)|*
- players involved in calcs: {ROTATION}
- calc:
    player_scores = []
    for player in ROTATION:
        nums = []
        denoms = []
        for game in player_games_this_season:  # MIN_g > 0
            STAR_g = PER_g * MIN_g
            w = exp(-days_since_game / k)
            nums.append(STAR_g * w)
            denoms.append(w)
        star_p = sum(nums) / (sum(denoms) + EPS)   # recency-weighted STAR per game
        player_scores.append({"player_id": player_id, "star_p": star_p})
    top3_players = sorted(player_scores, key=lambda x: x["star_p"], reverse=True)[:3]
    top3_avg = sum(p["star_p"] for p in top3_players) / 3.0
    top3_sum = sum(p["star_p"] for p in top3_players)

**Top Active Star Share**
- player_star_share|season|top1_share|* (not recency weighted -- straight season PER x MPG)
  player_star_share|season|top1_share_weighted_MIN_REC(k=20)|*
- players involved in calcs: {ROTATION}
- calc:
    player_scores = []
    for player in ROTATION:
        nums = []
        denoms = []
        for game in player_games_this_season:  # MIN_g > 0
            STAR_g = PER_g * MIN_g
            w = exp(-days_since_game / k)
            nums.append(STAR_g * w)
            denoms.append(w)
        star_p = sum(nums) / (sum(denoms) + EPS)   # recency-weighted STAR per game
        player_scores.append({"player_id": player_id, "star_p": star_p})

    # top active (top1) by recency-weighted STAR per game
    top1 = sorted(player_scores, key=lambda x: x["star_p"], reverse=True)[0]["star_p"]

    # share of top1 vs total active star mass (same units)
    total = sum(p["star_p"] for p in player_scores)
    top1_share = top1 / (total + EPS)

    *note: include k=25,30,35 versions of weighted_MIN_REC*


**Top 3 Active Star Shares**
- player_star_share|season|top3_share|* (not recency weighted -- straight season PER x MPG)
  player_star_share|season|top3_share_weighted_MIN_REC(k=20)|*
- players involved in calcs: {ROTATION}
- calc:
    player_scores = []
    for player in ROTATION:
        nums = []
        denoms = []
        for game in player_games_this_season:  # MIN_g > 0
            STAR_g = PER_g * MIN_g
            w = exp(-days_since_game / k)
            nums.append(STAR_g * w)
            denoms.append(w)
        star_p = sum(nums) / (sum(denoms) + EPS)   # recency-weighted STAR per game
        player_scores.append({"player_id": player_id, "star_p": star_p})

    # top3 by recency-weighted STAR per game
    top3 = sorted(player_scores, key=lambda x: x["star_p"], reverse=True)[:3]
    top3_sum = sum(p["star_p"] for p in top3)

    # share of top3 vs total active star mass (same units)
    total = sum(p["star_p"] for p in player_scores)
    top3_share = top3_sum / (total + EPS)

    *note: include k=25,30,35 versions of weighted_MIN_REC*

**Top Sufficient Usage Roster Star**
- player_star_score_all|season|top1|*
- players involved in calcs: {USAGE_PLAYERS}
*Top usage minimum roster player star scores this season (includes injured in sorting)*

**Depth/Playable**
- player_rotation_count|season|raw|*
- players: {ROTATION}
- calc: len({ROTATION})

**Continuity / cohesion proxy (active minutes together)**
- player_continuity|season|avg|*
- players: {ROTATION}
- calc: avg([min(1, p.GP / GP_THRESH) for p in {ROTATION}])
-----

# Injuries

## Injury Impact (minutes / availability)

**Rotation Minutes Lost**
- inj_min_lost|none|raw|*
- players involved in calcs: {INJ_PLAYERS} ∩ {USAGE_PLAYERS}
- calc:
    inj_min_lost = Σ_p [ MPG(p) ]
    where p ∈ ({INJ_PLAYERS} ∩ {USAGE_PLAYERS})

**Rotation Minutes Share Lost**
- inj_severity|none|raw|*
- players involved in calcs: {INJ_PLAYERS} ∩ {USAGE_PLAYERS}
- calc:
    team_rotation_mpg = Σ_p [ MPG(p) ]  where p ∈ {USAGE_PLAYERS}
    inj_severity = inj_min_lost / (team_rotation_mpg + EPS)

## Injury Impact (top-end / “star is missing”)

**Injured Star Share of Top-3 Star Mass**
- inj_star_share|none|raw|*
- players involved in calcs: {STAR}, Top3Team
- calc:
    Top3Team = top3({USAGE_PLAYERS}, key=star_score)

    # share of the team’s top-end star mass represented by the star (0 if star not out)
    inj_star_share =
        1[inj_star_out == 1] *
        ( star_score({STAR}) / (Σ_p∈Top3Team [ star_score(p) ] + EPS) )

**Top-3 Star Mass Missing (Share)**
- inj_star_score_share|none|top3_sum|*
- players involved in calcs: Top3Team, Top3Inj
- calc:
    Top3Team = top3({USAGE_PLAYERS}, key=star_score)
    Top3Inj  = top3(({USAGE_PLAYERS} ∩ {INJ_PLAYERS}), key=star_score)

    inj_star_score_share = clip(
        ( Σ_p∈Top3Inj [ star_score(p) ] ) / ( Σ_p∈Top3Team [ star_score(p) ] + EPS ),
        0.0,
        1.5
    )

**Top-1 Star Missing (Binary)**
- inj_top1_star_out|none|binary|* (bool)
- players involved in calcs: Top1TeamStar
- calc:
    Top1TeamStar = top1({USAGE_PLAYERS}, key=star_score)
    inj_top1_star_out = 1.0 if Top1TeamStar ∈ {INJ_PLAYERS} else 0.0

## Notes / helpers

- star_score(p) = MPG(p) * PER(p)   (season-to-date, using your {LOW_MPG}/{LOW_GP} filters via {USAGE_PLAYERS})
- clip(x, lo, hi) = min(hi, max(lo, x))
- EPS = 1e-6


-----