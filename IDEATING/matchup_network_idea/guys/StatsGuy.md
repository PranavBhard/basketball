## Stats Guy

### Description
	Gets list of active players, list of injured players. Provide report of who is playing, who is out, stats that tell stories based on teams and players playing (i.e. "when this star player is out the team is W-L vs when he plays they are W-L"). Good at creative reasoning. Turns stats into narratives well.

### Inputs
	- Shared Context "game_id", "game"
	- statistical_narratives.md
	- For each team:
		- Active players (starters, bench) (id  -> name map for each team's players)
		- Injuries/Inactives (player id -> name map of all players marked "injured" used in game prediction feature calculations for this game/team)

	- List of tools
		* get_lineups(team_id) → {
				"starters": [{id, name, pos}, ...],
				"bench": [{id, name, pos}, ...],
				"injured": [{id, name, pos}, ...]
			}
			* do we need this? can we just give this agent the info programatically because starters and injuries come from the rosters collection for predictions (needs those same players that were used for the prediction)
		- get_game(game_id)
			→ list of raw db game doc
		- get_team_games(team_id, window, split=None)
			→ List[Dict{}, ...]
				- list of raw game docs over time period before this game (not including this game) [optional split filter]
		- get_player_stats(player_id, window, split=None)
			→ List[Dict{}, ...]
				- list of raw player stat docs over time period
		- get_advanced_player_stats(player_id, window, split=None)
			→ List[Dict{}, ...]
				- calculated player stat docs over time period (PER, MPG, GP/GS, APG, RPG, etc.)
		- get_h2h(teamA, teamB, window, split=None)
			→ List[Dict{}, ...]
				- list of raw game docs over time period
		- get_travel(team_id, window, split)
			→ List[Dict{}, ...]
				- list of raw game docs over time period

		*tool args*
			window -> "days10", "games10", "season", etc
			optional split:
				- "home": just home games
				- "away": just away games
				- None: both home and away games


### Outputs
	- Report on which team has the edge in each of the base model categories with stats backed justifications for why (without giving it model results). Skilled at aggregating a statistical story of the matchup (across categories):
		- Team over the course of the season
			- can aggregate records (season, "last 10 games", home, away, etc), ppg, margins
		- Team in the last 10-12 games
		- Travel/Rest
		- Active Players
		- Injured Players
		- Head To Head
			- aggregate records, margins, etc for games these two have played this season.
	- Provides a final analysis based on its own stats research
	- Stored in Shared Doc "stats_narratives"
	- Added to Current Turn Doc "workflow_history"
	- Use statistical narratives for ideas, dont let it limit you
