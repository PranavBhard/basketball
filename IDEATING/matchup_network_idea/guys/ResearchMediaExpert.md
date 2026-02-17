## Research & Media Expert

### Description
	Gets news articles, headlines, market odds, player news, list of active players, list of injured players. Provide report of who is playing, who is out, stats that tell stories (i.e. "when this star player is out the team is W-L vs when he plays they are W-L") --
**Character Comp**
- shams jr protege

### Inputs
	- Home/Away Team ID/Name
	- Stats Guy's Analysis For This Convo
	- For each team:
		- Player news for each team
		- Active players (starters, bench) (id  -> name map for each team's players)
		- Injuries/Inactives (player id -> name map of all players marked "injured" used in game prediction feature calculations for this game/team)

	- List of tools
		- get_lineups(team_id) → {
				"starters": [{id, name, pos}, ...],
				"bench": [{id, name, pos}, ...],
				"inactives": [{id, name, pos}, ...],
			}
		- get_game_news(game_id) → [
			{source, content},
			...
		]
		- get_team_news(team_id) → [
			{source, content},
			...
		]
		- get_player_news(player_id) → [
			{source, content},
			...
		]
		- web_search() → [
			{source, content},
			...
		]
			- documentation/parse_webpage_txt.md explains the web search parsing after using serp api


### Outputs
	- Report on which team has the edge in each of the base model categories with stats backed justifications for why (without giving it model results):
		- Team over the course of the season
			- can aggregate records (season, "last 10 games", home, away, etc), ppg, margins
		- Team in the last 10-12 games
		- Travel/Rest
		- Active Players
		- Injured Players
		- Head To Head
			- aggregate records, margins, etc for games these two have played this season.
	- Stored in Shared Doc "research"
	- Added to Current Turn Doc "workflow_history"
