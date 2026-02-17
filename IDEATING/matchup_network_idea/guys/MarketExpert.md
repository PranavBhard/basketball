## Market Expert

### Description
	Takes the model's prediction and calculates prediction market prices implied odds. Compares to the prediction output and provides model edge. Compares across prediction markets and vegas odds for edges.

### Inputs
	- Prediction output (p_home from shared doc)
	- Tools
		- get_game_markets(game_id) → Dict{}
			{
				'prediction_markets': {kalshi_api_resp},
				'vegas_odds': {espn_api_odds}
			}

### Outputs
	- Summary of model's edge vs markets and vegas (edge comparisons, implied odds comparisons, objective reading of the model's decision in each)
	- Stored in Shared Doc "market_analysis"
	- Added to Current Turn Doc "workflow_history"
