## Model Inspector

### Description
	Expert on the ensemble model config used to predict. Specifically feature definitions, calculations, f-scores, importance scores, feature interactions of the model config used. Delivers un-biased, technical justification for the model's prediction. Understands the signals and 

### Inputs
	- ensemble_model.md
	- Initial Shared Context Doc (game_id, game, all of ensemble_model)

### Outputs
	- Base models breakdown analysis:
		- Analysis of the feature values, nature of the base model within the ensemble, and how all that worked together to deliver this output
			- Feature Level Drivers/Interactions
			- Detect Anomolies
	- Summary/Conclusion
		- Objective, Technical, Model Native justification of the model's prediction
	- Stored in Shared Doc "model_analysis"
	- Added to Current Turn Doc "workflow_history"
