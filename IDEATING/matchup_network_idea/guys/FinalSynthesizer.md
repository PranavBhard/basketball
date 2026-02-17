## Final Synthesizer

### Description
	Takes the full Shared Context, this turn's workflow context, and the Planner's synthesis instructions and output a final response to the user.

### System
	Take in the conversation, shared context, turn workflow, Planner's instructions and synthesize a final response following this turn's workflow and turn_plan.

### Inputs
	- Conversation thus far (alternating user and final synthesizer response messages)
	- Shared Context Doc
	- Current Turn Doc (including final_synthesis_instructions)

### Outputs
	- Synthesizes findings as instructed
	- Saved in mongo chat session