# Current Turn Doc

## Descriptoin
    Current turn context only -- not the whole conversation

## Where
    Instantiated on each user message. Stored in flask session storage/cache (ID'd by game_id)

## Structure
```json
{
  "user_message": "",
  "turn_plan": {
    "narrative": "The user is asking about ... in the context of ... for this game... The final answer should ...",
    "workflow": [
        {
          "agent": "",
          "instruction": ""
        },
        // ...
    ],
    "final_synthesis_instructions": ""
  },
  "workflow_history": [
      {
        "agent": "",
        "output": ""
      }
  ],
}
```