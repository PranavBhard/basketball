# Shared Context Doc

## Description
    Entire Conversation history and game/prediction info.

## Where
    Instantiated on game chat creation. Stored in mongo `{league}_agent_shared_context` (ID'd by game_id)

## Structure
```json
{
  // Game Info
  "game_id":"<game_id>",
  "game": {
    "away":{
      "name": "MIL",
      "full_name": "{teams_collection}.displayName",
      "team_id": "<>"
    },
    "home":{
      "name": "MIL",
      "full_name": "{teams_collection}.displayName",
      "team_id": "<>"
    },
    "date":"2026-01-29"
  },

  // Ensemble Model's Game Prediction
  "ensemble_model": {
    "p_home": 0.41,
    // B1, B2, etc are actual base db model config "name"
    "base_outputs": {
      "B1":0.374,
      "B2":0.320,
      "B3":0.591,
      "B4":0.558,
      "B5":0.494,
      "B6":0.562
    },
    "prediction_info": {
      // pulled from model predictions collection for this game -- stored on core prediction run
    }
  },

  // Agent History during this conversation
  "history": [
    {
      "agent": "<agent_name>",
      "system": "<system_message>",
      "messages": [], // additional messages?
      "tools": [
        {
          "name":"",
          "args": {},
          "output": ""
        },
        // ... more tool calls
      ],
      "output": "",
      "timestamp": "<utc_time>"
    }
  ],

  "latest_by_agent": {
    "<agent_name>": "<last_output>"
  }

