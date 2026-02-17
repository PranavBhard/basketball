#!/usr/bin/env python
import sys
sys.path.insert(0, '/Users/pranav/Documents/NBA')

import json
from bball_app.core.mongo import Mongo
from bson import json_util

db = Mongo().client.nba

# Get selected model config
config = db.model_config_nba.find_one({'selected': True})

if config:
    print(json.dumps(json.loads(json_util.dumps(config)), indent=2, default=str))
else:
    print("No selected config found")
