import os

# IMPORTANT:
# - Do not hardcode secrets in this repo.
# - Set these via environment variables instead.
#
# Required env vars:
# - MONGO_CONN_STR (or MONGODB_URI)
# - OPENAI_API_KEY
config = {
    "mongo_conn_str": os.environ.get("MONGO_CONN_STR") or os.environ.get("MONGODB_URI", ""),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "KALSHI_API_KEY": os.environ.get("KALSHI_API_KEY", ""),
    "KALSHI_PRIVATE_KEY_DIR": os.environ.get("KALSHI_PRIVATE_KEY_DIR", ""),
    "KALSHI_READ_KEY": os.environ.get("KALSHI_READ_KEY", ""),
    "KALSHI_READ_PRIVATE_KEY_DIR": os.environ.get("KALSHI_READ_PRIVATE_KEY_DIR", ""),
    "STAKE_TOKEN": os.environ.get("STAKE_TOKEN", ""),
    "SERP_API_KEY": os.environ.get("SERP_API_KEY", ""),
    "BASKETBALL_APP_URL": os.environ.get("BASKETBALL_HOST", ""),
    "FOOTBALL_APP_URL": os.environ.get("FOOTBALL_HOST", ""),
    "SOCCER_APP_URL": os.environ.get("SOCCER_HOST", ""),
}

