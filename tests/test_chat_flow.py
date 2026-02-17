#!/usr/bin/env python3
"""
Test script to replicate the chatbot message flow for debugging.

This script replicates the exact code flow that happens when a chat message
is sent through the web interface, allowing debugging of issues like the
"no games" error.

Usage:
    python tests/test_chat_flow.py
"""

import sys
import os
from bson import ObjectId
from datetime import datetime

# Add parent directory to path (the directory containing nba_app folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)  # This is the nba_app folder
parent_of_nba_app = os.path.dirname(nba_app_dir)  # Directory containing nba_app
if parent_of_nba_app not in sys.path:
    sys.path.insert(0, parent_of_nba_app)

# Now imports with nba_app prefix should work
from bball_app.core.mongo import Mongo
from bball_app.agents.modeler.modeler_agent import ModelerAgent
from langchain_core.messages import HumanMessage, AIMessage


def test_chat_flow(session_id: str, test_message: str = None):
    """
    Replicate the chatbot message flow for a given session.
    
    Args:
        session_id: MongoDB session ID to test
        test_message: Optional message to send. If None, will use the last user message
                     from the session (simulating a resend scenario)
    """
    print(f"Testing chat flow for session: {session_id}")
    print("=" * 80)
    
    # Initialize MongoDB connection
    mongo = Mongo()
    db = mongo.db
    
    # Verify session exists
    try:
        session_obj_id = ObjectId(session_id)
    except Exception as e:
        print(f"ERROR: Invalid session_id format: {e}")
        return
    
    session_doc = db.chat_sessions.find_one({'_id': session_obj_id})
    if not session_doc:
        print(f"ERROR: Session {session_id} not found in database")
        return
    
    print(f"Found session: {session_doc.get('name', 'Unnamed')}")
    print(f"Session has {len(session_doc.get('messages', []))} messages")
    print()
    
    # Load conversation history from MongoDB (same as web/app.py)
    conversation_history = []
    if session_doc and 'messages' in session_doc:
        print("Loading conversation history...")
        for msg in session_doc['messages']:
            if msg['role'] == 'user':
                conversation_history.append(HumanMessage(content=msg['content']))
                print(f"  User: {msg['content'][:100]}...")
            elif msg['role'] == 'assistant':
                conversation_history.append(AIMessage(content=msg['content']))
                print(f"  Assistant: {msg['content'][:100]}...")
    print()
    
    # Determine test message
    if test_message is None:
        # Find the last user message (to simulate resending)
        last_user_msg = None
        for msg in reversed(session_doc.get('messages', [])):
            if msg['role'] == 'user':
                last_user_msg = msg['content']
                break
        
        if last_user_msg:
            test_message = last_user_msg
            print(f"Using last user message from session:")
        else:
            print("ERROR: No user messages found in session. Please provide test_message.")
            return
    else:
        print(f"Using provided test message:")
    
    print(f"  {test_message}")
    print()
    print("=" * 80)
    print("Initializing ModelerAgent...")
    print("=" * 80)
    
    # Create agent (same as web/app.py)
    try:
        agent = ModelerAgent(session_id=session_id, db=db)
        print("Agent initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 80)
    print("DIRECT TOOL TEST: Calling build_dataset directly...")
    print("=" * 80)
    print()
    
    # Directly call build_dataset tool to replicate the error
    # This bypasses the LangChain agent and calls the tool directly
    from bball_app.core.training.dataset_builder import DatasetBuilder
    
    dataset_builder = DatasetBuilder(db=db)
    
    # Try to extract the actual dataset_spec from the most recent failed run
    # Look for experiment runs for this session
    recent_runs = list(db.experiment_runs.find(
        {'session_id': session_id},
        sort=[('created_at', -1)],
        limit=1
    ))
    
    dataset_spec = None
    if recent_runs:
        run = recent_runs[0]
        config = run.get('config', {})
        if 'splits' in config and 'features' in config:
            # Extract from experiment config
            dataset_spec = {
                'feature_blocks': config['features'].get('blocks', []),
                'individual_features': config['features'].get('features', None),
                'begin_year': config['splits'].get('begin_year'),
                'end_year': config['splits'].get('end_year'),
                'begin_date': config['splits'].get('begin_date'),
                'end_date': config['splits'].get('end_date'),
                'min_games_played': config['splits'].get('min_games_played', 15),
                'exclude_preseason': True,
                'include_per': config['features'].get('include_per', False),
                'diff_mode': config['features'].get('diff_mode', 'home_minus_away')
            }
            print(f"Extracted dataset_spec from run: {run.get('run_id')}")
    
    # Fallback: Use default spec based on error message
    if not dataset_spec:
        print("No recent run found, using default dataset_spec based on error message")
        dataset_spec = {
            'feature_blocks': ['basic_stats', 'advanced_stats'],  # Common feature blocks
            'individual_features': None,  # Will use feature_blocks
            'begin_year': 2022,
            'end_year': None,
            'begin_date': None,
            'end_date': None,
            'min_games_played': 15,
            'exclude_preseason': True,
            'include_per': False,  # Based on logs: "PER features disabled"
            'diff_mode': 'home_minus_away'
        }
    
    print("Dataset spec:")
    for key, value in dataset_spec.items():
        print(f"  {key}: {value}")
    print()
    print("Calling build_dataset()...")
    print()
    
    try:
        dataset_result = dataset_builder.build_dataset(dataset_spec)
        print("SUCCESS!")
        print(f"Dataset ID: {dataset_result.get('dataset_id')}")
        print(f"Row count: {dataset_result.get('row_count')}")
        print(f"Feature count: {dataset_result.get('feature_count')}")
        print(f"CSV path: {dataset_result.get('csv_path')}")
    except Exception as e:
        print("=" * 80)
        print("ERROR IN build_dataset():")
        print("=" * 80)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        # Print additional context if it's a ValueError about no training data
        if isinstance(e, ValueError) and "No training data generated" in str(e):
            print()
            print("=" * 80)
            print("DEBUGGING INFO FOR 'NO TRAINING DATA' ERROR:")
            print("=" * 80)
            print("This error typically occurs when:")
            print("  1. min_games_filter is too restrictive (teams haven't played enough games)")
            print("  2. games_played features are not being calculated correctly")
            print("  3. Date/season mismatches in queries")
            print()
            print("Check the logs above for:")
            print("  - 'Filtered:' messages showing games_played values")
            print("  - 'get_team_games_before_date' warnings about 0 games")
            print("  - Query diagnostics in the error message")
    
    print()
    print("=" * 80)
    print("OPTIONAL: Testing agent.chat() for comparison...")
    print("=" * 80)
    print("(This will show the full agent flow, but the direct tool call above")
    print(" should be sufficient for debugging the build_dataset issue)")
    print()
    
    # Optionally also test the full agent flow
    try:
        result = agent.chat(test_message, conversation_history=conversation_history)
        
        print("=" * 80)
        print("AGENT RESULT:")
        print("=" * 80)
        print(f"Response: {result.get('response', 'No response')[:500]}...")
        print()
        print(f"Run IDs: {result.get('run_ids', [])}")
        print(f"Runs this request: {result.get('runs_this_request', 0)}")
        print(f"Baseline run ID: {result.get('baseline_run_id')}")
        print(f"Run budget remaining: {result.get('run_budget_remaining', 0)}")
        
        tool_calls = result.get('tool_calls', [])
        if tool_calls:
            print(f"\nTool calls ({len(tool_calls)}):")
            for i, tc in enumerate(tool_calls, 1):
                print(f"  {i}. {tc.get('name', 'unknown')}")
                print(f"     Input: {tc.get('input', {})}")
        
    except Exception as e:
        print("=" * 80)
        print("ERROR IN agent.chat():")
        print("=" * 80)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Session ID from user
    SESSION_ID = "695210a7bc572333f06b1075"
    
    # Optional: provide a specific test message
    # If None, will use the last user message from the session
    TEST_MESSAGE = None
    
    # Allow override via command line
    if len(sys.argv) > 1:
        SESSION_ID = sys.argv[1]
    if len(sys.argv) > 2:
        TEST_MESSAGE = sys.argv[2]
    
    test_chat_flow(SESSION_ID, TEST_MESSAGE)

