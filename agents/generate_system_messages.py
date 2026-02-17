#!/usr/bin/env python3
"""
System Message Generator - Generates agent system messages from templates + core layer data.

This script ensures agent system messages stay in sync with the core layer (SSoT).
It reads template files with placeholders and injects dynamic content from:
- core/feature_sets.py (feature blocks, descriptions, layers)
- core/feature_registry.py (stat definitions, time periods, calc weights)

Usage:
    python agents/generate_system_messages.py

    # Or with specific agent:
    python agents/generate_system_messages.py --agent modeler

    # Check if regeneration is needed (for use in workflows):
    python agents/generate_system_messages.py --check-only

The script reads templates from agents/{agent}/system_message.template.txt
and writes generated output to agents/{agent}/system_message.txt

Auto-regeneration:
    Agent workflows can call `ensure_system_messages_current()` at startup to
    automatically regenerate system messages if core files have changed.
    This uses content hashing (not timestamps) for reliability.
"""

import os
import sys
import json
import hashlib
import argparse
from typing import Dict, List, Optional, Tuple
import re
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bball_app.core.features.registry import FeatureRegistry, FeatureGroups


# =============================================================================
# CONTENT GENERATORS
# =============================================================================

def generate_feature_blocks_list() -> str:
    """
    Generate the feature blocks list for system messages.

    Returns markdown-formatted list of feature blocks with descriptions.
    Uses FeatureGroups (SSoT) from feature_registry.py.
    """
    lines = []

    # Order blocks logically (by layer, then alphabetically within layer)
    layer_order = [1, 2, 3, 4]
    ordered_blocks = []
    seen = set()

    for layer in layer_order:
        for block in FeatureGroups.get_groups_by_layer(layer):
            if block not in seen:
                ordered_blocks.append(block)
                seen.add(block)

    # Add any blocks not in layers
    for block in FeatureGroups.get_all_groups():
        if block not in seen:
            ordered_blocks.append(block)

    for block_name in ordered_blocks:
        desc = FeatureGroups.get_group_description(block_name)
        if desc:
            lines.append(f"- `{block_name}`: {desc}")
        else:
            lines.append(f"- `{block_name}`")

    # Add point_predictions (meta-feature, not in FeatureGroups)
    lines.append("- `point_predictions`: Point prediction model outputs (`pred_margin`, `pred_home_points`, `pred_away_points`, `pred_point_total`) - meta-features representing a points regression model's predictions for game outcomes. `pred_margin` (predicted point differential) is particularly useful as a high-level summary feature.")

    return "\n".join(lines)


def generate_invalid_block_names() -> str:
    """
    Generate list of invalid block name variations to warn against.
    """
    invalid_names = [
        "Pace Volatility", "Schedule Fatigue", "Sample Size",
        "Elo Strength", "Absolute Magnitude", "Pace", "Player Per",
        "Elo Rating", "Injury Impact", "Rebounding", "Turnovers",
        "Points Decomposition"
    ]
    return ", ".join(f'"{name}"' for name in invalid_names)


def generate_valid_stat_names() -> str:
    """
    Generate list of valid stat names from the registry.
    """
    stats = sorted(FeatureRegistry.get_all_stat_names())
    return ", ".join(f"`{s}`" for s in stats)


def generate_valid_time_periods() -> str:
    """
    Generate list of valid time periods.
    """
    periods = sorted(FeatureRegistry.VALID_TIME_PERIODS)
    return ", ".join(f"`{p}`" for p in periods)


def generate_valid_calc_weights() -> str:
    """
    Generate list of valid calc weights.
    """
    weights = sorted(FeatureRegistry.VALID_CALC_WEIGHTS)
    return ", ".join(f"`{w}`" for w in weights)


def generate_valid_perspectives() -> str:
    """
    Generate list of valid perspectives.
    """
    perspectives = sorted(FeatureRegistry.VALID_PERSPECTIVES)
    return ", ".join(f"`{p}`" for p in perspectives)


def generate_layer_info() -> str:
    """
    Generate layer information for system messages.
    Uses FeatureGroups (SSoT) from feature_registry.py.
    """
    layer_descriptions = {
        1: "Outcome & ratings - core strength comparison",
        2: "Tempo, schedule, sample size - situational context",
        3: "Meta priors & normalization - global calibration",
        4: "Player talent & absolute magnitudes - detailed structure",
    }

    lines = []
    for layer_num in [1, 2, 3, 4]:
        groups = FeatureGroups.get_groups_by_layer(layer_num)
        if groups:
            desc = layer_descriptions.get(layer_num, "")
            lines.append(f"- **layer_{layer_num}**: {desc}")
            lines.append(f"  - Sets: {', '.join(f'`{s}`' for s in groups)}")
    return "\n".join(lines)


def generate_feature_count_summary() -> str:
    """
    Generate summary of feature counts per block.
    Uses FeatureGroups (SSoT) from feature_registry.py.
    """
    lines = []
    total = 0
    all_features = FeatureGroups.get_all_features(include_side=True)
    for block_name in FeatureGroups.get_all_groups():
        features = all_features.get(block_name, [])
        count = len(features)
        total += count
        lines.append(f"- `{block_name}`: {count} features")
    lines.append(f"\n**Total: {total} features**")
    return "\n".join(lines)


# =============================================================================
# PLACEHOLDER MAPPING
# =============================================================================

PLACEHOLDERS = {
    "{{FEATURE_BLOCKS_LIST}}": generate_feature_blocks_list,
    "{{INVALID_BLOCK_NAMES}}": generate_invalid_block_names,
    "{{VALID_STAT_NAMES}}": generate_valid_stat_names,
    "{{VALID_TIME_PERIODS}}": generate_valid_time_periods,
    "{{VALID_CALC_WEIGHTS}}": generate_valid_calc_weights,
    "{{VALID_PERSPECTIVES}}": generate_valid_perspectives,
    "{{LAYER_INFO}}": generate_layer_info,
    "{{FEATURE_COUNT_SUMMARY}}": generate_feature_count_summary,
}


# =============================================================================
# CONTENT HASHING & AUTO-REGENERATION
# =============================================================================

# Files that affect system message generation (content-hash based invalidation)
CORE_SOURCE_FILES = [
    'core/feature_sets.py',
    'core/feature_registry.py',
]

# Matchup network sources (templates + docs + referenced AgentNetworkIdea docs)
MATCHUP_NETWORK_SOURCE_GLOBS = [
    'agents/matchup_network/system_messages/*.md',
    'agents/matchup_network/docs/**/*.md',
    'AgentNetworkIdea/ensemble_model.md',
    'AgentNetworkIdea/context/*.md',
]

CACHE_FILE_NAME = '.system_message_cache.json'


def _get_nba_app_dir() -> str:
    """Get the nba_app base directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hash_file(filepath: str) -> str:
    """Compute SHA256 hash of file contents."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _get_source_hashes() -> Dict[str, str]:
    """Get content hashes for all source files that affect rendered messages."""
    nba_app_dir = _get_nba_app_dir()
    hashes = {}
    for rel_path in CORE_SOURCE_FILES:
        full_path = os.path.join(nba_app_dir, rel_path)
        if os.path.exists(full_path):
            hashes[rel_path] = _hash_file(full_path)

    # Add matchup_network globs
    for pattern in MATCHUP_NETWORK_SOURCE_GLOBS:
        full_pattern = os.path.join(nba_app_dir, pattern)
        for fp in sorted(glob.glob(full_pattern, recursive=True)):
            if os.path.isfile(fp):
                rel = os.path.relpath(fp, nba_app_dir)
                hashes[rel] = _hash_file(fp)
    return hashes


def _get_cache_path() -> str:
    """Get path to the cache file."""
    return os.path.join(_get_nba_app_dir(), 'agents', CACHE_FILE_NAME)


def _load_cache() -> Dict:
    """Load the cache file, returns empty dict if not found."""
    cache_path = _get_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: Dict) -> None:
    """Save the cache file."""
    cache_path = _get_cache_path()
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def is_regeneration_needed() -> Tuple[bool, List[str]]:
    """
    Check if system message regeneration is needed by comparing content hashes.

    Returns:
        Tuple of (needs_regeneration: bool, changed_files: list)
    """
    current_hashes = _get_source_hashes()
    cache = _load_cache()
    cached_hashes = cache.get('source_hashes', {})

    changed_files = []
    for filepath, current_hash in current_hashes.items():
        cached_hash = cached_hashes.get(filepath)
        if cached_hash != current_hash:
            changed_files.append(filepath)

    return len(changed_files) > 0, changed_files


def ensure_system_messages_current(silent: bool = True) -> bool:
    """
    Ensure system messages are up-to-date with core layer.

    Call this at agent workflow startup to auto-regenerate if needed.
    Uses content hashing (not timestamps) for reliability.

    Args:
        silent: If True, suppress output. If False, print status.

    Returns:
        True if messages were regenerated, False if already current.
    """
    needs_regen, changed_files = is_regeneration_needed()

    if not needs_regen:
        if not silent:
            print("System messages are current.")
        return False

    if not silent:
        print(f"Core files changed: {', '.join(changed_files)}")
        print("Regenerating system messages...")

    # Find and regenerate all agents with templates
    agents_dir = os.path.join(_get_nba_app_dir(), 'agents')
    regenerated = []

    for item in os.listdir(agents_dir):
        item_path = os.path.join(agents_dir, item)
        if os.path.isdir(item_path):
            template_path = os.path.join(item_path, 'system_message.template.txt')
            if os.path.exists(template_path):
                if generate_system_message(item, agents_dir):
                    regenerated.append(item)

    # Always regenerate matchup_network if sources changed
    if generate_matchup_network_system_messages(agents_dir):
        regenerated.append("matchup_network")

    # Update cache with current hashes
    cache = _load_cache()
    cache['source_hashes'] = _get_source_hashes()
    cache['last_generated'] = regenerated
    _save_cache(cache)

    if not silent:
        print(f"Regenerated: {', '.join(regenerated)}")

    return True


# =============================================================================
# TEMPLATE PROCESSING
# =============================================================================

def process_template(template_content: str) -> str:
    """
    Process a template by replacing all placeholders with generated content.

    Args:
        template_content: The template string with placeholders

    Returns:
        Processed string with placeholders replaced
    """
    result = template_content

    for placeholder, generator_func in PLACEHOLDERS.items():
        if placeholder in result:
            generated_content = generator_func()
            result = result.replace(placeholder, generated_content)

    return result


INCLUDE_RE = re.compile(r"\{\{INCLUDE:([^}]+)\}\}")


def process_includes(template_content: str, nba_app_dir: str) -> str:
    """
    Inline external markdown/text files into system message templates.

    Syntax:
      {{INCLUDE:relative/path/from/nba_app_root}}
    """
    result = template_content

    while True:
        m = INCLUDE_RE.search(result)
        if not m:
            break
        rel_path = (m.group(1) or "").strip()
        abs_path = os.path.join(nba_app_dir, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                included = f.read()
        except Exception as e:
            included = f"[INCLUDE_ERROR] Could not include {rel_path}: {e}"
        result = result[: m.start()] + included + result[m.end() :]

    return result


def generate_matchup_network_system_messages(agents_dir: str) -> bool:
    """
    Generate rendered system messages for matchup_network from markdown templates.

    Reads:
      agents/matchup_network/system_messages/*.md
    Writes:
      agents/matchup_network/system_messages/rendered/*.txt
    """
    nba_app_dir = os.path.dirname(agents_dir)
    base_dir = os.path.join(agents_dir, "matchup_network", "system_messages")
    rendered_dir = os.path.join(base_dir, "rendered")
    os.makedirs(rendered_dir, exist_ok=True)

    templates = [
        f for f in os.listdir(base_dir)
        if f.endswith(".md") and os.path.isfile(os.path.join(base_dir, f))
    ]
    if not templates:
        print(f"  No matchup_network templates found in {base_dir}")
        return False

    ok = True
    for filename in sorted(templates):
        template_path = os.path.join(base_dir, filename)
        out_name = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(rendered_dir, out_name)
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()
            # First: inline includes (docs)
            content = process_includes(template_content, nba_app_dir=nba_app_dir)
            # Then: core placeholders (optional)
            content = process_template(content)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Generated {output_path}")
        except Exception as e:
            ok = False
            print(f"  Error generating matchup_network message for {filename}: {e}")
    return ok


def generate_system_message(agent_name: str, agents_dir: str) -> bool:
    """
    Generate system message for a specific agent from its template.

    Args:
        agent_name: Name of the agent (e.g., 'modeler')
        agents_dir: Path to the agents directory

    Returns:
        True if successful, False otherwise
    """
    # Special-case: matchup_network uses markdown templates under system_messages/
    if agent_name == "matchup_network":
        return generate_matchup_network_system_messages(agents_dir)

    agent_dir = os.path.join(agents_dir, agent_name)
    template_path = os.path.join(agent_dir, "system_message.template.txt")
    output_path = os.path.join(agent_dir, "system_message.txt")

    if not os.path.exists(template_path):
        print(f"  No template found for {agent_name} at {template_path}")
        return False

    try:
        # Read template
        with open(template_path, 'r') as f:
            template_content = f.read()

        # Process template
        generated_content = process_template(template_content)

        # Write output
        with open(output_path, 'w') as f:
            f.write(generated_content)

        print(f"  Generated {output_path}")
        return True

    except Exception as e:
        print(f"  Error generating system message for {agent_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate agent system messages from templates"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Specific agent to generate (default: all agents with templates)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check if regeneration is needed (exit code 0 if current, 1 if stale)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-regenerate only if core files changed (idempotent, safe for startup)"
    )
    args = parser.parse_args()

    # Handle --check-only: just report status
    if args.check_only:
        needs_regen, changed_files = is_regeneration_needed()
        if needs_regen:
            print(f"Regeneration needed. Changed files: {', '.join(changed_files)}")
            sys.exit(1)
        else:
            print("System messages are current.")
            sys.exit(0)

    # Handle --auto: regenerate only if needed
    if args.auto:
        regenerated = ensure_system_messages_current(silent=False)
        sys.exit(0 if not regenerated else 0)  # Always success for --auto

    # Get agents directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = script_dir

    print("System Message Generator")
    print("=" * 50)
    print(f"Agents directory: {agents_dir}")
    print()

    # Find agents with templates
    if args.agent:
        agents = [args.agent]
    else:
        agents = []
        for item in os.listdir(agents_dir):
            item_path = os.path.join(agents_dir, item)
            if os.path.isdir(item_path):
                template_path = os.path.join(item_path, "system_message.template.txt")
                if os.path.exists(template_path):
                    agents.append(item)

    if not agents:
        print("No agents with templates found.")
        print("Create a system_message.template.txt file in an agent directory to use this script.")
        return

    print(f"Found {len(agents)} agent(s) with templates: {', '.join(agents)}")
    print()

    if args.dry_run:
        print("DRY RUN - showing available placeholders:")
        for placeholder in PLACEHOLDERS.keys():
            print(f"  {placeholder}")
        return

    # Generate system messages
    success_count = 0
    regenerated_agents = []
    for agent in agents:
        print(f"Processing {agent}...")
        if generate_system_message(agent, agents_dir):
            success_count += 1
            regenerated_agents.append(agent)

    # Update cache with current hashes after successful generation
    if success_count > 0:
        cache = _load_cache()
        cache['source_hashes'] = _get_source_hashes()
        cache['last_generated'] = regenerated_agents
        _save_cache(cache)

    print()
    print(f"Done. Generated {success_count}/{len(agents)} system messages.")
    print("Cache updated with current source file hashes.")


if __name__ == "__main__":
    main()
