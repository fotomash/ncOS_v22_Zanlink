#!/bin/bash
# This script organizes the ncOS_v22_Zanlink project directory
# according to the BUNDLE_STRUCTURE.md specification.

echo "🚀 Starting project organization..."

# --- Step 1: Create the standard top-level directories ---
echo "▶ Creating standard directories..."
mkdir -p agents
mkdir -p api
mkdir -p config
mkdir -p core
mkdir -p data
mkdir -p docs
mkdir -p integrations
mkdir -p logs
mkdir -p strategies
mkdir -p tests

# --- Step 2: Move integration and utility files from root to ./integrations/ ---
echo "▶ Moving integration files to ./integrations/ ..."
mv ncos_advanced_pattern_recognition.py \
   ncos_chatgpt_actions.py \
   ncos_data_package_manager.py \
   ncos_integration_bridge.py \
   ncos_llm_gateway.py \
   ncos_prompt_templates.py \
   ncos_realtime_pattern_stream.py \
   ncos_zanlink_bridge.py \
   offline_enrichment.py \
   secure_config.py \
   finnhub_data_fetcher_secure.py \
   fix_test_imports.py \
   tick_analysis_engine.py \
   tick_bar_integration.py \
   tick_analysis_usage_example.py \
   integrations/

# --- Step 3: Move config files from root to ./config/ ---
echo "▶ Moving config files to ./config/ ..."
mv ncos_chatgpt_schema_zanlink.yaml ncos_config_zanlink.json config/

# --- Step 4: Consolidate core logic from ./src/ and ./core/ subdirectories ---
echo "▶ Consolidating core application logic..."
# Move all agent definitions into the top-level 'agents' directory
mv src/ncos/agents/* agents/

# Move all engine and core logic files into the top-level 'core' directory
mv src/ncos/engines/* core/
mv src/ncos/core/* core/
# Also consolidate from the existing 'core' subdirectories
mv core/engines/* core/
mv core/orchestrators/* core/


# Move all strategy files into the top-level 'strategies' directory
mv src/ncos/strategies/* strategies/

# --- Step 5: Clean up now-empty or redundant directories ---
# Please review this section carefully before running.
# These commands will permanently delete the specified folders.

echo "▶ Cleaning up redundant directories. Review these commands!"
echo "   - Removing ./_deploy"
rm -rf _deploy

echo "   - Removing ./ncos_main_v22 (duplicate project)"
rm -rf ncos_main_v22

echo "   - Removing ./src (should be empty now)"
rm -rf src

echo "   - Removing ./zanllink (old module)"
rm -rf zanllink

echo "   - Removing ./core/agents, ./core/engines, ./core/orchestrators (should be empty now)"
rm -rf core/agents core/engines core/orchestrators

echo "   - Removing ./untitled\ folder"
rm -rf "untitled folder"


echo "✅ Organization complete!"
echo "Please review the new structure with the 'ls -R' command."