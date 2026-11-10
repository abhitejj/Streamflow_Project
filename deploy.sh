#!/usr/bin/env bash
# ==============================================================================
# ES 335 Hydro-ML Study — Automated Deployment & Documentation Sync Script
#
# Automates deploying static Plotly visualizations to `gh-pages` and
# synchronously updates project tracking documentation on `main`.
# ==============================================================================

set -e

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# 1. Input Argument Gate
# ------------------------------------------------------------------------------
if [ -z "$1" ]; then
    echo -e "${RED}================================================================${NC}"
    echo -e "${RED}[ERROR] Deployment aborted: Commit message argument is required.${NC}"
    echo -e "${RED}================================================================${NC}"
    echo -e "${YELLOW}Usage:   ./deploy.sh \"<progress update and commit message>\"${NC}"
    echo -e "${YELLOW}Example: ./deploy.sh \"Updated KGE component decomposition widgets\"${NC}"
    exit 1
fi

COMMIT_MSG="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo -e "\n${CYAN}================================================================${NC}"
echo -e "${CYAN}      ES 335 AUTOMATED DEPLOYMENT & ARTIFACT TRANSITION         ${NC}"
echo -e "${CYAN}================================================================${NC}"
echo -e "${BLUE}[i] Message: \"${COMMIT_MSG}\"${NC}\n"

# Ensure we are starting from 'main'
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "${CURRENT_BRANCH}" != "main" ]; then
    echo -e "${YELLOW}[!] Currently on branch '${CURRENT_BRANCH}'. Switching to 'main'...${NC}"
    git checkout main || {
        echo -e "${RED}[ERROR] Failed to switch to 'main' branch.${NC}"
        exit 1
    }
fi

# ------------------------------------------------------------------------------
# 2. Documentation Update (Main Branch)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[1/8] Updating docs/PROGRESS.md on 'main'...${NC}"
mkdir -p docs
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S %Z")

if [ ! -f "docs/PROGRESS.md" ]; then
    cat << 'HEADER_EOF' > docs/PROGRESS.md
# Project Progress & Deployment Log

This log tracks project milestones, pipeline updates, and deployments to `main` and `gh-pages`.

HEADER_EOF
fi

echo "- **${TIMESTAMP}**: ${COMMIT_MSG}" >> docs/PROGRESS.md
echo -e "${GREEN}[+] Appended progress entry to docs/PROGRESS.md${NC}"

# ------------------------------------------------------------------------------
# 3. Backend Commit on 'main'
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[2/8] Rebuilding audited outputs, then committing on 'main'...${NC}"
python3 scripts/audit_metrics.py > /dev/null
python3 scripts/export_benchmark_data.py > /dev/null
python3 scripts/export_explorer_data.py > /dev/null
python3 scripts/generate_interactive_plots.py > /dev/null
python3 scripts/uncertainty_tests.py > /dev/null
python3 scripts/render_report_results.py > /dev/null
git add data/processed/model_results/ data/processed/validation_reports/ assets/benchmark-data.js assets/explorer-data.js visualizations/source/ scripts/ models/
git add docs/PROGRESS.md README.md docs/DECISION_LOG.md docs/PROJECT_REPORT_DOCUMENTATION.md docs/AUDIT_LOG.md docs/REVIEW_FIX_PROGRESS.md deploy.sh
git add -u

if git diff-index --quiet HEAD --; then
    echo -e "${BLUE}[i] No changes to commit on 'main'.${NC}"
else
    git commit -m "${COMMIT_MSG}"
    echo -e "${GREEN}[+] Successfully committed backend changes to 'main'.${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Rebuild audited outputs and stage them to a temporary directory
#    Pipeline: audit_metrics.py -> export_benchmark_data.py -> generate_interactive_plots.py
#    (charts are written to visualizations/source/, the only chart source the site uses)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[3/8] Staging rebuilt chart sources and site data...${NC}"
TEMP_DIR="$(mktemp -d)"
python3 scripts/export_replay_data.py --assets-dir "${TEMP_DIR}"
VIZ_SRC="visualizations/source"
cp "${VIZ_SRC}"/interactive_*.html "${TEMP_DIR}"/
cp assets/benchmark-data.js assets/explorer-data.js tools/build_site_data.py "${TEMP_DIR}"/
echo -e "${GREEN}[+] Staged chart sources and site data in ${TEMP_DIR}.${NC}"
echo -e "${YELLOW}[4/8] (no local folders removed)${NC}"

# ------------------------------------------------------------------------------
# 5. Locate (or create) the gh-pages worktree; 'main' stays checked out here
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[5/8] Locating the 'gh-pages' worktree...${NC}"
SITE_DIR="$(git worktree list --porcelain | awk '/^worktree /{wt=substr($0,10)} /^branch refs\/heads\/gh-pages$/{print wt}')"
if [ -z "${SITE_DIR}" ]; then
    SITE_DIR="${REPO_ROOT}/site"
    git worktree add "${SITE_DIR}" gh-pages || { echo -e "${RED}[ERROR] Could not create gh-pages worktree.${NC}"; rm -rf "${TEMP_DIR}"; exit 1; }
fi
echo -e "${GREEN}[+] Using gh-pages worktree at ${SITE_DIR}.${NC}"
cp docs/PROJECT_REPORT_DOCUMENTATION.md docs/DECISION_LOG.md docs/PROGRESS.md "${TEMP_DIR}"/
cd "${SITE_DIR}"

echo -e "${YELLOW}Pulling latest UI changes from origin...${NC}"
if ! git pull --ff-only origin gh-pages; then
    echo -e "${RED}[ERROR] Failed to fast-forward gh-pages from origin. Resolve manually in ${SITE_DIR}.${NC}"
    echo -e "${RED}Temp artifacts are preserved in ${TEMP_DIR}.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 6. Port chart sources, data and docs; rebuild figdata.js, chart pages and table
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[6/8] Rebuilding site data on 'gh-pages'...${NC}"
mkdir -p visualizations/source tools assets docs
cp "${TEMP_DIR}"/interactive_*.html visualizations/source/
cp "${TEMP_DIR}"/benchmark-data.js "${TEMP_DIR}"/explorer-data.js assets/
mkdir -p assets/replay
cp "${TEMP_DIR}"/replay-manifest.js assets/
cp "${TEMP_DIR}"/replay/*.json assets/replay/
cp "${TEMP_DIR}"/build_site_data.py tools/
cp "${TEMP_DIR}"/PROJECT_REPORT_DOCUMENTATION.md "${TEMP_DIR}"/DECISION_LOG.md "${TEMP_DIR}"/PROGRESS.md docs/
python3 tools/build_site_data.py

# ------------------------------------------------------------------------------
# 7. Frontend Deployment on 'gh-pages'
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[7/8] Staging and committing frontend assets on 'gh-pages'...${NC}"
git add visualizations/ assets/ tools/build_site_data.py index.html docs/
[ -f ".nojekyll" ] && git add .nojekyll

if git diff-index --quiet HEAD --; then
    echo -e "${BLUE}[i] No new changes detected on 'gh-pages'.${NC}"
else
    git commit -m "${COMMIT_MSG}"
    echo -e "${GREEN}[+] Committed frontend assets to 'gh-pages'.${NC}"
fi

echo -e "${YELLOW}[*] Pushing 'gh-pages' to remote origin...${NC}"
git push origin gh-pages
echo -e "${GREEN}[+] Successfully pushed to origin/gh-pages.${NC}"

# ------------------------------------------------------------------------------
# 8. Cleanup
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[8/8] Cleaning up...${NC}"
cd "${REPO_ROOT}"
rm -rf "${TEMP_DIR}"
echo -e "${GREEN}[+] Cleaned up temporary staging. 'main' was never switched out.${NC}"

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}      DEPLOYMENT & PROGRESS SYNC FINISHED SUCCESSFULLY!         ${NC}"
echo -e "${GREEN}================================================================${NC}\n"
