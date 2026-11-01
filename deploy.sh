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
echo -e "${YELLOW}[2/8] Staging and committing backend documentation on 'main'...${NC}"
git add docs/PROGRESS.md README.md docs/DECISION_LOG.md docs/PROJECT_REPORT_DOCUMENTATION.md deploy.sh
git add -u

if git diff-index --quiet HEAD --; then
    echo -e "${BLUE}[i] No changes to commit on 'main'.${NC}"
else
    git commit -m "${COMMIT_MSG}"
    echo -e "${GREEN}[+] Successfully committed backend changes to 'main'.${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Artifact Transition to Temporary Directory
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[3/8] Staging visualization artifacts to temporary directory...${NC}"
TEMP_DIR="/tmp/es335_viz"
rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

VIZ_SRC="data/processed/visualizations"
if [ -d "${VIZ_SRC}" ]; then
    cp -r "${VIZ_SRC}"/* "${TEMP_DIR}"/
    echo -e "${GREEN}[+] Copied artifacts to ${TEMP_DIR}.${NC}"
    echo -e "${YELLOW}[4/8] Removing local generated folder '${VIZ_SRC}' to prevent checkout conflicts...${NC}"
    rm -rf "${VIZ_SRC}"
else
    echo -e "${RED}[ERROR] Source visualization directory '${VIZ_SRC}' not found!${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 5. Branch Switch & Sync 'gh-pages'
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[5/8] Checking out 'gh-pages' branch...${NC}"
if ! git checkout gh-pages; then
    echo -e "${RED}[ERROR] Failed to switch to 'gh-pages' branch.${NC}"
    # Restore artifacts before exiting
    mkdir -p "${VIZ_SRC}"
    cp -r "${TEMP_DIR}"/* "${VIZ_SRC}"/
    rm -rf "${TEMP_DIR}"
    exit 1
fi

echo -e "${YELLOW}Pulling latest UI changes from origin...${NC}"
if ! git pull origin gh-pages; then
    echo -e "${RED}[ERROR] Failed to pull from origin/gh-pages. You may have a merge conflict.${NC}"
    echo -e "${RED}Temp artifacts are preserved in ${TEMP_DIR}. Resolve the conflict manually.${NC}"
    exit 1
fi


# ------------------------------------------------------------------------------
# 6. Asset Porting to Root 'visualizations/'
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[6/8] Porting HTML widgets into root 'visualizations/' on 'gh-pages'...${NC}"
mkdir -p visualizations
cp -r "${TEMP_DIR}"/* visualizations/
echo -e "${GREEN}[+] HTML widgets copied into visualizations/.${NC}"

# ------------------------------------------------------------------------------
# 7. Frontend Deployment on 'gh-pages'
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[7/8] Staging and committing frontend assets on 'gh-pages'...${NC}"
git add visualizations/
[ -f "index.html" ] && git add index.html
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
# 8. Cleanup and Return to 'main'
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[8/8] Returning to 'main' branch and restoring workspace...${NC}"
git checkout main || {
    echo -e "${RED}[ERROR] Failed to switch back to 'main' branch.${NC}"
    exit 1
}

# Restore data/processed/visualizations on main
mkdir -p "${VIZ_SRC}"
cp -r "${TEMP_DIR}"/* "${VIZ_SRC}"/
rm -rf "${TEMP_DIR}"
echo -e "${GREEN}[+] Restored ${VIZ_SRC} on 'main' and cleaned up temporary staging.${NC}"

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}      DEPLOYMENT & PROGRESS SYNC FINISHED SUCCESSFULLY!         ${NC}"
echo -e "${GREEN}================================================================${NC}\n"
