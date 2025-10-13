#!/bin/bash

# Auto-push script for kg-healing-docred updates
# Usage: ./push_updates.sh "commit message"

set -e  # Exit on any error

# Check if commit message provided
if [ $# -eq 0 ]; then
    echo "Usage: ./push_updates.sh 'commit message'"
    exit 1
fi

COMMIT_MESSAGE="$1"

echo "Pushing updates to kg-healing-docred..."
echo "Commit message: $COMMIT_MESSAGE"
echo "=================================="

# Add all changes
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes to commit"
    exit 0
fi

# Commit with timestamp and message
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "[$TIMESTAMP] $COMMIT_MESSAGE"

# Push to origin
git push origin master

echo "=================================="
echo "Successfully pushed updates!"
echo "Repository: https://github.com/its-serah/kg-healing-docred"
