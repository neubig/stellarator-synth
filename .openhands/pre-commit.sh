#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running pre-commit hooks...${NC}"

# Run pre-commit on all files
pre-commit run --all-files

echo -e "${GREEN}Pre-commit hooks completed!${NC}"
