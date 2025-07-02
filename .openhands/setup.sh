#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up OpenHands development environment...${NC}"

# Install pre-commit
echo -e "${YELLOW}Installing pre-commit...${NC}"
pip install pre-commit

# Install pre-commit hooks
echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
pre-commit install

echo -e "${GREEN}OpenHands setup complete!${NC}"
