#!/bin/bash

# Script to test all Intuit tools

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Testing Intuit Tools =====${NC}"
echo "Running tests at $(date)"
echo

# Run the Python test script
python scripts/test_all_tools.py

# Check the exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}All tools tested successfully!${NC}"
else
    echo -e "\n${RED}Some tools failed testing. Check the logs above for details.${NC}"
    exit 1
fi

echo
echo "Test completed at $(date)"
echo -e "${YELLOW}===== Test Complete =====${NC}"