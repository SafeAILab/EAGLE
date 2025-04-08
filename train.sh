#!/bin/bash

# Script description
# Author: Kshitz Kaushik    
# Date: 2025-04-09

# Exit on error
set -e

# Define variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Functions
function cleanup() {
    # Add cleanup tasks here
    echo "Performing cleanup..."
}

# Trap errors
trap cleanup EXIT

# Main script logic
function main() {
    echo "Training LLama3.2-instruct-3B ..."
    echo "Fetching dataset..."
    mkdir -p training_data
    # wget --retry-connrefused --tries=3 --progress=bar -P training_data https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
    echo "Training ..."
    python "$SCRIPT_DIR/eagle/ge_data/ge_data_all_llama3instruct.py"

}

# Run main function
main "$@"