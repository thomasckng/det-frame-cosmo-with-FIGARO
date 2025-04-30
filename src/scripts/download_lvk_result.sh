#!/bin/bash

# File containing the list of events
EVENT_LIST="./lvk_result_list.txt"

# Base URLs for the files
BASE_URL_1="https://zenodo.org/records/6513631/files/"
BASE_URL_2="https://zenodo.org/records/8177023/files/"

# Directory to save the downloaded files
SAVE_DIR="../data/real/lvk_result/"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Read the event list file line by line
while IFS= read -r line; do
    # Determine the base URL based on the file name pattern
    if [[ $line == IGWN-GWTC2p1-v2-* ]]; then
        BASE_URL=$BASE_URL_1
    elif [[ $line == IGWN-GWTC3p0-v2-* ]]; then
        BASE_URL=$BASE_URL_2
    fi

    # Construct the full URL
    FILE_URL="${BASE_URL}${line}?download=1"
    FILE_PATH="${SAVE_DIR}${line}"

    # Check if the file already exists
    if [ -f "$FILE_PATH" ]; then
        echo "File $FILE_PATH already exists. Skipping download."
    else
        # Download the file using wget and save it to the specified directory
        wget -P "$SAVE_DIR" "$FILE_URL"
    fi

done < "$EVENT_LIST"

# Download the O3 search sensitivity estimates file and save it to ../data/real/
ADDITIONAL_FILE="../data/real/endo3_bbhpop-LIGO-T2100113-v12.hdf5"
if [ -f "$ADDITIONAL_FILE" ]; then
    echo "File $ADDITIONAL_FILE already exists. Skipping download."
else
    wget -O "$ADDITIONAL_FILE" "https://zenodo.org/records/7890437/files/endo3_bbhpop-LIGO-T2100113-v12.hdf5?download=1"
fi
