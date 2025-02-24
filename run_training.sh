#!/bin/bash


# experiment name
experiment=$1

# Generate timestamp in readable format (YYYY-MM-DD_HH-MM-SS)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Define log filename
logfile="train_log_$timestamp.log"

# Run training and save output to the log file
python train.py --experiment $experiment  | tee "log/$logfile"

