#!/bin/bash

# Generate timestamp in readable format (YYYY-MM-DD_HH-MM-SS)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Define log filename
logfile="train_log_$timestamp.log"

# Run training and save output to the log file
python train.py | tee "log/$logfile"

