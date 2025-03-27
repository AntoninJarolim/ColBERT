#!/bin/bash

# experiment name
experiment='add_max_linear'

# Generate timestamp in readable format (YYYY-MM-DD_HH-MM-SS)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Define log filename
logfile="train_log_$timestamp.log"

./start_syncing.sh &

# Run training and save output to the log file
python train.py --add_max_linear --ngpus 4 --experiment $experiment --ex_lambda 0.6 --accumsteps 16 | tee "log/$logfile"

