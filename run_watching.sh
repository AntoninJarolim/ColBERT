#!/bin/bash

# experiment name
experiment=$1


python -m utility.auto_inference.auto_inference $experiment
