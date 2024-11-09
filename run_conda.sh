#!/bin/bash

CONDA_ENV_NAME="tmp"

PYTHON_SCRIPT="gui.py"

source /home/kaal/miniforge3/etc/profile.d/conda.sh 
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
  echo "Failed to activate Conda environment: $CONDA_ENV_NAME"
  exit 1
fi

# Run the Python script
python "$PYTHON_SCRIPT"

# Deactivate the Conda environment after the script finishes
conda deactivate
