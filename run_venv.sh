#!/bin/bash

# Specify the path to the Python virtual environment
VENV_PATH="myenv"

# Specify the path to your Python script
PYTHON_SCRIPT="gui.py"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Check if the virtual environment was activated successfully
if [ $? -ne 0 ]; then
  echo "Failed to activate virtual environment: $VENV_PATH"
  exit 1
fi

# Run the Python script
python "$PYTHON_SCRIPT"

# Deactivate the virtual environment after the script finishes
deactivate
