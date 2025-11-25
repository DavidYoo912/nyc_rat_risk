#!/bin/bash
# run_pipeline.sh
# Executes the MLOps pipeline end-to-end: Data -> Train -> Evaluate -> Predict.

# Ensure the script exits immediately if any command fails.
set -e

# --- Configuration ---
ENV_NAME="ml_env"
# Assuming Python scripts are located in a 'src' subdirectory
SRC_DIR="./src"

# --- Dynamically define the explicit path to the virtual environment's Python interpreter ---
# Check if python3 exists inside the VENV, otherwise fall back to 'python'
if [ -f "./$ENV_NAME/bin/python3" ]; then
    VENV_PYTHON="./$ENV_NAME/bin/python3"
    VENV_PIP="./$ENV_NAME/bin/pip3"
elif [ -f "./$ENV_NAME/bin/python" ]; then
    VENV_PYTHON="./$ENV_NAME/bin/python"
    VENV_PIP="./$ENV_NAME/bin/pip"
else
    # Fallback to the python3 command used globally (less reliable)
    VENV_PYTHON="python3"
    VENV_PIP="pip3"
fi
# ------------------------------------------------------------------------------------------

# Function to check the status of the last executed command and exit on failure
check_status() {
    # Note: set -e handles most errors, but this function is useful for specific messaging.
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Exiting pipeline."
        exit 1
    fi
}

echo "--- Starting Proactive Risk Modeling Pipeline ---"

# Pre-check: Ensure python3 command exists for environment creation
if ! command -v python3 &> /dev/null
then
    echo "Error: 'python3' command could not be found. Please ensure Python 3 is installed and accessible."
    exit 1
fi

# 1. Setup Virtual Environment and Install Dependencies
echo "1. Setting up virtual environment and installing dependencies..."
if [ ! -d "$ENV_NAME" ]; then
    python3 -m venv $ENV_NAME
    check_status "Virtual environment creation"
fi

# Print the determined path for verification
echo "Using Python executable: $VENV_PYTHON"

# Install dependencies using the previously defined requirements.txt
# We use the explicit VENV_PIP path to ensure dependencies are installed correctly
$VENV_PIP install -r requirements.txt
check_status "Dependency installation"

echo "Dependencies installed successfully."

# 2. Run Data Pipeline (data_pipeline.py) - USING EXPLICIT VENV PYTHON
echo ""
echo "--- 2. Executing Data Pipeline (data_pipeline.py) ---"
$VENV_PYTHON $SRC_DIR/data_pipeline.py
check_status "Data Pipeline"

# 3. Run Model Training (model_train.py) - USING EXPLICIT VENV PYTHON
echo ""
echo "--- 3. Executing Model Training (model_train.py) ---"
$VENV_PYTHON $SRC_DIR/model_train.py
check_status "Model Training"

# 4. Run Model Evaluation (model_evaluate.py) - USING EXPLICIT VENV PYTHON
echo ""
echo "--- 4. Executing Model Evaluation (model_evaluate.py) ---"
$VENV_PYTHON $SRC_DIR/model_evaluate.py
check_status "Model Evaluation"

# 5. Run Model Prediction (model_predict.py) - USING EXPLICIT VENV PYTHON
echo ""
echo "--- 5. Executing Model Prediction (model_predict.py) ---"
$VENV_PYTHON $SRC_DIR/model_predict.py
check_status "Model Prediction"

# 6. Cleanup (No deactivate needed as we didn't use 'source')

echo ""
echo "--- Pipeline Complete Successfully! ---"
echo "All output artifacts (models, reports, maps) are available in the 'artifacts/' directory."