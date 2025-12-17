"""
Wizard Step Widgets

Step panels for the PSTM migration wizard.

8-step workflow (updated order):
1. Input Data - Load traces & headers
2. Survey - Geometry analysis
3. Output Grid - Define output by corner points + bin size (BEFORE velocity)
4. Velocity - Configure velocity model (AFTER output grid)
5. Data Selection - Flexible trace filtering (NO validation)
6. Algorithm - Migration parameters
7. Execution - Run migration
8. Results - View results and QC
"""

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.steps.input_step import InputDataStep
from pstm.gui.steps.survey_step import SurveyStep
from pstm.gui.steps.output_grid_step import OutputGridStep
from pstm.gui.steps.velocity_step import VelocityStep
from pstm.gui.steps.data_selection_step import DataSelectionStep
from pstm.gui.steps.algorithm_step import AlgorithmStep
from pstm.gui.steps.execution_step import ExecutionStep
from pstm.gui.steps.results_step import ResultsStep

# Ordered list for wizard construction
STEP_CLASSES = [
    InputDataStep,       # Step 1
    SurveyStep,          # Step 2
    OutputGridStep,      # Step 3 (before velocity)
    VelocityStep,        # Step 4 (after output grid)
    DataSelectionStep,   # Step 5
    AlgorithmStep,       # Step 6
    ExecutionStep,       # Step 7
    ResultsStep,         # Step 8
]

STEP_KEYS = [
    "input",
    "survey",
    "output_grid",
    "velocity",
    "data_selection",
    "algorithm",
    "execution",
    "results",
]

__all__ = [
    "WizardStepWidget",
    "InputDataStep",
    "SurveyStep",
    "OutputGridStep",
    "VelocityStep",
    "DataSelectionStep",
    "AlgorithmStep",
    "ExecutionStep",
    "ResultsStep",
    "STEP_CLASSES",
    "STEP_KEYS",
]
