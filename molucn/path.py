# Define the DIR to the data, model, logs, results, and colors
#
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data")
RESULT_PATH = os.path.join(ROOT_PATH, "results")
MODEL_PATH = os.path.join(ROOT_PATH, "model")
LOG_PATH = os.path.join(ROOT_PATH, "logs")
COLOR_PATH = os.path.join(ROOT_PATH, "colors")

