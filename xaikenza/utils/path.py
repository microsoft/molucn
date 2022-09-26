# Define the path to the data, model, logs, results, and colors
# 
import os

if 'AMLT_DATA_DIR' in os.environ:
    DATA_DIR = os.path.join(os.environ['AMLT_DATA_DIR'], 'data/')
else:
    DATA_DIR = './data/'
    
if 'AMLT_OUTPUT_DIR' in os.environ:
    OUTPUT_DIR = os.getenv('AMLT_OUTPUT_DIR', '/tmp')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'model/')
    LOG_DIR, RESULT_DIR, COLOR_DIR = OUTPUT_DIR, OUTPUT_DIR, OUTPUT_DIR
else:
    MODEL_DIR = './model/'
    LOG_DIR = './logs/'
    RESULT_DIR = './results/'
    COLOR_DIR = './colors/'
    