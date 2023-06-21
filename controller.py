"""
Controller module for the ARISE multi-task weak supervision models.

Loads ARISE models from ~/arise/MTL/arise.py via the hard link ~/arise.py.

To make debug requests using `curl`:
    curl -X POST "localhost:5555/" -H "Content-type: application/json" -d '{"name":"alice", "target": "bob"}'

"""

from flask import Flask, request, jsonify
import requests, json, os
import logging as log
from arise import *

app = Flask(__name__)
HOST, HOSTPORT = ("0.0.0.0", 5555)
UI_HOST, UI_HOSTPORT = ("0.0.0.0", 5000)

DATA_DIRECTORY = os.path.join(os.getcwd(), "arise", "MTL", "data") # where are the datafiles stored relative to this file?

TASKLIST = []   # which tasks should the model be trained for?
DATASET = "synth.csv" # "raw" or "synth"
LIFETIME = 3600 # seconds that a cached model should be valid (1 hour)
data_path = os.path.join(DATA_DIRECTORY, DATASET) # where is the CSV for this dataframe stored?

try:
    DF = pd.read_csv(data_path, index_col=0) # load from ~/arise/MTL/data/synth.csv
    DF_CT = 28 # number of entries in the dataset (not rows but distinct src/dest pairs)
except:
    log.warning("Failed to load default dataset. Hopefully this was intentional.")
    DF = ""
    DF_CT = 0

_MODEL_CONFIGURED = False # set to true once `setup()` has been run
model_predictions = [] # container for model predictions

@app.route('/', methods = ['GET', 'POST'])
def _index():
    global TASKLIST
    
    if request.method == "GET":
        print("Printing current task...")
        
        return TASKLIST
    elif request.method == "POST":
        print("Updating current task...")
        # print(request.values.get('name'))
        print (request.get_json())

        return "task updated"
    else:
        return "Invalid request."


@app.route('/set_tasks', methods = ['POST'])
def _set_tasks():
    global TASKLIST 
    
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = request.json
        
        if ('tasks' in data.keys()):
            if (len(data['tasks']) > 1):
                TASKLIST = data['tasks']
            else:
                TASKLIST = [json['tasks']] # support for single tasks
        return f"Updated task list: {TASKLIST}"
    else:
        return 'Content-Type not supported!'


@app.route('/set_dataset', methods = ['POST'])
def _set_dataset():
    """ Pass in the filename (.csv) of a datafile in ~/arise/MTL/data in the `dataset` field. """
    global DATASET, DF, DF_CT, data_path
    
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = request.json
        
        if ('dataset' in data.keys()):
            DATASET = data['dataset']
        else:
            return f"Error: no dataset specified"
        
        if ('df_ct' in data.keys()):
            DF_CT = data['df_ct'] # update number of pairs
        
        try:
            data_path = os.path.join(DATA_DIRECTORY, DATASET)
            DF = pd.read_csv(data_path, index_col=0)
        except:
            return f"Error reading dataset {DATASET}"
            
        return f"Updated dataset to {DATASET}"
    else:
        return 'Content-Type not supported!'


@app.route('/set_lifetime', methods = ['POST'])
def _set_lifetime():
    global LIFETIME 
    
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = request.json
        
        if ('lifetime' in data.keys()):
            LIFETIME = data['lifetime'] # support for single tasks
        return f"Updated cache life span to {LIFETIME} seconds."
    else:
        return 'Content-Type not supported!'


@app.route('/get_scores', methods = ['GET'])
def _get_scores():
    try:
        return SCORES
    except:
        return {}


@app.route('/get_preds', methods = ['GET'])
def _get_preds():
    # log.debug(f"Model predictions: \n{model_predictions}\n")
    
    # labels = {i : "[predictions]" for i in TASKLIST}
    # log.debug(f"Model Prediction: {type(model_predictions[0]['preds'][f'{TASKLIST[0]}_task'])}\n\n{model_predictions[0]['preds'][f'{TASKLIST[0]}_task']}")
    
    # fetch predictions and save query to output file
    # curl -X GET "http://localhost:5555/get_preds" -o preds.txt
    
    try:
        
        PREDS = { TASKLIST[i] : model_predictions[i]['preds'][f'{TASKLIST[i]}_task'].tolist() for i in range(len(TASKLIST)) }
        
        return jsonify(PREDS)
    except:
        log.warning("Failed to load preds.")
        return {}


@app.route('/get_dataset', methods = ['GET'])
def _get_data():
    return DATASET # filename in /data/


@app.route('/reset', methods = ['POST'])
def _reset_model():
    global TASKLIST, DF, DF_CT, _MODEL_CONFIGURED
    log.debug(f"Reconfiguring model for tasks {TASKLIST}.")
    
    setup(TASKLIST, DF, num_entries=DF_CT)
    
    _MODEL_CONFIGURED = True
    return f"Model reconfigured for tasks {TASKLIST}\n"


@app.route('/train', methods = ['POST'])
def _train_model():
    global model_predictions
    if not _MODEL_CONFIGURED:
        return "Error: model not yet configured."
    
    # Default model training parameters
    _lr = 0.01 # learning rate of 0.01 with Adam optimizer
    _epochs = 5 # use 5 training epochs
    _range = DF_CT # by default, train on all included datasets
    _cache_lifetime = 3600
    _use_cache = True
    
    # debug override
    _range = 3
    
    # If POST request is a valid JSON request with parameters `lr` and `epochs`, override the default training parameters.
    if (request.headers.get('Content-Type') == 'application/json'):
        data = request.json
        # log.warning(f"JSON: `{data}`, type: {type(data)}, vs request.json: {type(request.json)}, which is `{request.json}`")
        if ('lr' in data.keys()):     # model learning rate
            _lr = data['lr']
        if ('epochs' in data.keys()): # number of epochs to perform
            _epochs = data['epochs']
        if ('range' in data.keys()):  # which datasets to train on
            _range = data['range']
        if ('lifetime' in data.keys()): # cache lifetime
            _cache_lifetime = data['lifetime']
        if ('use_cache' in data.keys()): # if model should use cache or retrain
            _use_cache = (data['use_cache'] is True)
    else:
        log.debug("Content-Type is not application/json; training with default parameters")
    
    log.debug(f"Training for tasks {TASKLIST}.")
    
    model_predictions = train(TASKLIST, DF, datasets=range(_range), learning_rate=_lr, epochs=_epochs, cache_duration=_cache_lifetime, use_cache=_use_cache)
    
    return "model trained"


@app.route('/reply', methods = ['POST'])
def _reply():
    # Example CURL usage:
    # curl -X POST "localhost:5555/reply" -H "Content-type: application/json" -d '{"name":"alice", "data": "Bob"}'
    content = request.get_json(force=True)
    
    r = requests.post(f"http://{UI_HOST}:{UI_HOSTPORT}/log", \
        data=f'{{ "sender": "controller.py", "data": "{content["data"]}" }}')
    
    # image reply: (path is relative to project root `~/static/`)
    # path = "./figures/demo.png"
    try:
        path = content['path']
        p = requests.post(f"http://{UI_HOST}:{UI_HOSTPORT}/log", data=f'{{ "image": path}}')
    except:
        log.error("Could not find the 'path' field in request content!")
    
    # example of sending image from ~/figures/ directory (mirrored in ~/static/figures/)
    # requests.post(f"http://{HOST}:{HOSTPORT}/log", data=f'{{"image": "figures/demo.png"}}')
    # curl -X POST "localhost:5000/log" -d '{"image":"figures/demo.png"}'
    
    return ""


@app.route('/debug', methods = ['POST'])
def _debug():
    # Example CURL usage:
    # curl -X POST "localhost:5555/debug" -H "Content-type: application/json" -d '{"name":"alice", "data": "Bob"}'
    content = request.get_json(force=True)
    
    r = requests.post(f"http://{UI_HOST}:{UI_HOSTPORT}/debug", \
        data=f'{{ "sender": "controller.py", "data": "{content["data"]}" }}')
    return ""


if __name__ == "__main__":
    # Create and configure the default ARISE model(s)
    TASKLIST = ['loss', 'noise', 'congestion']
    # _reset_model()
    
    # Run the interactive controller on localhost:5555.
    app.run(HOST, port=HOSTPORT, debug=True)
    