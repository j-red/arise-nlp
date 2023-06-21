from tsfresh import extract_features, extract_relevant_features
import sys, os, math, random, glob, torch, logging
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn import *
import warnings # see https://docs.python.org/3/library/warnings.html
from warnings import *
warnings.filterwarnings("default") # can also be set to 'ignore'
from utils import * # imports get_full_path() and others
from snorkel.classification import DictDataset, DictDataLoader, Operation, Task, MultitaskClassifier, Trainer
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from snorkel.analysis import Scorer
from timeit import default_timer as timer
from datetime import datetime as dt

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
log = logging.getLogger('arise')
log.setLevel(logging.DEBUG) # can be DEBUG, INFO, WARNING, ERROR, CRITICAL; logger will show all messages BELOW specified level

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = get_full_path("data_28") + "/"
FIGURE_DIR = get_full_path("MTL", "Figures") + "/"
RIPE_DIR = get_full_path("ripe") + "/"
SYNTH_DATA = get_full_path("MTL", "data") + "/synth.csv"

CUDA_INDEX = 0 # -1 to disable GPU (force CPU), use 0..n depending on number of GPUs

METRICS = ["f1", "accuracy"]
LOSS_FUNC = F.cross_entropy

result_list = []
SCORES = {}


# Device Configuration
if torch.cuda.is_available() and CUDA_INDEX >= 0:
    device = torch.device(f'cuda:{CUDA_INDEX}')
    torch.cuda.set_device(CUDA_INDEX)
else:
    log.warning("Defaulting to CPU calculations.")
    device = torch.device('cpu')

log.debug(f"Using {device}.")


#### TSFRESH Feature Extraction
from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators

@set_property("fctype", "simple")
def count_nonzero(x):
    """ Returns the number of nonzero (non-loss) measurements in the time series x. """
    return np.count_nonzero(x)

@set_property("fctype", "simple")
def noise_threshold(x):
    """ Returns the noise threshold for a time series
        by taking 1.5 * the RTT value at the 75th percentile. """
    x = x[x != 0] # remove all loss points from consideration
    return np.percentile(x, 75) * 1.5

@set_property("fctype", "simple")
def congestion_threshold(x):
    """ Returns the congestion threshold for a time series
        by taking 1.2 * the RTT value at the 30th percentile. """
    x = x[x != 0] # remove all loss points from consideration
    return np.percentile(x, 30) * 1.2


# Add custom features to list of feature calculators:
# https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html
feature_calculators.__dict__["count_nonzero"] = count_nonzero
feature_calculators.__dict__["noise_thresh"] = noise_threshold
feature_calculators.__dict__["congestion_thresh"] = congestion_threshold

custom = {
    "quantile": [{"q": 0.75}],
    #     "length": None, # number of entries in each time series
    "median": None,
    "mean": None,
    "noise_thresh": None,
    "congestion_thresh": None
}
disable_progress_bar = True

#### BUILDING LABELING FUNCTIONS

VOTE    =  1
NORMAL  =  0
ABSTAIN = -1

def label_noise(rtt, client_index):
    noise_threshold = features['rtt__noise_threshold'][client_index]
    if rtt >= noise_threshold:
        label = VOTE
    else:
        label = NORMAL
    return label

def label_outage(rtt, client_index):
    if rtt == 0:
        label = VOTE
    else:
        label = NORMAL
    return label

def label_congestion(rtt, client_index):
    cong_threshold = features['rtt__congestion_threshold'][client_index]
    noise_threshold = features['rtt__noise_threshold'][client_index]

    if rtt >= cong_threshold and rtt < noise_threshold:
        label = VOTE
    else:
        label = NORMAL
    return label


def generate_data(df):
    """ DF should be a labeled, original dataset. """
    global features

    newdata = pd.DataFrame(columns=["id", "i", "rtt", "loss", "congestion", "noise"])

    for i, row in df.iterrows():
        # print(f"Index {i}; data: \n{row['rtt']}\n")
        ID = row['id']

        try:
            index = row['i']
        except:
            index = row['index']

        # Appending original data
        rtt = row['rtt']
        l, c, n = (row['loss'], row['congestion'], row['noise']) # labels
        newdata = newdata.append({'id': ID, 'i': index, 'rtt': rtt, 'loss': l, 'congestion': c, 'noise': n}, ignore_index=True)

        # Appending synthesized data -- loss
        rtt = 0
        l, c, n = (VOTE, NORMAL, NORMAL) # labels
        newdata = newdata.append({'id': ID, 'i': index, 'rtt': rtt, 'loss': l, 'congestion': c, 'noise': n}, ignore_index=True)

        # Appending synthesized data -- congestion
        cong_threshold = features['rtt__congestion_threshold'][ID]
        noise_thresh = features['rtt__noise_threshold'][ID]
        rtt = np.random.randint(cong_threshold, noise_thresh)
        l, c, n = (NORMAL, VOTE, NORMAL) # labels
        newdata = newdata.append({'id': ID, 'i': index, 'rtt': rtt, 'loss': l, 'congestion': c, 'noise': n}, ignore_index=True)

        # Appending synthesized data -- noise
        noise_scale = 5 # scale upper bound for noise by this. original is 4.
        gap = 500 # ensure a gap of at least XXms between synthetic congestion and noise RTT values
        noise_thresh = features['rtt__noise_threshold'][ID]
        rtt = np.random.randint(noise_thresh + gap, noise_thresh * noise_scale + gap)
        l, c, n = (NORMAL, NORMAL, VOTE) # labels
        newdata = newdata.append({'id': ID, 'i': index, 'rtt': rtt, 'loss': l, 'congestion': c, 'noise': n}, ignore_index=True)

    return newdata.reset_index(drop=True)


def setup(tasks : list, df, num_entries):
    log.debug(f"Loaded dataset containing {len(df)} entries.")

    #### Perform feature extraction with custom settings
    try:
        df.reset_index(inplace=True)
    except:
        log.debug("Failed to reset index in place. Continuing.")


    features = extract_features(df, default_fc_parameters=custom, column_id="id", \
                                column_sort="index", column_value="rtt", \
                                disable_progressbar=disable_progress_bar).round(5)

    """ SPLIT INTO TRAINING/TESTING/VALIDATION SETS """

    # X values are the measurements (rtt), Y values are the associated labels
    X_train, X_validate, X_test = {}, {}, {}
    Y_train, Y_validate, Y_test = {}, {}, {}

    for i in range(num_entries):
        master = get(df, index=i)

        for task in tasks:
            splt = split_df(master)
            splt_2 = split_df(master, ct=4)
            splt_2 = splt_2[1].append(splt_2[2], ignore_index=True)

            X_train[task] = splt[0]['rtt'].apply(lambda x : np.array([x]))
            X_validate[task] = splt[1]['rtt'].apply(lambda x : np.array([x]))
            X_test[task] = splt_2['rtt'].apply(lambda x : np.array([x]))

            Y_train[task] = splt[0][task]
            Y_validate[task] = splt[1][task]
            Y_test[task] = splt_2[task]

        rtts = master['rtt'].apply(lambda x : np.array([x])) # convert rtt's into np array of sample arrays


def train_and_evaluate(df, i : int, tasks : list, show_conf_matrix=False, n_epochs=10, lr=0.01, directory="", savefig=False, \
                       cache_model=True, titleprefix="", show_progress=False, model_name="MTL_Latest", \
                       use_cached=True, time_override=False, train_time=0, cache_lifetime=3600):
    """ Iteratively train and evaluate the ith synthesized dataset. Results are stored in the SCORES dictionary, rather
        than being returned. 
        
        Parameters:
          - train_time refers to the datetime timestamp that training was initiated
          - cache_lifetime refers to the amount of time that a cached model will be considered valid (in seconds). Default is one hour.
          - use_cached refers to if the model will attempt to use a cached model if one is present. If false, it will retrain the model each time.
          - cache_model refers to whether or not the model should cache its attempts to train. Set this to false if experimenting with volatile hyperparameters.
    """
    global SCORES, METRICS, LOSS_FUNC, SEED, result_list
    
    _tasks = ' '.join(sorted(tasks)) # sort list of tasks and use them to 
    CACHE_DIR = f"./.cache/{_tasks}/"  # cache the proper models in proper directories
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    
    X_train, X_validate, X_test = {}, {}, {}
    Y_train, Y_validate, Y_test = {}, {}, {}

    master = get(df, index=i)
    for task in tasks:
        splt = split_df(master)
        splt_2 = split_df(master, ct=4)
        splt_2 = splt_2[1].append(splt_2[2], ignore_index=True)

        X_train[task] = splt[0]['rtt'].apply(lambda x : np.array([x]))
        X_validate[task] = splt[1]['rtt'].apply(lambda x : np.array([x]))
        X_test[task] = splt_2['rtt'].apply(lambda x : np.array([x]))

        Y_train[task] = splt[0][task]
        Y_validate[task] = splt[1][task]
        Y_test[task] = splt_2[task]

    rtts = master['rtt'].apply(lambda x : np.array([x])) # convert rtt's into np array of sample arrays

    """ Now, we define the dataloaders for the model. These will access the input data dictionaries for each
        individual dataset and load the required portions needed for validation, training, and testing evaluations. """

    loaders = {t : [] for t in ["train", "valid", "test"] + tasks} # used for confusion matrix generation
    dataloaders = []

    for task_name in tasks:
        for split, X, Y in (
                ("train", X_train, Y_train),
                ("valid", X_validate, Y_validate),
                ("test", X_test, Y_test),
            ):

            try:
                X_dict = {f"{task_name}_data": torch.FloatTensor(X[task_name])}
            except:
                X_dict = {f"{task_name}_data": torch.FloatTensor(list(X[task_name]))}

            try:
                Y_dict = {f"{task_name}_task": torch.LongTensor(Y[task_name])}
            except:
                Y_dict = {f"{task_name}_task": torch.LongTensor(list(Y[task_name]))}

            dataset = DictDataset(f"{task_name}_Dataset", split, X_dict, Y_dict)
            dataloader = DictDataLoader(dataset, batch_size=32) # batch size is the number of data points loaded in at time
            dataloaders.append(dataloader)

            loaders[split].append(dataloader)     # add current loader to list sorted by data type (test, train, or val)
            loaders[task_name].append(dataloader) # add current loader based on task

    """ Here, we define the initial layers of the perceptron model. 'base_mlp' indicates the layer shared between the
        different tasks, while the 'head_module' denotes the model's prediction layer. """

    # Define a two-layer MLP module and a one-layer prediction "head" module
    # base_mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

    numLayers = 4 # default 4
    hiddenSize = 4 # default 4
    in_features = 1

    base_mlp = nn.Sequential(
                    nn.Linear(in_features, hiddenSize),
                    nn.ReLU(),
                    # nn.LSTM(in_features, hidden_size=hiddenSize, num_layers=1, bidirectional=True), # put these first?
                    # nn.ReLU(),
                    nn.Linear(hiddenSize, 4),
                    nn.ReLU()
                )


    """ Here, we define the tasks themselves. These are the layers of the MTL model that share information through
        the 'base_mlp' perceptron module. The tasks function equivalently, and are merely described differently as a result
        of adherence to the Snorkel Multi-Task learning documentation formatting. """

    def initialize_task(taskname : str, base):
        """ A more modular method of defining the same tasks as seen above. """
        global LOSS_FUNC, METRICS

        in_features = 4  # number of elements in hidden layer
        out_features = 2

        task = Task(
            name = f"{taskname}_task",
            module_pool = nn.ModuleDict({"base_mlp": base, f"{taskname}_head": nn.Linear(in_features, out_features)}),
            loss_func = LOSS_FUNC,
            output_func = partial(F.softmax, dim=1),
            scorer = Scorer(metrics=METRICS),
            op_sequence = [
                    Operation("base_mlp", [("_input_", f"{taskname}_data")]), # base multi-layered perception
                    Operation(f"{taskname}_head", ["base_mlp"]),
                ]
            )

        return task


    TASKS = [initialize_task(t, base_mlp) for t in tasks]
    # TASKS = [loss_task, noise_task, congestion_task] # original tasks

    """ Finally, we train the Multi Task classifier and score it according to our evaluation metrics list described above.
        By default, we use the model's F1 score as the basis for evaluation. The duration of the training process for each
        individual dataset within the model is also recorded. """

    model = MultitaskClassifier(TASKS, name="ARISE_Classifier", device=CUDA_INDEX)

    trainer_config = { # see https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/training/trainer.py
        "seed": SEED,
        "n_epochs": n_epochs,
        "lr": lr,
        "lr_scheduler": "linear", # one of ["constant", "linear", "exponential", "step"]
        "progress_bar": show_progress,
        "checkpointing": True
    }
    trainer = Trainer(**trainer_config)


    start = timer() # start timer for loading or training
    
    # if use_cache == False, the model will be forced to retrain
    # log.warning(f"USE_CACHED: {use_cached}\n\n")
    if use_cached:
        try:
            cached_models = sorted(os.listdir(CACHE_DIR)) # list dirs in /cache/<tasks>/
            most_recent = cached_models[-1] # take most recent
            log.debug(f"Attempting to load models from timestamp {most_recent}")
            
            # check to ensure most recent is within threshold for model age
            _now = int(dt.now().timestamp())
            _age = _now - int(most_recent)
            if (_age >= cache_lifetime):
                log.warning(f"Most recent cache was updated more than {cache_lifetime} seconds ago, so the model will be retrained. Increase the 'cache_lifetime' parameter in the train() function call to prevent retraining.")
                use_cached = False
                train.fit(model, dataloaders)
            else:
                # age within valid range
                log.debug(f"Loading model at {CACHE_DIR}{most_recent}/{i}")
                model.load(f"{CACHE_DIR}{most_recent}/{i}")
                log.debug(f"Model loading complete. Age: {_age} seconds")
        except Exception as e:
            use_cached = False # if we fail to load config, simply regen model config
            log.error(f"Error loading cached model in {CACHE_DIR}:\n\t{e}")
            
            trainer.fit(model, dataloaders)
    else:
        trainer.fit(model, dataloaders)
    
    end = timer() # end model training timer
    

    model.eval() # change the model to evaluation mode (disable dropout layers, etc)


    results = model.score(dataloaders, as_dataframe=False)
    # print('model.score:', results)

    SCORES[i] = {"Time": round(end - start, 5), "Scores": results, "Index": i, "Iterations": 0}

    """  Cache MTL Model to specified filepath: """
    log.warning(f"cache_model: {cache_model}, use_cached: {use_cached}")
    if cache_model and not use_cached: # don't overwrite existing models
        if (not os.path.exists(f"{CACHE_DIR}{train_time}")):
            os.makedirs(f"{CACHE_DIR}{train_time}")
        
        model.save(f"{CACHE_DIR}{train_time}/{i}") # '<cache dir>/<timestamp>/<index>'
        log.debug(f"Wrote model to {CACHE_DIR}{train_time}/{i}")
    else:
        log.warning(f"This model will not be cached.")

    """ Evaluating results """
    typ = "test" # which set of the data should be used to evaluate?
    result_list = []

    for dl in loaders[typ]:
        res = model.predict(dl, return_preds=True)
        result_list.append(res)

        # print('model.predict:', res)
        # print(res['preds'].keys())
        key = list(res['preds'].keys())[0]
        preds = res["preds"][f"{key}"] # predictions

    return result_list


def train(tasks, df, datasets=[0], learning_rate=0.01, epochs=10, use_cache=True, save_cache=True, cache_duration=3600):
    global SCORES, SUPPORTED_TASKS
        
    avgs = []
    # results = {}
    
    current_time = int(dt.now().timestamp()) # get cache time for models

    for i in datasets:
        print(f"{'='*5} Training on dataset {i:02d} {'='*5}")
        
        modelname = f"{current_time} {i:02d}"

        # CAIDA -- train on augmented, test on original/raw
        results = train_and_evaluate(df, i, tasks, savefig=False, show_conf_matrix=False, \
            n_epochs=epochs, lr=learning_rate, titleprefix="", train_time=current_time, \
            cache_model=save_cache, show_progress=True, use_cached=use_cache, \
            model_name=modelname, cache_lifetime=cache_duration)
        
        # Raw CAIDA data
        # results = train_and_evaluate(raw, i, tasks, savefig=False, show_conf_matrix=False, \
        #                             n_epochs=EPOCHS, lr=rate, titleprefix="raw_", cache_model=save_cache, \
        #                             show_progress=True, use_cached=use_cache, model_name=f"{i:02d}_MTL_raw")

    avg = 0
    time = 0

    pprint(SCORES)

    for i in SCORES:
        for j in SCORES[i]['Scores']:
            avg += SCORES[i]['Scores'][j] / len(SCORES[i]['Scores'])

        time += SCORES[i]["Time"]

    avg /= len(SCORES)
    avgs.append(avg)
    print(f"\nMean F1 score: `{avg}`  ")
    print(f"Average Time: `{round(time / len(SCORES), 5)}`")

    print("Done.")

    return results


# Old version: for running models directly. New version uses backend in ~/controller.py.
if __name__ == '__main__':
    tasks = ["loss", "noise", "congestion"]
    # tasks = ["loss", "noise"]
    # tasks = ["loss", "noise", "congestion", "changepoint"] # keyerror with changepoint
    # tasks = ["loss", "noise", "congestion", "changepoint", "query1", "query2"]

    dataset = "synth" # choose from "raw", "synth"

    if dataset == "raw":
        df = get_all_datasets() # from utils.py
    elif dataset == "synth":
        datafile = os.path.join(os.getcwd(), "arise", "MTL", "data", "synth.csv")

        try: # Read synth data from CSV by default, otherwise, synthesize it again.
            print(f"\n\nDatafile: {datafile}\n\n")
            df = pd.read_csv(datafile, index_col=0)           
        except:
            log.warning(f"{datafile} not found; regenerating.")

            df = master
            new_df = pd.DataFrame(columns=['id', 'i', 'datetime', 'rtt']) # empty dataframe to append modified DataFrames to

            for client_id in range(28):
                # Synthesize data for better performance with weak supervision.
                print(f"Synthesizing data for client {client_id:02d}...")

                current_df = df[df['id'] == client_id] # df containing only values for current client
                noise_threshold = features['rtt__noise_threshold'][client_id]
                cong_threshold = features['rtt__congestion_threshold'][client_id]

                # Apply labeling functions to original datasets.
                current_df['loss'] = current_df['rtt'].apply(lambda x: label_outage(x, client_id))
                current_df['congestion'] = current_df['rtt'].apply(lambda x: label_congestion(x, client_id))
                current_df['noise'] = current_df['rtt'].apply(lambda x: label_noise(x, client_id))
                current_df.rename(columns = {'index':'i'}, inplace=True)
                del current_df['datetime']

                # Synthesize new data in line with the current df.
                current_df = generate_data(current_df)

                new_df = new_df.append(current_df) # append modified dataframes to new modified set

            new_df = new_df.astype({"id": 'int', "i": 'int', "loss": 'int', \
                                    "congestion": 'int', "noise": 'int'}) # trim labels to ints, rather than floats

            synth = new_df.reset_index(drop=True)
            del synth["datetime"]
            print("Done.")

            synth.to_csv(datafile, encoding='utf-8', header=True)
    else:
        raise Exception("Invalid dataset!")

    num = 3 # num datasets to train, up to 28
    setup(tasks, df, num_entries=num)
    train(tasks, df, datasets=range(num), learning_rate=0.01, epochs=5, use_cache=True, \
        cache_duration=3600)
