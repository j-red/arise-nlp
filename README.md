# ACIA

ARISE Conversational Intelligence Agent

## Getting started

This project uses two conda environments running Python 3.7.6 and Python 3.8.5 with several data science and natural language processing packages such as PyTorch, Tensorflow, Snorkel, and Rasa. Follow the instructions below to install and configure the necessary environment.

The project consists of four primary components listed below. 
* The Web Interface functions as the frontend/UI for the framework, and operates using a Flask webserver to serve the frontend page on `localhost:5000`. This server sends operator queries to the NLU engine for semantic parsing, and receives information from the Action Server to display in the chatlog. To run this server, use `python3 server.py` in the root of this repository.
* The Natural Language Understanding (NLU) engine parses operator queries into entities and intents, and feeds this information into the appropriate action via the Action Server. The NLU engine operates using the configuration `.yml` files in the subfolders of `~/nlp/`, and can be run by executing the `./run` file in `~/nlp/nlu-engine/` while using the NLP conda environment (Python 3.8.5).
* The Action Server receives information from the NLU engine and acts as the executor for most of the logic in the framework. It interfaces with the other components in the system via the `Action` classes in `~/webhook/actions/actions.py`, which can be modified to control or send information to other components in the framework. 
* The ARISE multi-task learning models are controlled via the `controller.py` file in the root of the project repository. This also operates using a headless Flask webserver and receives configuration information or control commands via port 5555 (see `controller.py` for more information and example commands using `curl`). This file loads `arise.py`, which contains the core logic for the multi-task models themselves, including the caching, loading, training, and execution of tasks.

Any component listed below as using Python 3.7.6 should use the `arise` environment, and any component using Python 3.8.5 should use the `nlp` environment. Instructions for setting up the environments are below.

## Component Overview

|     Component    | Python Version | Primary Directory |     Activation Command     |        Executes        |  Port  |
|:----------------:|:--------------:|:-----------------:|:--------------------------:|:----------------------:|:------:|
|   Web Interface  |      3.7.6     |        `~/`       |   `conda activate arise`   |      `python3 ~/server.py`     |  5000  |
| ARISE MTL Model Interface |      3.7.6     |        `~/`       |   `conda activate arise`   |   `python3 ~/controller.py`    |  5555  |
|    NLU Engine    |      3.8.5     |      `~/nlp/`     |   `conda activate nlp`   | `~/nlp/nlu-engine/run` |  5005  |
|   Action Server  |      3.8.5     |    `~/webhook/`   | `conda activate nlp` |     `~/webhook/run`    |  5055  |

## Using Virtual Environments with Conda

First, download and install Miniconda.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Then from the root of the repository, run: (this may take awhile)

`conda create -n arise python=3.7.6`

Activate the environment with

`conda activate arise`

And install the remaining requirements with 

`pip install -r ./arise/requirements.txt`

Once this is complete, open a second terminal window and create the NLP environment for Python 3.8.5 using `conda env create -n nlp python=3.8.5`, then activate with `conda activate nlp`, and install the requirements with `pip install -r nlp-requirements.txt`.

## Natural Language Understanding (NLU) Engine  

The `rasa` NLU engine is responsible for parsing natural language and extracting entities and other information to inform the subsequent actions executed by the action server.

To run this component, navigate to `~/nlp/nlu-engine` and execute `./run` while in the NLP conda environment (`conda activate nlp`) to bring up the model in the background (i.e., expose the model to REST API for remote interaction).

The model can be retrained by modifying the data `.yml` files in this directory and subdirectories, and subsequently retrained with `rasa train`.

### Adding new tasks to the NLU engine
To incorporate new tasks, modify the `domain.yml`, `stories.yml`, and `lookups.yml` in `nlp/nlu-engine/` and `nlp/nlu-engine/data/`. These files dictate what intents are supported/can be recognized by the NLP agent, and the actions they can call (which are stored in `webhook/actions/actions.py`). 

For more information on incorporating tasks or modifying the chatlog stories, see the [Rasa documentation pages](https://rasa.com/docs/rasa/stories/).

## Webhook Action Server  

To load the webhook action server, navigate to `~/webhook` and execute `./run` while in the NLP conda environment to expose the action server to the REST API. 

The default action server will use port `5055` on your local machine to communicate with the NLU engine and ARISE models.

## ARISE MTL Models

The multi-task weak supervision models based on [ARISE](https://gitlab.com/onrg/arise) are located in `~/arise/MTL/Multitask.ipynb` as an interactive notebook, or `~/controller.py` as executable code. Activate the ARISE conda environment with `conda activate arise` and initialize the model controller with `python3 controller.py`.

The labeling function and data management portions of the ARISE framework are defined in `arise.py`. To modify the labeling functions, adjust the `tsfresh` features or rewrite the labeling functions themselves. To allow flexibility/modulariy in LF definitions, it may be desirable to define the labeling functions in a Rasa story, rather than explicitly in code. 

## Chatbot Interface

To allow these components to interact with one another, activate the `arise` conda environment and run `python3 server.py` in the root directory of this repository. 
This will expose a local webserver on port `5000`, accessible at [localhost:5000](http://localhost:5000/) in a web browser. 
(Note: if running on a remote server, you will need to forward this port to your local machine using e.g., `ssh user@remoteserver -L 5000:localhost:5000` to then access it on your computer.)
This server will communicate with the various components of the framework to relay natural language to and from the operator, as well as display graphical results or perform other frontend duties.