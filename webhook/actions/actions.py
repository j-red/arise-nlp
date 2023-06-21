""" Network Wizard Webhook actions """
# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/

import requests
import traceback
import json
import logging
log = logging.getLogger(__name__)

from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# from utils import config
from database import client
from nile import builder, compiler

from .parser import parse_feedback
from .response import make_card_response, make_simple_response, reset_output_context
from .beautifier import beautify_intent

import datetime as dt
from actions.task import *
from timefhuman import timefhuman # for parsing natural language datetimes
datetime_format = "%Y-%m-%d %H:%M:%S"

DB_STORE = False # should actions attempt to contact the database?

MTL_HOST = "localhost"
MTL_PORT = "5000" # flask server
ARISE_HOST = "0.0.0.0"
ARISE_PORT = "5555" # backend ML server

FORMAT = '%a, %B %-d at %I:%M%p'

PREV_QUERY = []

ACTIVE_TASKS = [] # currently loaded model

def chat(msg, debug=False):
    if debug:
        return requests.post(f"http://{MTL_HOST}:{MTL_PORT}/debug", data=f'{{ "sender": "actions.py", "data": "{str(msg)}" }}')
    else:
        return requests.post(f"http://{MTL_HOST}:{MTL_PORT}/log", data=f'{{ "sender": "actions.py", "data": "{str(msg)}" }}')


class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_show_time"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # print("TRACKER", json.dumps(tracker, indent=4, sort_keys=True))
        dispatcher.utter_message(text=f"{dt.datetime.now()}")

        return []

    
class ActionDebug(Action):
    def name(self) -> Text:
        return "action_debug"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # print("TRACKER", json.dumps(tracker, indent=4, sort_keys=True))
        print("\nHELLO WORLD\n")
        dispatcher.utter_message(text=f"The time is currently: {dt.datetime.now()}")
        return []

    
class ActionDetect(Action):
    def name(self) -> Text:
        return "action_detect"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY
        
        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            # print("type of entities:", type(entities), entities)
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")

            for entity in entities:

                NEW_ENTITIES[entity["entity"]] = entity

                print(f"Entity: {entity['entity']},  Value: {entity['value']}")
                # print(entity["start"], entity["end"])
            print()

            # SUBTASKS = [ent["value"] for ent in entities if ent["entity"] == "subtask"]
            SUBTASKS = []
            START = ""
            END = ""
            
            # -----------------------------------------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
            
            # if "window" in ENTS:
            #     pass
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                    SUBTASKS.append(ent["value"])
                
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
            print(f"Attempting to parse start and end times as natural language... Start=`{START}`, END=`{END}`", end="\n")
            
            task = ClassificationTask(SUBTASKS, start=START, end=END)
            
            print(f"\n{task}\n")
            
            # -----------------------------------------------------------------------------------------------
            
            # intent = builder.build(entities)
            intent = builder.build(NEW_ENTITIES)
            # print("Intent:", intent)

            speech = "Is this what you want?"
            response = make_card_response(
                "Nile Intent",
                intent,
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            #### tracking
            if DB_STORE:
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))

        # dispatcher.utter_message(text="Constructing task...") 
        # dispatcher.utter_message(text=f"Is this correct? {response}")
        
        try:
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'Subtasks': SUBTASKS})
            postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json=task.as_dict(), timeout=60)
            print(postreq.text)
            
            # dispatcher.utter_message(text=f"Registered query for {postreq.text} classification.")
            # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)}\tEnd: {task.end.strftime(FORMAT)}")
            
            getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", timeout=60)
            
            # current_query = task_from_dict(eval(getreq.text))
            current_query = getreq.form

            dispatcher.utter_message(text=f"Current Query: {current_query}")
        except:
            print("Error in Action_Detect")
        
        PREV_QUERY = SUBTASKS

        return response

    
class ActionModifyQuery(Action):
    def name(self) -> Text:
        return "action_modify"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY
        
        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")            
            SUBTASKS = []
            START, END = "", ""
            
            # -----------------------------------------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                        SUBTASKS.append(ent["value"])
                        
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
        except ValueError as err:
            traceback.print_exc()
            response = make_simple_response("{}".format(err))
        # dispatcher.utter_message(text=f"Is this correct? {response}")
        # dispatcher.utter_message(text=f"MODIFY_TIMES request received. New Start: '{START}', End: '{END}'")
        
        postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'Subtasks': SUBTASKS, 'Start': START, 'End': END})
        print(postreq.text)
        
        # dispatcher.utter_message(text=f"Registered query_exists for {postreq.text}.")    
        # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)} \tEnd: {task.end.strftime(FORMAT)}")

        # getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi") # this prints the current time the request was received
        # dispatcher.utter_message(text=f"Start time: {getreq.text}")
        
        getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi")
        
        try:
            current_query = task_from_dict(eval(getreq.text))
            dispatcher.utter_message(text=f"Updated Query: {current_query}")
        except:
            dispatcher.utter_message(text=f"Invalid Query attempt in modify_query with text " + getreq.text)
        
        if (SUBTASKS != []):
            PREV_QUERY = SUBTASKS

        return


class ActionCompareQuery(Action):
    def name(self) -> Text:
        return "action_compare"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # print("TRACKER", json.dumps(tracker, indent=4, sort_keys=True))
        dispatcher.utter_message(text=f"COMPARE_QUERY request received.")

        dispatcher.utter_message(text=f"There was a changepoint in the traffic approximately 12 hours ago. The latency is now an average of 600ms higher.")

        return []
    
    
class ActionQuery(Action):
    def name(self) -> Text:
        return "action_query"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY
        
        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            # print("type of entities:", type(entities), entities)
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")

            for entity in entities:

                NEW_ENTITIES[entity["entity"]] = entity

                print(f"Entity: {entity['entity']},  Value: {entity['value']}")
                # print(entity["start"], entity["end"])
            print()

            # SUBTASKS = [ent["value"] for ent in entities if ent["entity"] == "subtask"]
            SUBTASKS = []
            START = ""
            END = ""
            
            # -----------------------------------------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
            
            # if "window" in ENTS:
            #     pass
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                    SUBTASKS.append(ent["value"])
                
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
            print(f"Attempting to parse start and end times as natural language... Start=`{START}`, END=`{END}`", end="\n")
            
            task = ClassificationTask(SUBTASKS, start=START, end=END)
            
            print(f"\n{task}\n")
            
            # -----------------------------------------------------------------------------------------------
            
            # intent = builder.build(entities)
            intent = builder.build(NEW_ENTITIES)
            # print("Intent:", intent)

            speech = "Is this what you want?"
            response = make_card_response(
                "Nile Intent",
                intent,
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            #### tracking
            if DB_STORE:
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))

        # dispatcher.utter_message(text="Constructing task...") 
        # dispatcher.utter_message(text=f"Is this correct? {response}")
        
        try:
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'Subtasks': SUBTASKS})
            postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json=task.as_dict(), timeout=60)
            print(postreq.text)
            
            # dispatcher.utter_message(text=f"Registered query for {postreq.text} classification.")
            # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)}\tEnd: {task.end.strftime(FORMAT)}")
            
            getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", timeout=60)
            
            current_query = SUBTASKS
            # current_query = task_from_dict(eval(getreq.text))

            dispatcher.utter_message(text=f"Current Query: {current_query}")

            if (SUBTASKS != []):
                PREV_QUERY = SUBTASKS

        except:
            print("Error in Action_Query")
        
        return response


class ActionFind(Action):
    def name(self) -> Text:
        return "action_find"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY

        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")

            for entity in entities:
                NEW_ENTITIES[entity["entity"]] = entity

                print(f"Entity: {entity['entity']},  Value: {entity['value']}")
                # print(entity["start"], entity["end"])
            print()

            # SUBTASKS = [ent["value"] for ent in entities if ent["entity"] == "subtask"]
            SUBTASKS = []
            START = ""
            END = ""
            
            # -----------------------------------------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
            
            # if "window" in ENTS:
            #     pass
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                    SUBTASKS.append(ent["value"])
                
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
            print(f"Attempting to parse start and end times as natural language... Start=`{START}`, END=`{END}`", end="\n")
            
            task = FindQuery(SUBTASKS, start=START, end=END)
            
            print(f"\n{task}\n")
            
            # -----------------------------------------------------------------------------------------------
            
            # intent = builder.build(entities)
            intent = builder.build(NEW_ENTITIES)
            # print("Intent:", intent)

            speech = "Is this what you want?"
            response = make_card_response(
                "Nile Intent",
                intent,
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            #### tracking
            if DB_STORE:
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))

        # dispatcher.utter_message(text="Constructing task...") 
        # dispatcher.utter_message(text=f"Is this correct? {response}")
        
        try:
            postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'operation': 'find', 'subtask': SUBTASKS})
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json=task.as_dict(), timeout=60)
            # print(postreq.text)

            if (SUBTASKS == []):
                SUBTASKS = PREV_QUERY
            
            dispatcher.utter_message(text=f"Registered FIND query for `{SUBTASKS}`.")
            # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)}\tEnd: {task.end.strftime(FORMAT)}")
            
            # getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", timeout=60)
            
            # current_query = task_from_dict(eval(getreq.text))
            # dispatcher.utter_message(text=f"Current Query: {current_query}")
        except:
            print("Error in Action FIND")
        return response
 
    
class ActionInfer(Action):
    def name(self) -> Text:
        return "action_infer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY
        
        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")

            for entity in entities:
                NEW_ENTITIES[entity["entity"]] = entity

                print(f"Entity: {entity['entity']},  Value: {entity['value']}")
                # print(entity["start"], entity["end"])
            print()

            # SUBTASKS = [ent["value"] for ent in entities if ent["entity"] == "subtask"]
            SUBTASKS = []
            START = ""
            END = ""
            
            # ------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
            
            # if "window" in ENTS:
            #     pass
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                    SUBTASKS.append(ent["value"])
                
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
            # print(f"Attempting to parse start and end times as natural language... Start=`{START}`, END=`{END}`", end="\n")
            
            if (SUBTASKS == []):
                SUBTASKS = PREV_QUERY
            else:
                PREV_QUERY = SUBTASKS

            task = InferQuery(SUBTASKS, start=START, end=END)
            
            print(f"\n{task}\n")
            
            # -----------------------------------------------------------------------------------------------
            
            # intent = builder.build(entities)
            intent = builder.build(NEW_ENTITIES)
            # print("Intent:", intent)

            speech = "Is this what you want?"
            response = make_card_response(
                "Nile Intent",
                intent,
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            #### tracking
            if DB_STORE:
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))
        try:
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'operation': 'find', 'subtask': SUBTASKS})
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json=task.as_dict(), timeout=60)
            # print(postreq.text)
            
            # dispatcher.utter_message(text=f"INFER request received.")
            chat("Received INFER request.")
            chat("INFER Request received", debug=True)
            # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)}\tEnd: {task.end.strftime(FORMAT)}")

            #### TODO: compare with any updates, changes, congestion, etc.
            # and look for potential sources of causality.
            # e.g., run congestion detection to determine if that is why it is unusual?

            # dispatcher.utter_message(text=f"It looks like there are high levels of congestion (~73%) in this period of time. There was also a system update 16 hours prior.")
            chat(f"It looks like there are high levels of congestion (~73%) in this period of time. There was also a system update 16 hours prior.")
            
            # getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", timeout=60)
            # current_query = task_from_dict(eval(getreq.text))
            # dispatcher.utter_message(text=f"Current Query: {current_query}")
        except:
            chat("Error in Action INFER", debug=True)
        return response


class ActionDefine(Action):
    def name(self) -> Text:
        return "action_define"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY
        
        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")

            for entity in entities:
                NEW_ENTITIES[entity["entity"]] = entity

                print(f"Entity: {entity['entity']},  Value: {entity['value']}")
            print()

            
            SUBTASKS = []
            START = ""
            END = ""
            
            # ------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
            
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                    SUBTASKS.append(ent["value"])
                
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
            # print(f"Attempting to parse start and end times as natural language... Start=`{START}`, END=`{END}`", end="\n")
            
            if (SUBTASKS == []):
                SUBTASKS = PREV_QUERY
            else:
                PREV_QUERY = SUBTASKS

            task = InferQuery(SUBTASKS, start=START, end=END)
            
            print(f"\n{task}\n")
            
            # -----------------------------------------------------------------------------------------------
            
            # intent = builder.build(entities)
            intent = builder.build(NEW_ENTITIES)
            speech = "Is this what you want?"

            response = make_card_response(
                "Nile Intent",
                intent, 
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            #### tracking
            if DB_STORE:
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))
        try:
            postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'operation': 'find', 'subtask': SUBTASKS})
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json=task.as_dict(), timeout=60)
            # print(postreq.text)
            
            dispatcher.utter_message(text=f"Registered DEFINE query for `{SUBTASKS}`.")
            # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)}\tEnd: {task.end.strftime(FORMAT)}")
            
            # getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", timeout=60)
            
            # current_query = task_from_dict(eval(getreq.text))
            # dispatcher.utter_message(text=f"Current Query: {current_query}")
        except:
            print("Error in Action DEFINE")
        return response


class ActionRefresh(Action):
    def name(self) -> Text:
        return "action_refresh"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        global PREV_QUERY
        
        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = self.name

        try:
            entities = tracker.latest_message["entities"]
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")

            for entity in entities:
                NEW_ENTITIES[entity["entity"]] = entity

                print(f"Entity: {entity['entity']},  Value: {entity['value']}")
            print()

            
            SUBTASKS = []
            START = ""
            END = ""
            
            # ------------------------------------------------------------
    
            ENTS = [ent['entity'] for ent in entities] # get list of entity types
            
    
            for ent in entities:
                if ent["entity"] == "subtask" and ent["value"] not in SUBTASKS:
                    SUBTASKS.append(ent["value"])
                
                if "window" in ENTS:
                    if ent["entity"] == "window":
                        START = ent["value"][0]["value"] # first value in window should be start time
                        END = ent["value"][1]["value"]   # second is end time
                else:
                    if ent["entity"] == "start":
                        for i in range(len(ent["value"])):
                            if not START:
                                START = ent["value"][i]["value"]
                            else:
                                END = ent["value"][i]["value"]

                    if ent["entity"] == "time" or ent["entity"] == "date":
                        if not START:
                            START = ent["value"]
                        else:
                            END = ent["value"]
            
            # print(f"Attempting to parse start and end times as natural language... Start=`{START}`, END=`{END}`", end="\n")
            
            if (SUBTASKS == []):
                SUBTASKS = PREV_QUERY
            else:
                PREV_QUERY = SUBTASKS

            task = InferQuery(SUBTASKS, start=START, end=END)
            
            print(f"\n{task}\n")
            
            # -----------------------------------------------------------------------------------------------
            
            # intent = builder.build(entities)
            intent = builder.build(NEW_ENTITIES)
            speech = "Is this what you want?"

            response = make_card_response(
                "Nile Intent",
                intent, 
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            #### tracking
            if DB_STORE:
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))
        try:
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json={'operation': 'find', 'subtask': SUBTASKS})
            # postreq = requests.post(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", json=task.as_dict(), timeout=60)
            # print(postreq.text)
            
            dispatcher.utter_message(text=f"Registered REFRESH command.")
            # dispatcher.utter_message(text=f"Start: {task.start.strftime(FORMAT)}\tEnd: {task.end.strftime(FORMAT)}")
            
            # getreq = requests.get(f"http://{MTL_HOST}:{MTL_PORT}/taskapi", timeout=60)
            
            # current_query = task_from_dict(eval(getreq.text))
            # dispatcher.utter_message(text=f"Current Query: {current_query}")
        except:
            print("Error in Action REFRESH")
        return response
    
    
class ActionPenTest(Action):
    def name(self) -> Text:
        return "action_pentest"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global ACTIVE_TASKS
        try:
            entities = tracker.latest_message["entities"]
            print("\nENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True), "\n")
        except:
            entities = []
        
        targets = [i['value'] for i in entities if i['entity'] == "host"]

        # dispatcher.utter_message(text=f"Examining communication between {targets[0].title()} and {targets[1].title()}. Please wait.")
        chat("Examining host communication...", debug=False)

        #### TODO: 
        # - select flow data path between specified hosts
        # - use ARISE LOSS task to determine if hosts are communicating
        # - respond accordingly
        ###
        
        
        # Set the list of tasks to train
        log.debug("===== Posting list of tasks =====")
        chat("Posting list of tasks", debug=True)
        
        _TASKS = ["loss", "noise", "congestion"] # tasks for the new model
        
        try:
            postreq = requests.post(f"http://{ARISE_HOST}:{ARISE_PORT}/set_tasks", json={"tasks": _TASKS})
            # update tasks after reloading the model
        except:
            log.error("Error posting new list of tasks to train. Is controller.py available?")
            
        
        # Reload the data with appropriate labels
        log.debug("===== Reloading model =====")
        chat("Reloading model", debug=True)
        try:
            if (ACTIVE_TASKS != _TASKS):
                postreq = requests.post(f"http://{ARISE_HOST}:{ARISE_PORT}/reset")
                ACTIVE_TASKS = _TASKS
            else:
                chat("All tasks already loaded. No need to reload model.", debug=True)
        except:
            log.warning("Error reseting model.")

        # Train the model with the loaded tasks (will load if cached)
        log.debug("===== Training model =====")
        chat("Training model", debug=True)
        try:
            # postreq = requests.post(f"http://{ARISE_HOST}:{ARISE_PORT}/train", json={"lr": 0.01, "epochs": 3, "range": 2})
            postreq = requests.post(f"http://{ARISE_HOST}:{ARISE_PORT}/train", json={"lr": 0.01, "epochs": 5})
        except:
            log.warning("Error posting request to train model.")

        
        # Query the results
        log.debug("===== Querying results =====") 
        chat("Querying results", debug=True)
        try:
            getreq = requests.get(f"http://{ARISE_HOST}:{ARISE_PORT}/get_scores")
            # log.info(f"Scores: \n{getreq.json()}")
            chat(f"Scores received", debug=True)
            # Scores are easily readable in getreq.json()
        except:
            log.warning("Failed to fetch results.")
        
        # TODO: Get specific predictions back to determine if hosts are/are not communicating
        log.debug("===== Fetching predictions =====") 
        chat("Fetching predictions", debug=True)
        try:
            predreq = requests.get(f"http://{ARISE_HOST}:{ARISE_PORT}/get_preds")
            log.info(f"Preds received")
            # Predictions are easily readable in predreq.json(), which is a dictionary with a field for each task
            # e.g., {'loss' : [0,0,0,1,0,1,...], 'noise' : [...], 'congestion' : [...]}
            
            log.info(f"Received prediction info for {predreq.json().keys()}")
        except:
            log.warning("Failed to fetch predictions. Abandoning execution.")
            return []
        
        _results = predreq.json() # {'loss' : [0,0,0,1,0,1,...], 'noise' : [1, 0, ...], 'congestion' : [...]}
        # results = predreq.json() # {0: {'loss' : [0,0,0,1,0,1,...], 'noise' : [1, 0, ...], 'congestion' : [...]}, 1: {...}}
        
        """
        To determine if hosts are communicating:
        We use a loss threshold, meaning if more than some percent of measurements are NOT loss, then
        the hosts are communicating with one another.
        """
        # log.debug(f"There are {len(results.keys())} entries to handle.")        
        # _results = results[0] # TODO: process over all possible
        
        assert len(_results['loss']) > 0 # verify there are entries
        
        comm_threshold = 0.1 # if > 10% of traffic is not loss, hosts are communicating
        losses = _results['loss'].count(1) # count number of positive ('True') predicted labels
        loss_ratio = losses / float(len(_results['loss']))
        
        chat(f"Loss count: {losses}. Total entries: {len(_results['loss'])}. Loss ratio: {round(loss_ratio * 100, 2)}%", debug=True)
        
        communicating = loss_ratio < (1 - comm_threshold)
        
        # dispatcher.utter_message(text=f"It would appear that these hosts are {'not' if not communicating else ''} communicating right now.")
        if (communicating):
            chat(f"These hosts appear to be communicating. Approximately {round(((1 - loss_ratio)*100), 2)}% of the traffic between these hosts is active.")
        else:
            chat(f"These hosts are not communicating right now.")
        
        
        #### TODO : load next model in preparation for next query
        # - modify load model to check which model is active and only refresh if necessary
        # - add infer query, compare query ML classifiers
        # - preload next model in background (with thread? daemon?)
        
        """
        Now, prepare preemptively for the next stages in the story:
          - use thread to do this in background?
          - set the expected next tasks; reload the model if needed
        """
        _NEXT_TASKS = ["noise", "congestion"]
        try:
            # Set the list of tasks to train
            log.debug("===== Pre-posting next list of tasks =====")
            postreq = requests.post(f"http://{ARISE_HOST}:{ARISE_PORT}/set_tasks", json={"tasks": _NEXT_TASKS})
            
            # Reload the data with appropriate labels
            log.debug("===== Pre-reloading model =====")
            if (ACTIVE_TASKS != _NEXT_TASKS):
                postreq = requests.post(f"http://{ARISE_HOST}:{ARISE_PORT}/reset")
                ACTIVE_TASKS = _NEXT_TASKS
            else:
                chat("All tasks already loaded. No need to preload model.", debug=True)
        except:
            chat(f"Failed to pre-load {", ".join(_NEXT_TASKS)} model.", debug=True)
        
        return []
    

    
""" The following Actions based on those provided in the Lumi source code. Not used in ARISE NLP. """

'''
class ActionBuild(Action):
    def name(self) -> Text:
        return "action_build"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """Webhook action to build Nile intent from Dialogflow request"""
        # print("Tracker:", tracker)
        # print("TRACKER", json.dumps(tracker, indent=4, sort_keys=True))

        uuid = ""  # request.get("session").split("/")[-1]
        text = ""  # request.get("queryResult").get("queryText")
        
        response = {}

        NEW_ENTITIES = {}
        NEW_ENTITIES["id"] = "action_build"

        try:
            entities = tracker.latest_message["entities"]
            print("type of entities:", type(entities), entities)
            print("ENTITIES: \n", json.dumps(entities, indent=4, sort_keys=True))

            for entity in entities:

                NEW_ENTITIES[entity["entity"]] = entity

                print(
                    "Entity: {},  Value: {}, Start: {}, End: {}".format(
                        entity["entity"],
                        entity["value"],
                        entity["start"],
                        entity["end"],
                    )
                )

            # intent = builder.build(entities)

            intent = builder.build(NEW_ENTITIES)

            speech = "Is this what you want?"
            response = make_card_response(
                "Nile Intent",
                intent,
                speech,
                beautify_intent(intent),
                suggestions=["Yes", "No"],
            )

            # tracking
            if (DB_STORE):
                db = client.Database()
                db.insert_intent(uuid, text, entities, intent)
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))

        dispatcher.utter_message(text="Under construction...")

        return response


class ActionDeploy(Action):
    def name(self) -> Text:
        return "action_deploy"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """Webhook action to deploy Nile intent after user confirmation"""
        uuid = ""  # request.get("session").split("/")[-1]
        
        try:
            if (DB_STORE):
                db = client.Database()
                intent = db.get_latest_intent(uuid)
                db.update_intent(intent["_id"], {"status": "confirmed"})

            merlin_program, _ = compiler.compile(intent["nile"]) # where is `intent[]` passed in?
            if merlin_program:
                if DB_STORE:
                    db.update_intent(
                        intent["_id"], {"status": "compiled", "merlin": merlin_program}
                    )
                return make_simple_response("Okay! Intent compiled and deployed!")

            # TODO: fix deploy API after user study
            # res = requests.post(config.DEPLOY_URL, {"intent": intent["nile"]})
            # if res.status["code"] == 200:
            #     return make_simple_response("Okay! Intent compiled and deployed!")
            #     db.update_intent(intent["_id"], {"status": "deployed"})
            
            return make_simple_response("Sorry. Something went wrong during deployment.")
        except:
            return make_simple_response("Sorry. Something went wrong during deployment.")


class ActionFeedback(Action):
    def name(self) -> Text:
        return "action_feedback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """Webhook action to receive feedback from user after rejecting built intent"""
        uuid = ""  # request.get("session").split("/")[-1]
        db = client.Database()
        intent = db.get_latest_intent(uuid)

        feedback = parse_feedback(tracker)
        entity_types = ["Location", "Group", "Middlebox", "Service", "Traffic"]

        query = tracker["queryResult"]["queryText"].lower()
        if "cancel" in query or "start over" in query:
            response = make_simple_response("Okay, cancelled. Please start over.")
            return reset_output_context(tracker, response)

            # slot-filling
        if "entity" not in feedback and "value" not in feedback:
            return make_simple_response(
                "First of all, what entity did I miss?", suggestions=entity_types
            )
        elif "entity" not in feedback:
            return make_simple_response(
                "What type of entity is '{}'?".format(feedback["value"]),
                suggestions=entity_types,
            )
        elif "value" not in feedback:
            suggestions = []
            for word in intent["text"].split():
                entities = intent["entities"].values()
                if word not in entities:
                    suggestions.append(word)
            return make_simple_response(
                "Great! And what word is a {}?".format(feedback["entity"]),
                suggestions=suggestions,
            )

        missing_entities = {}
        if "missingEntities" in intent:
            missing_entities = intent["missingEntities"]

        if feedback["entity"] not in missing_entities:
            missing_entities[feedback["entity"]] = {}
        missing_entities[feedback["entity"]][feedback["value"]] = True

        db.update_intent(
            intent["_id"], {"status": "declined", "missingEntities": missing_entities}
        )

        dispatcher.utter_message(text="Under construction...")
        print("training feedback", uuid, intent)
        return make_simple_response(
            "Okay! And is there anything else I missed?", suggestions=["Yes", "No"]
        )


class ActionFeedbackConfirm(Action):
    def name(self) -> Text:
        return "action_feedback_confirm"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """Webhook action to confirm feedback received from the user"""
        extracted_entities = tracker.latest_message["entities"]
        print("ENTITIES", json.dumps(extracted_entities, indent=4, sort_keys=True))

        dispatcher.utter_message(text="Under construction...")

        uuid = ""  # request.get("session").split("/")[-1]
        db = client.Database()
        intent = db.get_latest_intent(uuid)

        print("INTENT CONFIRM", intent)
        entities = intent["entities"]
        for entity, values in intent["missingEntities"].items():
            entity_key = entity
            if entity == "middlebox":
                entity_key = "middleboxes"
            elif entity == "service":
                entity_key = "services"
            elif entity == "traffic":
                entity_key = "traffics"
            elif entity == "protocol":
                entity_key = "protocols"
            elif entity == "operation":
                entity_key = "operations"
            elif entity == "location":
                entity_key = "locations"

            if entity_key not in entities:
                entities[entity_key] = list(values.keys())
            else:
                entities[entity_key] += list(values.keys())

        try:
            nile = builder.build(entities)
            speech = "So, is this what you want then?"
            response = make_card_response(
                "Nile Intent",
                nile,
                speech,
                beautify_intent(nile),
                suggestions=["Yes", "No"],
            )

            # tracking
            db.update_intent(intent["_id"], {"status": "pending", "nileFeedback": nile})
        except ValueError as err:
            traceback.print_exc()
            # TODO: use slot-filling to get the missing info
            # TODO: use different exceptions to figure out whats missing
            response = make_simple_response("{}".format(err))

        return response
'''