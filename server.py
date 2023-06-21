from flask import Flask, render_template, request
# pip install flask-socketio==4.3.2 # old version required? no doesn't work
# pip install simple-websocket
from flask_socketio import SocketIO, emit
import requests
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = '6d2bdfec8e9843718756a1168081affd'
# socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins="*")

HOST, HOSTPORT = ("0.0.0.0", 5000)
NLU_HOST, NLU_PORT = ("localhost", 5005)


# To place images in the chat log:
# requests.post(f"http://{HOST}:{HOSTPORT}/log", data=f'{{ "image": "/static/img/adaptive.png"}}')


def reply(text : str):
    """ prints contents to the chatlog response window """
    socketio.emit('my response', {'data' : text})
    return

def reply_image(path : str):
    # return requests.post(f"http://{HOST}:{HOSTPORT}/log", data=f'{{ "image": path}}')
    return requests.post(f"http://{HOST}:{HOSTPORT}/log", json={"image": path})

def debug(text : str):
    """ logs debug text to the chat window """
    # # Example CURL usage:
    # # curl -X POST "localhost:5555/debug" -H "Content-type: application/json" -d '{"name":"alice", "data": "Bob"}'
    # content = request.get_json(force=True)
    # r = requests.post(f"http://{UI_HOST}:{UI_HOSTPORT}/debug", data=f'{{ "sender": "controller.py", "data": "{content["data"]}" }}')
    
    socketio.emit('debug', {'classes': 'debug', 'data' : text})
    return




@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api', methods = ['POST'])
def api():
    query = request.values.get('input', '').strip() # get input from javascript and trim whitespace
    
    if (query[0] in '/!#'): # if first char of query is a command operator !, /, or #
        if (query[0] == "!"):
            # treat ! as a direct command executor; will send to rasa api with leading /
            # this is useful for calling specific intents, e.g., '/debug' will trigger the debug intent
            query = query[1:] # trim first char
            r = requests.post(f"http://{NLU_HOST}:{NLU_PORT}/webhooks/rest/webhook", data=f'{{"message": "/{query}" }}')
            return ""

        query = query[1:] # trim first char


        # run a command
        if query == "help":
            return "Enter an input query in the box below."
        elif query == "clear":
            return "COMMAND-CLEAR"
        elif query == "lorem" or query == "lorum":
            return """
                Molestiae quam aperiam consequatur illum rerum ea dolores qui. Hic eos inventore enim eum voluptatibus recusandae. Amet suscipit distinctio similique qui similique dolorum provident. Aliquam vero magnam omnis.

                In aut sit et. Temporibus accusantium iste deleniti sint sed esse. Non culpa molestiae corrupti laborum inventore placeat vel. Esse inventore facilis quia non repudiandae iusto. Inventore et dolor voluptatum qui delectus est. Sit quia est sequi deserunt soluta distinctio harum.

                Consequatur magnam natus voluptate sunt dolor. Ullam esse doloribus iusto quo nisi numquam atque omnis. Facilis voluptates corrupti sunt consequatur similique sint vitae beatae. Et ipsum nam voluptas.

                Sint qui impedit culpa. Et doloribus aut omnis optio in. Nihil possimus qui officiis libero quod occaecati.

                Quidem minus dignissimos consequatur omnis tempora beatae. Ea in excepturi exercitationem est id qui. Quod maiores repellat placeat facere sed. Voluptate quasi perspiciatis impedit velit dolor ut doloribus harum.
                """
        else:
            return "Invalid command"  


    try:
        #### to send a query via command line:
        # curl -X POST localhost:5005/model/parse -d '{ "text": "hello" }'
        #### this request only PARSES the query, but does not run any actions from it
        # r = requests.post(f"http://{NLU_HOST}:{NLU_PORT}/model/parse", data=f'{{ "text": "{query}" }}')

        #### this request RUNS ACTIONS based on the query and returns the results
        #### webhook: https://forum.rasa.com/t/rest-api-implementation/12624/3
        # curl -X POST localhost:5005/webhooks/rest/webhook -d '{ "sender": "j4", "message": "hello" }'
        r = requests.post(f"http://{NLU_HOST}:{NLU_PORT}/webhooks/rest/webhook", data=f'{{ "sender": "", "message": "{query}" }}')
        
        # requests.post(f"http://0.0.0.0:5005/webhooks/rest/webhook", data=f'{{ "sender": "", "message": "hello" }}')
    except Exception as e:
        return("Error posting query to webhook endpoint:\n", e)

    # return this response to the server
    try:
        # response = r.json()[0]
        # socketio.emit('my response', {'data' : response['text']})
        # socketio.emit('my response', {'classes': 'debug', 'data' : r.text})
        for i in r.json():
            socketio.emit('my response', {'data' : i['text']})
        socketio.emit('my response', {'classes': 'debug', 'data' : r.text})

        # return response['text'] + "\n" + r.text

    except Exception as e:
        socketio.emit('my response', {'data' : "Sorry, I don't understand. Could you rephrase?"})
        socketio.emit('my response', {'classes': 'debug', 'data' : json.dumps(r.json(), indent=4)})
        
        # return f"Sorry, I don't understand. Could you rephrase?\n (response: {r.json()})"

    # return json.dumps(r.json(), indent=4)
    return ""


@app.route('/taskapi', methods = ['GET', 'POST'])
def taskapi():
    if request.method == "GET":
        print("Printing current task...")
        
        return "<current task>"
    else:
        print("Updating current task...")

        return "task updated"


@app.route('/log', methods = ['POST'])
def chatlog():
    """ append whatever resource is posted to the chatlog window. """
    # to post to this port: (note this is on port 5000)
    # requests.post(f"http://{HOST}:{HOSTPORT}/log", data=f'{{ "sender": "", "message": "hello", "data": "hello, world!" }}')

    content = request.get_json(force=True) # a dictionary of key, value pairs
    # the content['data'] field is what will be printed
    
    """
    files = list(request.files)
    if (len(files) > 0):
        # request included a file: assume it is an image and append?
        pass
        return
    """ # instead of sending files, we just send the resource path

    # curl -X POST localhost:5000/log -d '{ "sender": "j4", "message": "hello" }'
    # curl -XPOST http://localhost:5000/log -d '{ "data":"hello" }'
    
    # non-socket event handlers need to state `socketio.emit` instead of just emit
    # socketio.emit('my response', {'data': content['reply']})
    
    if 'data' in content.keys():
        reply(content['data']) # post data value to chatlog as response
        
    if 'image' in content.keys():
        # reply_image(content['image'])
        socketio.emit('image', {'path' : content['image']})
    
    return ""


@app.route('/debug', methods = ['POST'])
def debuglog():
    """ append whatever resource is posted to the chatlog debug window. """
    
    content = request.get_json(force=True) # a dictionary of key, value pairs
    return debug(content['data'])


@socketio.event # decorator to catch all events
def my_event(message): 
    print('my_event received a message:', message)
    emit('my response', {'data': 'got it!'})

@socketio.on('my event')   # Decorator to catch an event called "my event":
def test_message(message): # test_message() is the event callback function.
    print('test_message received a message:', message)
    emit('individual response', {'data': 'msg received!'}) # Trigger a new event called "individual response" 
                                                           # that can be caught by another callback later in the program.

if __name__ == "__main__":
    # app.run(HOST, port=HOSTPORT, debug=True)
    socketio.run(app, host=HOST, port=HOSTPORT, debug=True)


# curl -XPOST localhost:5005/model/parse -d '{ "text": "debug" }'