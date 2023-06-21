// Client Side Javascript to receive data
$(document).ready(function(){
    
    // start up the SocketIO connection to the server - the namespace 'test' is also included here if necessary
    // var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var socket = io.connect('http://' + document.domain + ':' + location.port);
    
    // this is a callback that triggers when the "my response" event is emitted by the server.
    socket.on('my response', function(msg) {
        console.debug(msg);

        let classes = msg.classes != "undefined" ? msg.classes : "";
        let data = msg.data != "undefined" ? msg.data : "";
        
        // if there is an image path attached to the message:
        if (msg.image) {
            // create the div and img object and append it
            $('#history').append(`
            <div class="action reply logtext logimg ${classes}">
                <img src='${msg.image}' class="reply image">
            </div>
            `)
        } else {
            // otherwise, append the text as normal
            $('#history').append(`<div class="action reply logtext ${classes}">${data}</div>`)
        }

        

        
        
        // scroll to bottom of div
        let log = document.getElementById("history");
        log.scrollTop = log.scrollHeight;
    });


    // debug callback
    socket.on('debug', function(msg) {
        console.debug(msg);

        let classes = msg.classes != "undefined" ? msg.classes : "";
        let data = msg.data != "undefined" ? msg.data : "";
        
        // if there is an image path attached to the message:
        if (msg.image) {
            // create the div and img object and append it
            $('#history').append(`
            <div class="action reply logtext logimg ${classes}">
                <img src='${msg.image}' class="reply image">
            </div>
            `)
        } else {
            // otherwise, append the text as normal
            $('#history').append(`<div class="action reply logtext ${classes} debug">${data}</div>`)
        }
        
        // scroll to bottom of div
        let log = document.getElementById("history");
        log.scrollTop = log.scrollHeight;
    });


    // image callback
    // debug callback
    socket.on('image', function(msg) {
        console.debug('received image path to post from path ' + msg.path);

        // create the div and img object and append it
        $('#history').append(`
        <div class="action reply logtext logimg">
            <img src='static/${msg.path}' class="reply image">
        </div>
        `)
        
        // $('#history').append(`
        // <div class="action reply logtext logimg">
        //     <img src='static/img/adaptive.png' class="reply image">
        // </div>
        // `)
        
        // scroll to bottom of div
        let log = document.getElementById("history");
        log.scrollTop = log.scrollHeight;
    });


    
    //example of triggering an event on click of a form submit button
    $('form#emit').submit(function(event) {
        socket.emit('my event', {data: $('#emit_data').val()});
        return false;
    });
    
    console.debug("socket functionality loaded");
});