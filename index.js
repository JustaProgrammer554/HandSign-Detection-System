// Establish a connection to the Socket.IO server
var socket = io();
console.log("DDDD")
// Listen for the 'sign_detected' event from the server
socket.on('sign_detected', function(data) {
    // Update the 'Sign' span with the recognized sign
    document.getElementById('Sign').innerText = data.sign;
    console.log(data.sign)
});