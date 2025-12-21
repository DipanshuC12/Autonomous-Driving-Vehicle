import ssl
# SSL FIX
if not hasattr(ssl, 'wrap_socket'):
    def wrap_socket(sock, keyfile=None, certfile=None,
                    server_side=False, cert_reqs=ssl.CERT_NONE,
                    ssl_version=ssl.PROTOCOL_TLS, ca_certs=None,
                    do_handshake_on_connect=True,
                    suppress_ragged_eofs=True,
                    ciphers=None):
        context = ssl.SSLContext(ssl_version)
        if certfile or keyfile: context.load_cert_chain(certfile, keyfile)
        if ca_certs: context.load_verify_locations(ca_certs)
        context.verify_mode = cert_reqs
        if ciphers: context.set_ciphers(ciphers)
        return context.wrap_socket(sock, server_side=server_side,
                                   do_handshake_on_connect=do_handshake_on_connect,
                                   suppress_ragged_eofs=suppress_ragged_eofs)
    ssl.wrap_socket = wrap_socket

import socketio
import eventlet
import numpy as np
import base64
from flask import Flask
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
model = load_model('model.h5', compile=False)

# ================= CONFIGURATION =================
# If the car understeers (hits outside wall), INCREASE this.
# If the car wobbles/oversteers (hits inside wall), DECREASE this.
STEERING_SENSITIVITY = 2.0 

MAX_SPEED = 20
TURN_SPEED = 8
# =================================================

def preProcessing(img):
    # MUST MATCH UTILS.PY EXACTLY
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data["speed"])
        
        # 1. Process Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = preProcessing(image)
        image = np.array([image])
        
        # 2. Predict
        raw_steering = float(model.predict(image, verbose=0)[0][0])
        
        # 3. Apply Sensitivity (The Fix)
        steering_angle = raw_steering * STEERING_SENSITIVITY
        
        # 4. Active Speed Control
        # If steering is sharp, slow down to TURN_SPEED. Otherwise MAX_SPEED.
        if abs(steering_angle) > 0.2:
            target_speed = TURN_SPEED
        else:
            target_speed = MAX_SPEED
            
        # 5. Throttle Logic
        if speed > target_speed:
            throttle = -0.5 # BRAKE HARD if overspeeding
        elif speed < target_speed:
            throttle = 1.0 - (speed/target_speed) # Accelerate gently
            if speed < 5 and target_speed > 8: throttle = 1.0 # Hill climb assist
        else:
            throttle = 0
            
        print(f'Raw: {raw_steering:.2f} | Adj: {steering_angle:.2f} | Speed: {speed:.1f}')
        sendControl(steering_angle, throttle)
    else:
        sio.emit('manual', data={})

@sio.on('connect')
def connect(sid, environ):
    print("Connected")
    sendControl(0, 0)

def sendControl(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)