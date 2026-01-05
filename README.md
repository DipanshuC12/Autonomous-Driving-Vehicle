End-to-End Self-Driving Car (Behavioral Cloning)
A deep learning project that teaches a car to drive autonomously on a virtual track using behavioral cloning techniques.

This project uses a Convolutional Neural Network (CNN) based on the NVIDIA architecture to predict steering angles from raw camera images. The system features a robust Universal Controller capable of handling sharp turns, shadows, and high-speed driving (up to 28 MPH) with active braking logic.

🌟 Key Features
NVIDIA CNN Architecture: A powerful 9-layer network optimized for self-driving tasks.

"Ruthless" Data Balancing: Automatically removes 80% of "straight road" data to force the model to learn cornering.

Advanced Augmentation: Includes random Shadow Injection, Zoom, Pan, and Brightness shifts to generalize to new tracks.

Universal PID Controller: A custom testSim.py controller that adapts sensitivity based on road curvature.

Active Speed Logic:

Cruising: 28 MPH on straights.

Cornering: Automatically brakes to 9 MPH for sharp turns.

Hill Climb Assist: Boosts throttle on steep inclines (Jungle Track).

⚙️ Methodology
1. Data Collection & "Recovery" Strategy
Instead of just driving perfectly, the dataset was built using specific drills:

Center Driving: 2 laps of smooth driving.

Zig-Zag Maneuvers: Intentionally weaving left/right to teach recovery.

Corner Saves: Driving toward a wall and recording the sharp correction back to the center.

2. Preprocessing
Every image passes through a pipeline before entering the model:
2. Preprocessing
Every image passes through a pipeline before entering the model:

YUV Color Space: Used by NVIDIA for better lane feature detection.

Cropping: Removed the top 60px (sky/trees) and bottom 25px (car hood).

Blurring: Gaussian Blur (3x3) to remove noise.

YUV Color Space: Used by NVIDIA for better lane feature detection.

Cropping: Removed the top 60px (sky/trees) and bottom 25px (car hood).

Blurring: Gaussian Blur (3x3) to remove noise.

🚀 How to Run
Prerequisites
Python 3.x

TensorFlow / Keras

OpenCV (cv2)

SocketIO / Flask (for the simulator connection)

Udacity Self-Driving Car Simulator

📝 Credits

Based on the behavioral cloning concepts from the Udacity Self-Driving Car Nanodegree. Enhanced with custom augmentation and control logic.
