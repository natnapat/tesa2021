# WU one-hot ทีมที 12


#############################################################################################
#############################################################################################
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time
import requests
import urllib.parse
#import paho.mqtt.client as mqtt

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='.')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.75)
parser.add_argument('--video', help='Name of the video file',
                    default='test_clip.h264')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()
#url = 'http://165.22.251.170/api/push-data'
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu
myobj = {}
status = {"status": 1}
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)


def draw_text_on_image(img_draw, count):
    cv2.rectangle(img_draw, (0, 0), (500, 120), (0, 0, 0), -1)
    cv2.putText(img_draw, 'Count : ' + str(count),
                (10, 50),                  # bottomLeftCornerOfText
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.5,                      # fontScale
                (0, 255, 255),            # fontColor
                2)                        # lineType


# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
j = 0
count = 0
input_mean = 127.5
input_std = 127.5
elapsed_time = []

#######connect MQTT #########
# host = "192.168.43.177"
# port = 1883
# client = mqtt.Client()
# client.username_pw_set("tgr_user", "tgr_pass")
# client.connect(host)

#############################
toggle1 = False
toggle2 = False
###########################
# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)


#LINE_ACCESS_TOKEN = "2htvMQDYS8oFHy37ORge9VLxPMHDtZnhskj7s8l6jEW"
#URL_LINE = "https://notify-api.line.me/api/notify"


def line_text(message):
    msg = urllib.parse.urlencode({"message": message})
    #LINE_HEADERS = {'Content-Type': 'application/x-www-form-urlencoded',
    #                "Authorization": "Bearer "+LINE_ACCESS_TOKEN}
    #session = requests.Session()
    #session_post = session.post(URL_LINE, headers=LINE_HEADERS, data=msg)
    #print(session_post.text)


# def line_pic(message, path_file):
#     file_img = {'imageFile': open(path_file, 'rb')}
#     msg = ({'message': message})
#     LINE_HEADERS = {"Authorization": "Bearer "+LINE_ACCESS_TOKEN}
#     session = requests.Session()
#     session_post = session.post(
#         URL_LINE, headers=LINE_HEADERS, files=file_img, data=msg)
#     print(session_post.text)


#requests.post('http://165.22.251.170/api/push', data=status)
#line_text("Conveyor Status ON")

while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()

    if not ret:
        print('Reached the end of the video!')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_rgb = frame
    frame_resized = cv2.resize(frame_rgb, (320, 320))
    imgg = np.array(frame_resized)
    #frame_resized = cv2.resize(frame_rgb, (480, 320))
    input_data = np.expand_dims((frame_resized), axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    draw_text_on_image(frame, count)
    # Retrieve detection results
    # Bounding box coordinates of detected objects
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[
        0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[
        0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold

    for i in range(len(scores)):
        #host = "192.168.43.177"
        #port = 1883
        #client = mqtt.Client()
        #client.username_pw_set("tgr_user", "tgr_pass")
        
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            start_time = time.time()
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
            
            object_name = labels[int(classes[i])]
            #########   COUNTING HERE !!!!!! #################
            if xmin > 210 and xmax > 210:
                maybe = True
                do_once = False
            if xmax < 210 and maybe == True:
                if maybe == True and do_once == False:
                    count += 1
                    print("Count: " + str(count))
                    diameter = xmax - xmin
                    print("Diameter: "+str(diameter))
                    ####### SIZE DETERMINATION ###########
                    if diameter < 180 and object_name == "lime":
                        cv2.imwrite('defect_small_lime_' +
                                    str(count)+'.jpg', frame)
                        # line_pic("defect_small_lime_"+str(count), 'defect_small_lime_' +
                        #          str(count)+'.jpg')
                        # myobj = {"found": 'small_lime',
                        #          "qty": count}
                        #requests.post(url, data=myobj)
                        print("Captured Small lime")
                        #client.connect(host)
                        #client.publish("led/control","toggle1")
                    ##########################################
                    do_once = True
                maybe = False
                detected = True

            ##################################################

            # Draw label
            # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(
                scores[i]*100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            # Draw white box to put label text in

            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)

            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text
            end_time = time.time()
            elapsed_time.append((end_time - start_time))

    # All the results have been drawn on the frame, so it's time to display it.
    frame = cv2.resize(frame, (420, 320))
    cv2.line(frame, (210, 0), (210, 320), (0, 0, 255), 2)
    cv2.imshow('TFLITE Lime detector', frame)

    # Press 'q' to quit
    key = cv2.waitKey(int(1000/fps))
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('w'):
        cv2.imwrite('out_put_'+str(j)+'.jpg', frame)
        j = j + 1

# Clean
#line_text("Conveyor Status Off")
status = {"status": 0}
#requests.post('http://165.22.251.170/api/push', data=status)
print("Elapsed: " + str(sum(elapsed_time)))
print(fps)
video.release()
cv2.destroyAllWindows()
