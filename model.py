from video import stream_video
from utils import YOLOV3_SPP_WEIGHTS, YOLOV3_SPP_CFG, COCO_NAMES
import numpy as np
import cv2

def open_stream(video_url): # Uses opencv to correctly format the video stream to be properly used
    stream_url = stream_video(video_url)
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Could not open video stream!")
        exit()
    
    # Process this later to probably downscale using preprocessing.py
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {width}x{height}")

    return cap

def load_model(cap):
    net = cv2.dnn.readNet(YOLOV3_SPP_WEIGHTS, YOLOV3_SPP_CFG)

    if net.empty():
        print("Error: YOLO model failed to load")
        exit()
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(COCO_NAMES, 'r') as f: # Loads COCO Class names
        classes = [line.strip() for line in f.readlines()]
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): # Breaks look when key 'q' is pressed
            break
        
        ret, frame = cap.read() # Reads frame by frame
        if not ret:
            print("Failed to retrieve frame. Exiting program")
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)



        class_ids, confidences, boxes = [], [], []
        height, width, chanels = frame.shape

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5: # An object has been detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangular Coordinates around Object
                    x = int(center_x - w / 2)
                    y= int(center_y - h / 2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Drawing the bounding boxes
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0,255,0)
                cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x,y - 10), font, 1, color, 2)

        cv2.imshow("Real-Time Object Detection", frame)
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    scaffold = open_stream('https://www.youtube.com/watch?v=65KsVRG1ao8&ab_channel=MagicSwitzerland')
    load_model(scaffold)

main()        






