import os
import kagglehub as kh
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

def download_model(model_url):
    down_model_path = kh.model_download(model_url)
    return down_model_path

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
               'potted plant', 'bed', 'dining table', 'toilet', 'TV', 
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
               'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
               'toothbrush']

def detect_objects(image, model):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...] 

    detections = model(input_tensor)

    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()

    return detection_boxes, detection_classes, detection_scores

def count_people(detection_boxes, detection_classes, detection_scores, threshold=0.5):
    if len(detection_classes.shape) != 1:
        detection_classes = detection_classes.flatten()
    if len(detection_scores.shape) != 1:
        detection_scores = detection_scores.flatten()
    if len(detection_boxes.shape) != 2:
        detection_boxes = detection_boxes.reshape(detection_boxes.shape[0], -1)
    
    data = {
        'class': detection_classes,
        'score': detection_scores,
        'box': [box.tolist() for box in detection_boxes]
    }
    
    df = pd.DataFrame(data)
    df['class_name'] = df['class'].apply(lambda x: class_names[x] if x < len(class_names) else 'unknown')

    person_df = df[(df['class_name'] == 'person') & (df['score'] > threshold)]

    person_count = person_df.shape[0]

    return person_count, person_df

def main():
    model_url = 'tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/640x640/1'

    down_model_path = download_model(model_url)

    model = load_model(down_model_path)

    image_path = 'image.jpg'  # substitua pelo nome da imagem
    image = Image.open(image_path)
    image = image.convert('RGB')  
    image_np = np.array(image)

    detection_boxes, detection_classes, detection_scores = detect_objects(image_np, model)

    person_count, person_df = count_people(detection_boxes, detection_classes, detection_scores)

    print(f'Pessoas na fila: {person_count}')

if __name__ == "__main__":
    main()
