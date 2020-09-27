import eel, os, random, sys, re
import time
import gzip
import csv
import hashlib
from random import shuffle

##

# import torchvision.models as models
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import *
from PIL import *
import urllib 
import sqlite3 
import pickle
import zlib
import math
import datetime


verbose = True

feature_store_path = "data/feature_store.db"

feature_store = sqlite3.connect(feature_store_path)


resnext50_model = models.resnext50_32x4d(pretrained=True)
# strip last layer of resnext:
modules=list(resnext50_model.children())[:-1]
resnext50_sll_model=nn.Sequential(*modules)


fasterrcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# ~255MB, will take some time to download first time

bicycle_label_coco = 2 # label within coco dataset
bicycle_label_oi = "/m/0199g" # label within open images dataset


class SimpleClassifier(nn.Module):  # inherit pytorch's nn.Module
    """ Classifier with 1 hidden layer 

    """
    
    def __init__(self, num_labels, num_inputs):
        super(SimpleClassifier, self).__init__() # call parent init

        # Define model with one hidden layer with 128 neurons
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec, return_all_layers=False):
        # Define how data is passed through the model and what gets returned

        hidden1 = self.linear1(feature_vec).clamp(min=0) # ReLU
        output = self.linear2(hidden1)
        log_softmax = F.log_softmax(output, dim=1)

        if return_all_layers:
            return [hidden1, output, log_softmax]
        else:
            return log_softmax
                                




COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# from: https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
# GET FASTER R CNN COCO DATASET PREDICTION FOR A BICYCLE
def get_fasterrcnn_prediction(img):
    fasterrcnn_model.eval()
    # img = Image.open(file) # Load the image
    transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image

    height = len(img[0])
    width = len(img[0][0])

 
    pred = fasterrcnn_model([img]) # Pass the image to the model
   
    pred_label = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  
  
  
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
  
    max_bike = 0.0
    bbox = [0, 0, width, height]
    for ind in range(0, len(pred_label)):
      # print(pred_label[ind])
      if pred_label[ind] == 'bicycle':
          if pred_score[ind] > max_bike:
              max_bike = pred_score[ind]
              bbox = pred_boxes[ind] # left, top, right, bottom

    # Option: return *every* bike box instead of just top

    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    
    # get [0-1] range ratio
    if box_width > box_height:
        ratio = (box_height / box_width) / 2
    else:
        ratio =  (2 - (box_width / height)) / 2
                
    bbox[0] = bbox[0] / width
    bbox[1] = bbox[1] / height
    bbox[2] = bbox[2] / width
    bbox[3] = bbox[3] / height
    
    width_scale = bbox[2] - bbox[0]
    height_scale = bbox[3] - bbox[1]
    
    horiz_center = (bbox[2] - bbox[0]) / 2
    vert_center = (bbox[3] - bbox[1]) / 2   
          
    return [max_bike, ratio, width_scale, height_scale, horiz_center, vert_center] + bbox



# GET RESNEXT50 IMAGENET DATASET PREDICTION FOR A BICYCLE
def get_resnext_features(img):

    # img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = resnext50_sll_model(input_batch)
        
    output = output.reshape(1, -1)  
    return output.squeeze().detach().tolist()
    

def make_feature_vector(image_id, url, label=""):
    # CHECK IF WE K?NOW ITS MISSING/BAD
    if url_is_missing(url) or is_bad_image(url):
        return None

    # CHECK IF WE'VE STORED IT
    feature_list = get_features_from_store(image_id)
    
    # EXTRACT FEATURES FROM COCO & IMAGENET MODELS
    if not feature_list:
        try:
            img = Image.open(urllib.request.urlopen(url))
        except urllib.error.HTTPError:        
            record_missing_url(url)
            return None

        try:
            imagenet_features = get_fasterrcnn_prediction(img)
            coco_features = get_resnext_features(img)            
            feature_list = imagenet_features + coco_features
            # Store it for fast reference next time 
            add_to_feature_store(image_id, feature_list, url, label)
        except RuntimeError:
            print("Problem with "+url)
            record_bad_image(url)
            return None

        
    
    vector = torch.Tensor(feature_list)    
    return vector.view(1, -1)
    

def load_annotations(annotation_filepath, image_filepath):
    global bicycle_label_oi
    annotations = {}
    annotated_data = []
    
    file = gzip.open(annotation_filepath, mode='rt')
    csvobj = csv.reader(file, delimiter = ',',quotechar='"')
    for row in csvobj:
        if row[2] == bicycle_label_oi:
            image_id = row[0]
            label = row[3]
            annotations[image_id] = label
                  
    file = gzip.open(image_filepath, mode='rt')    
    csvobj = csv.reader(file, delimiter = ',',quotechar='"')
    for row in csvobj:
        # ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
        image_id = row[0]
        if image_id in annotations:
            url = row[2]
            if url_is_missing(url) or is_bad_image(url):
                continue
                
            # TEMP:
            if not get_features_from_store(image_id):
                continue # TODO: remove later
                
            annotation = annotations[image_id]
            annotated_data.append([image_id,url,annotation])
                  
                  
    return annotated_data
    
  

def train_model(training_data, validation_data = "", evaluation_data = "", batch_size=100, num_epochs=1000, num_labels=2, num_inputs=2058, model=None):
    """Train model on the given training_data

    Tune with the validation_data
    Evaluate accuracy with the evaluation_data
    """

    if model == None:
        model = SimpleClassifier(num_labels, num_inputs)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epochs training
    for epoch in range(num_epochs):
        if verbose:
            print("Epoch: "+str(epoch))
        current = 0

        # make a subset of data to use in this epoch
        # with an equal number of items from each label

        shuffle(training_data) #randomize the order of the training data        
        bicycle = [row for row in training_data if '1' in row[2]]
        not_bicycle = [row for row in training_data if '0' in row[2]]
        
        epoch_data = bicycle[:batch_size]
        epoch_data += not_bicycle[:batch_size]
        shuffle(epoch_data) 
                
        # train our model
        for item in epoch_data:
            image_id = item[0]
            url = item[1]
            label = int(item[2])

            model.zero_grad() 

            feature_vec = make_feature_vector(image_id, url)
            if feature_vec == None:
                print("no features for "+url)
                continue
            
            target = torch.LongTensor([int(label)])

            log_probs = model(feature_vec)

            # compute loss function, do backward pass, and update the gradient
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()    


        fscore, auc = evaluate_model(model, evaluation_data)
        fscore = round(fscore,3)
        auc = round(auc,3)
        print("Fscore/AUC = "+str(fscore)+" "+str(auc))

    
    fscore, auc = evaluate_model(model, evaluation_data)
    fscore = round(fscore,3)
    auc = round(auc,3)

    # save model to path that is alphanumeric and includes number of items and accuracies in filename
    timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
    training_size = "_"+str(len(training_data))
    accuracies = str(fscore)+"_"+str(auc)
                     
    model_path = "models/"+timestamp+accuracies+training_size+".params"

    torch.save(model.state_dict(), model_path)
    
    
    return model
  
 

def evaluate_model(model, evaluation_data):
    """Evaluate the model on the held-out evaluation data

    Return the f-value for disaster-bicycle and the AUC
    """

    bicycle_confs = [] # bicycle items and their confidence of being bicycle
    not_bicycle_confs = [] # not bicycle items and their confidence of being _bicycle_

    true_pos = 0.0 # true positives, etc 
    false_pos = 0.0
    false_neg = 0.0

    with torch.no_grad():
        for item in evaluation_data:
            image_id = item[0]
            url = item[1]
            label = item[2]

            feature_vector = make_feature_vector(image_id, url)
            if feature_vector == None:
                continue
            log_probs = model(feature_vector)

            # get probability that item is bicycle
            prob_bicycle = math.exp(log_probs.data.tolist()[0][1]) 

            if(label == "1"):
                # true label is bicycle
                bicycle_confs.append(prob_bicycle)
                if prob_bicycle > 0.5:
                    true_pos += 1.0
                else:
                    false_neg += 1.0
                    print("FN: "+url)

            else:
                # no bicycle
                not_bicycle_confs.append(prob_bicycle)
                if prob_bicycle > 0.5:
                    false_pos += 1.0
                    print("FP: "+url)

    # Get FScore
    if true_pos == 0.0:
        fscore = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = (2 * precision * recall) / (precision + recall)

    # GET AUC
    not_bicycle_confs.sort()
    total_greater = 0 # count of how many total have higher confidence
    for conf in bicycle_confs:
        for conf2 in not_bicycle_confs:
            if conf < conf2:
                break
            else:                  
                total_greater += 1


    denom = len(not_bicycle_confs) * len(bicycle_confs) 
    auc = total_greater / denom

    return[fscore, auc]


 
  
def create_feature_tables():
    with feature_store:
        feature_store.execute("""
           CREATE TABLE IF NOT EXISTS feature (
               image_id TEXT NOT NULL PRIMARY KEY,
               url TEXT,
               features TEXT,
               label TEXT
            );        
        """)

        feature_store.execute("""
           CREATE TABLE IF NOT EXISTS url_missing (
               url TEXT NOT NULL PRIMARY KEY
            );        
        """)

        feature_store.execute("""
           CREATE TABLE IF NOT EXISTS bad_image (
               url TEXT NOT NULL PRIMARY KEY
            );        
        """)

        
def record_missing_url(url):
    sql = 'INSERT OR REPLACE INTO url_missing (url) values(?)'
    feature_store.executemany(sql, [(url,)])

def url_is_missing(url):
    with feature_store:
        data = feature_store.execute("SELECT * FROM url_missing WHERE url = '"+url+"'")
        for row in data: 
            return True # it exists      
        return False


def record_bad_image(url):
    sql = 'INSERT OR REPLACE INTO bad_image (url) values(?)'
    feature_store.executemany(sql, [(url,)])

def is_bad_image(url):
    with feature_store:
        data = feature_store.execute("SELECT * FROM bad_image WHERE url = '"+url+"'")
        for row in data: 
            return True # it exists
        return False

 
 
        

def add_to_feature_store(image_id, features, url="", label=""):
    sql = 'INSERT OR REPLACE INTO feature (image_id, url, features, label) values(?, ?, ?, ?)'
    pickled_features = pickle.dumps(features, pickle.HIGHEST_PROTOCOL)
    compressed_features = zlib.compress(pickled_features)
    feature_store.executemany(sql, [(image_id, url, compressed_features, str(label))])



def get_features_from_store(image_id):
    with feature_store:
        data = feature_store.execute("SELECT image_id, url, features, label FROM feature WHERE image_id = '"+image_id+"'")
        for row in data: 
            try:             
                compressed_features = row[2]                
                pickled_features = zlib.decompress(compressed_features)
                features = pickle.loads(pickled_features)
            except Exception as e:
                print("Couldn't load "+image_id+" : "+str(e))
                return False
            return(features)
        
        return False
    
create_feature_tables()


annotation_filepath = 'data/oidv6-train-annotations-human-imagelabels.csv.gz'    
image_filepath = 'data/oidv6-train-images-with-labels-with-rotation.csv.gz'

training_annotations = load_annotations(annotation_filepath, image_filepath)  

    
annotation_filepath = 'data/validation-annotations-human-imagelabels.csv.gz'
image_filepath = 'data/validation-images-with-rotation.csv.gz'  
  
validation_annotations = load_annotations(annotation_filepath, image_filepath)  
   
annotation_filepath = 'data/test-annotations-human-imagelabels.csv.gz'
image_filepath = 'data/test-images-with-rotation.csv.gz'  
  
evaluation_annotations = load_annotations(annotation_filepath, image_filepath)  
   
# TEMP TO TEST   
# training_annotations = validation_annotations

model = train_model(training_annotations, validation_annotations, evaluation_annotations)
