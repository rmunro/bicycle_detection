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
import requests
import sqlite3 
import pickle
import zlib
import math
import statistics
import datetime


verbose = True

feature_store_path = "data/feature_store.db"

# a subset of Open Images training data to be within 100MB github limit
training_labels_path = 'data/oidv6-train-annotations-human-imagelabels-reduced.csv.gz'    
training_images_path = 'data/oidv6-train-images-with-labels-with-rotation-reduced.csv.gz'
    
validation_labels_path = 'data/validation-annotations-human-imagelabels.csv.gz'
validation_images_path = 'data/validation-images-with-rotation.csv.gz'  
  
evaluation_labels_path = 'data/test-annotations-human-imagelabels.csv.gz'
evaluation_images_path = 'data/test-images-with-rotation.csv.gz'  
  
new_training_data_path = 'data/new-training-data.csv'

unlabeled_annotations = []  
validation_annotations = []  
evaluation_annotations = []

pending_annotations = [] # annotations pending being stored
new_training_data = {} # new training data by url

validation_urls = {} # validation item urls

new_annotation_count = 0
min_training_items = 5 # min items for each class to start training

high_uncertainty_items = [] # items queued for annotation because of uncertainty
model_based_outliers = [] # items queued for annotation because they are outliers and uncertain
number_sampled_to_cache = 1000 # how many to keep in memory to support rapid annotation

total_time = 0.0 # total time to download new images and extract features
total_downloads = 0 # total number of images downloaded 
total_pending = 0 # total number waiting to be processed

current_accuracies = []
current_model = None

feature_store = sqlite3.connect(feature_store_path)

eel.init('./')

# Download models. ~255MB, so will take some time to download first time
resnext50_model = models.resnext50_32x4d(pretrained=True)
modules=list(resnext50_model.children())[:-1] # strip last layer of resnext:
resnext50_sll_model=nn.Sequential(*modules)

fasterrcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

bicycle_label_coco = 2 # label within coco dataset
bicycle_label_oi = "/m/0199g" # label within open images dataset

image_id_urls = {} # image_ids indexed by url


class SimpleClassifier(nn.Module):  # inherit pytorch's nn.Module
    """ Linear Classifier with no hidden layers

    """
    
    def __init__(self, num_labels, num_inputs):
        super(SimpleClassifier, self).__init__() # call parent init
        self.linear = nn.Linear(num_inputs, num_labels)
        

    def forward(self, feature_vec, return_all_layers=False):
        # Define how data is passed through the model and what gets returned

        output = self.linear(feature_vec)
        log_softmax = F.log_softmax(output, dim=1)

        if return_all_layers:
            return [output, log_softmax]
        else:
            return log_softmax
    
    

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
     
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
  
    max_bike = 0.0
    bbox = [0, 0, width, height]
    for ind in range(0, len(pred_boxes)):
      if ind == bicycle_label_coco:
          if pred_score[ind] > max_bike:
              max_bike = pred_score[ind]
              bbox = pred_boxes[ind] # left, top, right, bottom
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    
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
    global total_time
    global total_downloads
    
    if url_is_missing(url) or is_bad_image(url):
        return None

    # CHECK IF WE'VE STORED IT
    feature_list = get_features_from_store(image_id)
    
    # EXTRACT FEATURES FROM COCO & IMAGENET MODELS
    if not feature_list:
        start_time = time.time()
        
        try:
            img = Image.open(urllib.request.urlopen(url))
        except urllib.error.HTTPError:        
            record_missing_url(url)
            return None

        try:
            imagenet_features = get_fasterrcnn_prediction(img)
            eel.sleep(0.1)
            coco_features = get_resnext_features(img)          
            eel.sleep(0.1)
  
            feature_list = imagenet_features + coco_features
            # Store it for fast reference next time 
            add_to_feature_store(image_id, feature_list, url, label)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_downloads += 1
            if verbose:
                print("average number of seconds to process new image: "+str(total_time/total_downloads))
            
        except RuntimeError:
            print("Problem with "+url)
            record_bad_image(url)
            return None
            
            

    vector = torch.Tensor(feature_list)    
    return vector.view(1, -1)
    

def load_training_data(filepath):
    # FOR ALREADY LABELED ONLY
    # csv format: [IMAGE_ID, URL, LABEL,...]
    global image_id_url
    
    if not os.path.exists(filepath):
        return []
    
    new_data = {}
    
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for item in reader:
            image_id = item[0]
            url = item[1]
            label = item[2]            
            new_data[url] = label
            image_id_urls[url] = image_id

    return new_data


def load_annotations(annotation_filepath, image_filepath, load_all = False):
    '''Load Open Images Annotations
      assume these are static, so we can pickle them to be loaded quicker     
    '''
    
    cached_data = get_data_structure_store(image_filepath)
    if cached_data:
        if verbose:
            print("loaded cached data "+image_filepath)
        return cached_data

    global bicycle_label_oi
    annotations = {}
    annotated_data = []
    
    c = 0
    
    file = gzip.open(annotation_filepath, mode='rt')
    csvobj = csv.reader(file, delimiter = ',',quotechar='"')
    for row in csvobj:
        if row[2] == bicycle_label_oi:
            image_id = row[0]
            label = row[3]
            annotations[image_id] = label
        c += 1
        if c == 10000:
            eel.sleep(0.01)
            c = 0
                  
    file = gzip.open(image_filepath, mode='rt')    
    csvobj = csv.reader(file, delimiter = ',',quotechar='"')
    for row in csvobj:
        # ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
        image_id = row[0]
        if image_id in annotations or load_all:
            url = row[2]
            thumbnail_url = row[10]
            
            if url_is_missing(url) or is_bad_image(url):
                continue
            if url_is_missing(thumbnail_url) or is_bad_image(thumbnail_url):
                thumbnail_url = url
                
            if image_id in annotations:                 
                label = annotations[image_id]
            else:
                #implicit negative
                label = 0
                
            annotated_data.append([image_id,url,label,thumbnail_url])
            image_id_urls[url] = image_id

        c += 1
        if c == 10000:
            eel.sleep(0.01)
            c = 0       
            
    store_data_structure(image_filepath, annotated_data)
                  
    return annotated_data
    
  

def train_model(batch_size=20, num_epochs=100, num_labels=2, num_inputs=2058, model=None):
    """Train model on the given training_data

    Tune with the validation_data
    Evaluate accuracy with the evaluation_data
    """
    global new_training_data
    global min_training_items

    if model == None:
        model = SimpleClassifier(num_labels, num_inputs)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if len(new_training_data) == 0:
        return None
        
    urls = list(new_training_data.keys())

    # epochs training
    for epoch in range(num_epochs):
        current = 0

        # make a subset of data to use in this epoch
        # with an equal number of items from each label
        
        bicycle = []
        not_bicycle = []

        shuffle(urls) #randomize the order of the training data 
        for url in urls:
            label = new_training_data[url]
            if len(bicycle) >= batch_size and len(not_bicycle) >= batch_size:
                break
            elif new_training_data[url] == "1" and len(bicycle) < batch_size:
                bicycle.append([image_id_urls[url], url, label])
            elif new_training_data[url] == "0" and len(not_bicycle) < batch_size:
                not_bicycle.append([image_id_urls[url], url, label])
        
        if len(bicycle) < min_training_items or len(not_bicycle) < min_training_items:
            if verbose or True:
                print("Not yet enough labels to train")
                print(len(bicycle))
                print(len(not_bicycle))
            return None
        
        epoch_data = bicycle + not_bicycle
        shuffle(epoch_data) 
 
        if verbose or True:
            print("Epoch: "+str(epoch))

                
        # train our model
        for item in epoch_data:
            image_id = item[0]
            url = item[1]
            label = int(item[2])
            if verbose or True:
                print(".")

            feature_vec = make_feature_vector(image_id, url)
            if feature_vec == None:
                print("no features for "+url)
                continue
            
            target = torch.LongTensor([int(label)])

            log_probs = model(feature_vec)
            eel.sleep(0.01) # let other processes in


            # compute loss function, do backward pass, and update the gradient
            loss = loss_function(log_probs, target)
            
            model.zero_grad() 

            loss.backward()
            optimizer.step()    


        fscore, auc, precision, recall, ave_loss = evaluate_model(model, False, 100)
        fscore = round(fscore,3)
        auc = round(auc,3)
        print("Fscore/AUC = "+str(fscore)+" "+str(auc)+" "+str(precision)+" "+str(recall))

    
    if model is not None:
        fscore, auc, precision, recall, ave_loss = evaluate_model(model, False, 1000)
        fscore = round(fscore,3)
        auc = round(auc,3)

        # save model to path that is alphanumeric and includes number of items and accuracies in filename
        timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
        training_size = "_"+str(len(urls))
        accuracies = str(fscore)+"_"+str(auc)
                     
        model_path = "models/"+timestamp+accuracies+training_size+".params"

        torch.save(model.state_dict(), model_path)
        # TODO: only replace model when loss and/or accuracy better on validation data
    
    return model
  
 
 

def evaluate_model(model, use_evaluation = True, limit = -1):
    """Evaluate the model on the held-out evaluation data

    Return the f-value for disaster-bicycle and the AUC
    """
    
    global evaluation_annotations
    global validation_annotations

    bicycle_confs = [] # bicycle items and their confidence of being bicycle
    not_bicycle_confs = [] # not bicycle items and their confidence of being _bicycle_

    true_pos = 0.0 # true positives, etc 
    false_pos = 0.0
    false_neg = 0.0
    true_neg = 0.0
    
    total_loss = 0.0
    
    loss_function = nn.NLLLoss()
    
    if use_evaluation:
        evaluation_data = evaluation_annotations
        if verbose:
            print("running evaluation data")
    else:
        evaluation_data = validation_annotations
        if verbose:
            print("running validation data")

    if len(evaluation_data) == 0:
        if verbose:
            print("evaluation data not loaded")
        return[0,0,0,0] # not loaded yet

    with torch.no_grad():
        count = 0
        for item in evaluation_data:
            if limit > 0 and count > limit:
                break   
        
            image_id = item[0]
            url = item[1]
            label = item[2]
            if verbose:
                print(".")

            feature_vector = make_feature_vector(image_id, url)
            if feature_vector == None:
                continue
            log_probs = model(feature_vector)
            eel.sleep(0.01)


            # get probability that item is bicycle
            prob_bicycle = math.exp(log_probs.data.tolist()[0][1]) 

            # record loss if we have a label
            if label != None:
                target = torch.LongTensor([int(label)])           
                loss = loss_function(log_probs, target)
                total_loss += loss

        
            if(label == "1"):
                # true label is bicycle
                bicycle_confs.append(prob_bicycle)
                if prob_bicycle > 0.5:
                    true_pos += 1.0
                elif prob_bicycle < 0.5:
                    false_neg += 1.0
            else:
                # no bicycle
                not_bicycle_confs.append(prob_bicycle)
                if prob_bicycle > 0.5:
                    false_pos += 1.0
                elif prob_bicycle < 0.5:
                    true_neg += 1.0
                    
            count += 1
            
    print(str(true_pos)+" "+str(false_pos)+" "+str(false_neg)+" "+str(true_neg))

    ave_loss = total_loss / len(evaluation_data)

    # Get FScore
    if true_pos == 0.0:
        fscore = 0.0
        precision = 0.0
        recall = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = (2 * precision * recall) / (precision + recall)

    # GET AUC
    not_bicycle_confs.sort()
    total_greater = 0 # count of how many total have higher confidence
    for conf in bicycle_confs:
        for conf2 in not_bicycle_confs:
            if conf <= conf2:
                break
            else:                  
                total_greater += 1

    denom = len(not_bicycle_confs) * len(bicycle_confs) 
    auc = total_greater / denom

    conf_b = statistics.mean(bicycle_confs)
    conf_n = statistics.mean(not_bicycle_confs)
    print("ave confs: "+str(conf_b)+" "+str(conf_n))
    print("ave loss: "+str(ave_loss))

    return[fscore, auc, precision, recall, ave_loss]



def get_quantized_logits(logits):
    ''' Returns the quanitized (0-1) logits
    
    '''
    # TODO: QUANTIZE WHEN EVALUATING VALIDATION DATA
    return 1- (logits[0] + logits[1])
    

def get_random_prediction(model):
    '''Get predictions on unlabeled data 
    
    '''
    global unlabeled_annotations
    global high_uncertainty_items 
    global model_based_outliers 
    global number_sampled_to_cache
    
    random.choose(unlabeled_annotations) 
    with torch.no_grad():        
        image_id = item[0]
        url = item[1]
        
        feature_vector = make_feature_vector(image_id, url)
        if feature_vector == None:
            return
            
        log_probs, logits = model(feature_vector, return_all_layers = True)     
    
        prob_bicycle = math.exp(log_probs.data.tolist()[0][1]) 
            
        least_conf = 2 * (1 - max(prob_bicycle, 1-prob_bicycle))

        outlier_score = get_quantized_logits(logits.data.tolist()[0])
            
        high_uncertainty_items = [] # items queued for annotation because of uncertainty
        model_based_outliers = [] # items queued for annotation because they are outliers and uncertain

        if len(high_uncertainty_items) < number_sampled_to_cache:
            if verbose or True:
                print("adding item to sampled list bc below cache")
            item[4] = least_conf
            high_uncertainty_items.append(item)
        elif least_conf > high_uncertainty_items[-1][0]:
            if verbose or True:
                print("adding item to sampled list bc high uncertainty")
            item[4] = least_conf
            high_uncertainty_items.append(item)
            high_uncertainty_items.pop(-1)
            high_uncertainty_items.sort(reverse=True, key=lambda x: x[4]) # TODO: RIGHT 
                
        if least_conf > 0.5:
            if len(model_based_outliers) < number_sampled_to_cache:
                if verbose or True:
                    print("adding item to sampled list bc below cache")
                item[4] = outlier_score
                model_based_outliers.append(item)
            elif least_conf > model_based_outliers[-1][0]:
                if verbose or True:
                    print("adding item to sampled list bc high outlier score")
                item[4] = outlier_score
                model_based_outliers.append(item)
                model_based_outliers.pop(-1)
                model_based_outliers.sort(reverse=True,key=lambda x: x[4])
        eel.sleep(0.1)        
    
 
 
 
 
  
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
        
        feature_store.execute("""
           CREATE TABLE IF NOT EXISTS data_structure (
               name TEXT NOT NULL PRIMARY KEY,
               data TEXT
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


def store_data_structure(structure_name, data):
    sql = 'INSERT OR REPLACE INTO data_structure (name, data) values(?, ?)'
    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    compressed_data = zlib.compress(pickled_data)
    feature_store.executemany(sql, [(structure_name, compressed_data)])
    
 
def get_data_structure_store(structure_name):
    with feature_store:
        data = feature_store.execute("SELECT name, data FROM data_structure WHERE name = '"+structure_name+"'")
        for row in data: 
            try:             
                compressed_data = row[1]                
                pickled_data = zlib.decompress(compressed_data)
                data = pickle.loads(pickled_data)
            except Exception as e:
                print("Couldn't load "+name+" : "+str(e))
                return False
            return(data)        
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



def add_pending_annotations():
    global pending_annotations
    global image_id_urls
    global new_training_data_path
    global total_pending
    
    while True:
        not_cached = 0

        # copy to avoid race conditions
        unprocessed_annotations = pending_annotations.copy()
        if verbose:
            print("processing "+str(len(unprocessed_annotations))+ " annotations")

        for annotation in unprocessed_annotations:
            url = annotation[0]
            image_id = image_id_urls[url]
            features = get_features_from_store(image_id)
            if not features:
                not_cached += 1
                
        total_pending = not_cached    
            
        while len(unprocessed_annotations) > 0:
            print("saving annotation")
            annotation = unprocessed_annotations.pop()
    
            url = annotation[0]
            is_bicycle = annotation[1]
            image_id = image_id_urls[url]
            if is_bicycle:
                label = 1
            else:
                label = 0
            append_data(new_training_data_path, [[image_id, url, label]])
            new_training_data[url] = label                
        
            if store_features:
                features = make_feature_vector(image_id, url, label)            
                eel.sleep(0.1) # allow other processes in
             
        eel.sleep(1)
        


def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()

 
@eel.expose 
def training_loaded():
    return len(unlabeled_annotations) > 0
       
@eel.expose 
def validation_loaded():
    return len(validation_annotations) > 0
       
 
@eel.expose   
def get_current_accuracies():
    global current_accuracies
    return current_accuracies   


@eel.expose
def estimate_processing_time():
    global total_time 
    global total_downloads 
    global total_pending 
    
    if total_downloads == 0:
        return 0 # no info yet
    else:
        return (total_time / total_downloads) * total_pending

@eel.expose
def add_annotation(url, is_bicycle):
    global pending_annotations
        
    if url not in validation_urls:
        if verbose:
            print("adding annotation for "+url)
        
        pending_annotations.append([url, is_bicycle])
        eel.sleep(0.01)
    else:
        if verbose:
            print("skipping validation: "+url)


@eel.expose
def get_next_image():
    global validation_annotations
    global unlabeled_annotations
    global test_annotations
    global high_uncertainty_items
    global model_based_outliers 
    
    annotations = unlabeled_annotations
    
    if len(validation_annotations) == 0:
        return [] # not yet loaded
    if len(unlabeled_annotations) == 0:
        return get_validation_image()
    
    
    strategy = random.randint(0,9)
    if strategy == 0:
        return get_validation_image()
    elif strategy == 1 or len(high_uncertainty_items) == 0:
        return get_random_image()
    elif strategy < 9:
        return get_uncertain_image()
    else:
        return get_outlier_image()
    

# get image with high uncertainty    
def get_uncertain_image():
    # TODO
    return []

    
# get image that is model-based outlier and also uncertain
def get_outlier_image():
    # TODO
    return []
    
    
        

def get_validation_image():
    global validation_annotations
    shuffle(validation_annotations)
    label = random.randint(0,1)
    for item in validation_annotations:
        if str(item[2]) != str(label):
            continue
          
        url = item[1]  
        thumbnail_url = item[3]
        if url_is_missing(url) or is_bad_image(url) or not test_if_url_ok(url):
            continue
        if not test_if_url_ok(thumbnail_url):
            thumbnail_url = url
        
        return [url, thumbnail_url, label]
        
    return [] # if there are no items
        
        

def get_random_image():
    global unlabeled_annotations

    url = ""
    while url == "":
        item = random.choice(unlabeled_annotations)
        image_id = item[0]
        url = item[1]
        label = "" # we're getting new labels so ignore OI ones            
        thumbnail_url = item[3]
    
        if url in new_training_data or url_is_missing(url) or is_bad_image(url):
            url = ""
            continue
        try:
            if not test_if_url_ok(url):
                url = ""
                break
            if not test_if_url_ok(thumbnail_url):
                thumbnail_url = url
                
            return [url, thumbnail_url, label] 

        except:
            print(" error with url "+url+" thumb "+thumbnail_url)
            url = ""

    
def test_if_url_ok(url):
    if len(url) == 0:
        return False
    response = requests.head(url)
    if response.status_code != 200:
        record_missing_url(url)
        return False
    return True 
  
    
create_feature_tables()



def load_data():
    global validation_annotations
    global unlabeled_annotations
    global test_annotations
    global new_training_data_path
    global new_training_data

    print("loading val")
    validation_annotations = load_annotations(validation_labels_path, validation_images_path, load_all = False)  
    for item in validation_annotations:
        validation_urls[item[1]] = True

    print(len(validation_annotations))


    print("loading existing annotations")
    new_training_data = load_training_data(new_training_data_path)
    print(len(new_training_data))
    
    print("loading eval")
    evaluation_annotations = load_annotations(evaluation_labels_path, evaluation_images_path, load_all = False)  

    print(len(evaluation_annotations))

    print("loading train")
    unlabeled_annotations = load_annotations(training_labels_path, training_images_path, load_all = True)  
    print("all data loaded")

    print(len(unlabeled_annotations))



def continually_retrain():
    while True:        
        train_model()
        eel.sleep(20) # Use eel.sleep(), not time.sleep()



def continually_sample():
    global current_model
    while True:        
        if current_model != None:
            get_random_prediction()
        else:
            # no model yet, wait
            eel.sleep(20) # Use eel.sleep(), not time.sleep()




eel.spawn(load_data)

eel.spawn(add_pending_annotations)

eel.spawn(continually_retrain)

eel.spawn(continually_sample)


eel.start('bicycle_detection.html', size=(1350, 900))




