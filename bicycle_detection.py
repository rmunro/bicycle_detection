import eel, os, random, sys, re
import time
import gzip
import csv
import hashlib
import random
from random import shuffle
from typing import List, Optional, Dict, NamedTuple, Tuple, Any, Set, Union
from pathlib import Path


##

# import torchvision.models as models
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import urllib
from urllib.error import HTTPError 
import requests
import sqlite3 
import pickle
import zlib
import math
import statistics
import datetime

##
NOT_BICYCLE_CLASS = "0"
BICYCLE_CLASS = "1"

verbose = True

FEATURE_STORE_PATH = "data/feature_store.db"

# a subset of Open Images training data to be within 100MB github limit
TRAINING_LABELS_PATH = 'data/oidv6-train-annotations-human-imagelabels-reduced.csv.gz'
TRAINING_IMAGES_PATH = 'data/oidv6-train-images-with-labels-with-rotation-reduced.csv.gz'

VALIDATION_LABELS_PATH = 'data/validation-annotations-human-imagelabels.csv.gz'
VALIDATION_IMAGES_PATH = 'data/validation-images-with-rotation.csv.gz'

EVALUATION_LABELS_PATH = 'data/test-annotations-human-imagelabels.csv.gz'
EVALUATION_IMAGES_PATH = 'data/test-images-with-rotation.csv.gz'

NEW_TRAINING_DATA_PATH = 'data/new-training-data.csv'

TRAININD_DATA_IMAGE_ID = 0
TRAININD_DATA_ANNOTATION_URL = 1
TRAININD_DATA_ANNOTATION_LABEL = 2

class TrainingData(NamedTuple):
    image_id: str
    url: str
    label: str


class AnnotationDataWithScore(NamedTuple):
    image_id: str
    url: str
    label: str
    thumbnail_url: str
    score: float

    
class AnnotationData(NamedTuple):
    image_id: str
    url: str
    label: str
    thumbnail_url: str

    def with_score(self, score: float)->AnnotationDataWithScore:
        return AnnotationDataWithScore(
                self.image_id, 
                self.url, 
                self.label, 
                self.thumbnail_url,
                score)
    
    

unlabeled_items: List[AnnotationData] = []
validation_annotations: List[AnnotationData] = []
evaluation_annotations: List[AnnotationData] = []


class PendingAnnotationData(NamedTuple):
    url: str
    is_bicycle: bool


pending_annotations: List[PendingAnnotationData] = []  # annotations pending being stored

# image_id ->url
new_training_data: Dict[str, str] = {}  # new training data by url

validation_urls: Set[str] = set()  # validation item urls

min_training_items = 5  # min items for each class to start training

# For entries in high_uncertainty_items and model_based_outliers the confidence field should be set
high_uncertainty_items: List[AnnotationDataWithScore] = []  # items queued for annotation because of uncertainty
model_based_outliers: List[AnnotationDataWithScore] = []  # items queued for annotation because they are outliers and uncertain

number_sampled_to_cache = 10  # how many active learning samples in memory to support rapid annotation
number_to_sample_per_train = 50  # how many items to predict over for each new model
# TODO: make these bigger before release


total_time = 0.0 # total time to download new images and extract features
total_downloads = 0 # total number of images downloaded 

ACCURACIES_FSCORE_INDEX = 0

current_accuracies: List[float] = [-1, -1, -1, -1, -1]
current_model: Optional[nn.Module] = None

feature_store = sqlite3.connect(FEATURE_STORE_PATH)

eel.init('./')

# Download models. ~255MB, so will take some time to download first time
resnext50_model = models.resnext50_32x4d(pretrained=True)
modules=list(resnext50_model.children())[:-1] # strip last layer of resnext:
resnext50_sll_model=nn.Sequential(*modules)

fasterrcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

BICYCLE_LABEL_COCO = 2  # label within coco dataset
BICYCLE_LABEL_OI = "/m/0199g"  # label within open images dataset

image_id_urls: Dict[str, str] = {}  # image_ids indexed by url


class SimpleClassifier(nn.Module):  # inherit pytorch's nn.Module
    """ Linear Classifier with no hidden layers

    """
    
    def __init__(self, num_labels:int, num_inputs:int)->None:
        super(SimpleClassifier, self).__init__() # call parent init
        self.linear = nn.Linear(num_inputs, num_labels)
        

    def forward(self, feature_vec:torch.Tensor, return_all_layers:bool=False)->Union[List[float], float]:
        # Define how data is passed through the model and what gets returned

        output = self.linear(feature_vec)
        log_softmax = F.log_softmax(output, dim=1)

        if return_all_layers:
            return [output, log_softmax]
        else:
            return log_softmax
    
    

# from: https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
# GET FASTER R CNN COCO DATASET PREDICTION FOR A BICYCLE
def get_fasterrcnn_prediction(img: Image.Image) -> List[float]:
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
    bbox: List[float] = [0, 0, width, height]
    for ind in range(0, len(pred_boxes)):
        if ind == BICYCLE_LABEL_COCO:
            if pred_score[ind] > max_bike:
                max_bike = pred_score[ind]
                bbox = pred_boxes[ind]  # left, top, right, bottom
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
def get_resnext_features(img: Image.Image) -> List[float]:
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
  

def make_feature_vector(image_id:str, url: str, label: str = "") -> Optional[torch.Tensor]:
    global total_time
    global total_downloads
    
    if url_is_missing(url) or is_bad_image(url):
        return None

    # CHECK IF WE'VE STORED IT
    feature_list = get_features_from_store(image_id)
    
    # EXTRACT FEATURES FROM COCO & IMAGENET MODELS
    if feature_list is None:
        start_time = time.time()
        
        try:
            img = Image.open(urllib.request.urlopen(url))
        except HTTPError:        
            record_missing_url(url)
            return None

        try:
            # ensure that image is RGB image, as needed for resnet
            if img.mode != 'RGB':
                print(f"Converted {url} to rgb image")
                img = img.convert('RGB')
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
                print(f"average number of seconds to process new image: {total_time / total_downloads: 0.2f}")

        except RuntimeError as e:
            print(f"Problem with {url} {e}")
            record_bad_image(url)
            return None
            
            

    vector = torch.Tensor(feature_list)    
    return vector.view(1, -1)
    


def load_training_data(filepath: str) -> Dict[str, str]:
    # FOR ALREADY LABELED ONLY
    # csv format: [IMAGE_ID, URL, LABEL,...]
    if not os.path.exists(filepath):
        print(f"No existing training data found")
        return {}
    
    new_data = {}
    
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for item in reader:
            image_id = item[TRAININD_DATA_IMAGE_ID]
            url = item[TRAININD_DATA_ANNOTATION_URL]
            label = item[TRAININD_DATA_ANNOTATION_LABEL]
            new_data[url] = label
            image_id_urls[url] = image_id
    print(f"Loaded {len(new_data)} training data points")
    return new_data


def load_annotations(annotation_labels_filepath: str, annotation_image_filepath: str, load_all: bool = False) -> List[
    AnnotationData]:
    '''Load Open Images Annotations
      assume these are static, so we can pickle them to be loaded quicker     
    '''
    
    cached_data = get_data_structure_store(annotation_image_filepath)
    if cached_data is not None:
        for annotated_data_point in cached_data:
            image_id_urls[annotated_data_point.url] = annotated_data_point.image_id
        if verbose:
            print(f"loaded cached data {annotation_image_filepath} : {len(cached_data)} data points")
        return cached_data

    annotations: Dict[str, str] = {}
    c = 0
    ten_thousands = 0

    file = gzip.open(annotation_labels_filepath, mode='rt', encoding="utf8")
    csvobj = csv.reader(file, delimiter=',', quotechar='"')
    for row in csvobj:
        if row[2] == BICYCLE_LABEL_OI:
            image_id = row[0]
            label = row[3]
            annotations[image_id] = label
        c += 1
        if c == 10000:
            eel.sleep(0.01)
            c = 0
            ten_thousands += 1
            if ten_thousands % 100 == 0:
                print(f"processed {ten_thousands * 10000} points from {annotation_labels_filepath}")

    print(f"Loaded {len(annotations)} annotated points from {annotation_labels_filepath}")

    annotated_data = []
    file = gzip.open(annotation_image_filepath, mode='rt', encoding="utf8")
    csvobj = csv.reader(file, delimiter=',', quotechar='"')

    c = 0
    ten_thousands = 0
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
                label = NOT_BICYCLE_CLASS

            annotated_data.append(AnnotationData(image_id, url, label, thumbnail_url))

        c += 1
        if c == 10000:
            eel.sleep(0.01)
            c = 0
            ten_thousands += 1
            if ten_thousands % 10 == 0:
                print(f"processed {ten_thousands * 10000} points from {annotation_image_filepath}")

    print(f"Loaded {len(annotated_data)} data points from {annotation_image_filepath}. load_all={load_all}")

    for annotated_data_point in annotated_data:
        image_id_urls[annotated_data_point.url] = annotated_data_point.image_id
    store_data_structure(annotation_image_filepath, annotated_data)
    return annotated_data
    
  


def train_model(batch_size: int = 20, num_epochs: int = 40, num_labels: int = 2, num_inputs: int = 2058, model: Optional[nn.Module]=None) -> \
        Optional[nn.Module]:
    """Train model on the given training_data

    Tune with the validation_data
    Evaluate accuracy with the evaluation_data
    """
    global current_model
    global current_accuracies

    if len(new_training_data) == 0:
        return None
        
    urls = list(new_training_data.keys())
    if verbose:
        print(f"Has {len(urls)} points for training")

    if model is None:
        model = SimpleClassifier(num_labels, num_inputs)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epochs training
    for epoch in range(num_epochs):
        current = 0

        # make a subset of data to use in this epoch
        # with an equal number of items from each label
        
        bicycle: List[TrainingData] = []
        not_bicycle: List[TrainingData] = []

        shuffle(urls) #randomize the order of the training data 
        for url in urls:
            label = new_training_data[url]
            if len(bicycle) >= batch_size and len(not_bicycle) >= batch_size:
                break
            elif new_training_data[url] == BICYCLE_CLASS and len(bicycle) < batch_size:
                bicycle.append(TrainingData(image_id_urls[url], url, label))
            elif new_training_data[url] == NOT_BICYCLE_CLASS and len(not_bicycle) < batch_size:
                not_bicycle.append(TrainingData(image_id_urls[url], url, label))

        if len(bicycle) < min_training_items or len(not_bicycle) < min_training_items:
            if verbose:
                print(f"Not yet enough labels to train: {len(bicycle)} of {len(urls)}")
            return None
        
        epoch_data = bicycle + not_bicycle
        shuffle(epoch_data) 
 
        if verbose or True:
            print("Epoch: "+str(epoch))

                
        # train our model
        for item in epoch_data:
            try:
                image_id = item.image_id
                url = item.url
                int_label = int(item.label)

                feature_vec = make_feature_vector(image_id, url)
                if feature_vec is None:
                    print(f"no features for {url}")
                    continue
            
                target = torch.LongTensor([int_label])

                model.zero_grad() 

                log_probs = model(feature_vec)
                eel.sleep(0.01) # let other processes in
            
                # compute loss function, do backward pass, and update the gradient
                loss = loss_function(log_probs, target)
            
                

                loss.backward()
                optimizer.step() 
            except RuntimeError as e:
                print("Warning, error while training:")
                print(e)


        fscore, auc, precision, recall, ave_loss = evaluate_model(model, False, -1)
        fscore = round(fscore,3)
        auc = round(auc,3)
        if verbose:
            print(f"F-Score {fscore} AUC {auc} Precision {precision} Recall {recall}")

        if fscore > 0 and fscore > current_accuracies[ACCURACIES_FSCORE_INDEX]:

            # evaluate on all *evaluation* data and save model 
            test_fscore, test_auc, test_precision, test_recall, test_ave_loss = evaluate_model(model, True, -1)

            if verbose:
                print(f"Test: F-Score {test_fscore} AUC = {test_auc} Precision {test_precision} Recall {test_recall}")


            test_auc = round(test_auc,3)

            # save model to path that is alphanumeric and includes number of items and accuracies in filename
            timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
            training_size = "_"+str(len(urls))
            accuracies = str(test_fscore)+"_"+str(test_auc)
                     
            model_path = "models/"+timestamp+accuracies+training_size+".params"
            torch.save(model.state_dict(), model_path)
                
            current_accuracies = [fscore, auc, precision, recall, ave_loss]
            current_model = model

    if current_model is None:
        if verbose:
            print("Not getting predictions: we don't have a good model yet") 
    else:
        if verbose:
            print("Getting predictions across unlabeled items so we can sample with active learning")

        for i in range(number_to_sample_per_train):
            get_random_prediction()
        

    return model
  
 
 


def evaluate_model(model:nn.Module, use_evaluation: bool = True, limit:int=-1) -> List[float]:
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
            print("data not loaded")
        return[0,0,0,0,0] # not loaded yet

    
    count = 0
    for item in evaluation_data:
        if limit > 0 and count > limit:
            break

        image_id = item.image_id
        url = item.url
        label = item.label

        feature_vector = make_feature_vector(image_id, url)
        if feature_vector is None:
            continue
                
        with torch.no_grad():
            log_probs = model(feature_vector)
        eel.sleep(0.01)


        # get probability that item is bicycle
        prob_bicycle = math.exp(log_probs.data.tolist()[0][1]) 

        # record loss if we have a label
        if label is not None:
            target = torch.LongTensor([int(label)])
            loss = loss_function(log_probs, target)
            total_loss += loss

        if label == BICYCLE_CLASS:
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
    data_type = 'Evaluation Data' if use_evaluation else 'Validation Data'
    print(f"{data_type} - TruePos:{true_pos} False Pos: {false_pos} False Neg {false_neg} TrueNeg {true_neg}")

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
    print(f"{data_type} ave confs: {conf_b} {conf_n}")
    print(f"{data_type} ave loss: {ave_loss}")

    return[fscore, auc, precision, recall, ave_loss]


def load_most_recent_model(num_labels: int = 2, num_inputs: int = 2058) -> None:
    global current_model
    global current_accuracies
    

    existing_models = sorted(Path('models/').iterdir(), key=os.path.getmtime)
    
    if len(existing_models) == 0:
        return

    last_model = existing_models[-1]
    # In order not to load Readme.md
    if last_model.name.endswith(".params"):
        
        current_model = SimpleClassifier(num_labels, num_inputs)
        current_model.load_state_dict(torch.load(str(last_model)))
        
        current_accuracies = evaluate_model(current_model, False, -1)
                
        print(f"loaded model: {last_model}")
    
    
        



def get_quantized_logits(logits: List[float])->float:
    ''' Returns the quanitized (0-1) logits
    
    '''
    # TODO: QUANTIZE WHEN EVALUATING VALIDATION DATA
    return 1- (logits[0] + logits[1])
    


def get_random_prediction(model:Optional[nn.Module]=None) -> None:
    '''Get predictions on unlabeled data 
    
    '''
    
    if model is None:
        model = current_model 
    if model is None:
        # should not happen -> safeguard
        return 
    
    item: AnnotationData = random.choice(unlabeled_items)
    with torch.no_grad():
        image_id = item.image_id
        url = item.url

        feature_vector = make_feature_vector(image_id, url)
        if feature_vector is None:
            return
            
        logits, log_probs = model(feature_vector, return_all_layers = True)     
    
        prob_bicycle = math.exp(log_probs.data.tolist()[0][1]) 
            
        least_conf = 2 * (1 - max(prob_bicycle, 1-prob_bicycle))
        # TODO: work out why this is typically -20

        outlier_score = get_quantized_logits(logits.data.tolist()[0])
            
        if len(high_uncertainty_items) < number_sampled_to_cache:
            new_high_uncertainty_item = item.with_score(least_conf)
            if verbose or True:
                print(f"adding an initial item to uncertainty samples {new_high_uncertainty_item}")
                print(len(high_uncertainty_items))
            high_uncertainty_items.append(new_high_uncertainty_item)
        elif least_conf > high_uncertainty_items[-1].score:
            new_high_uncertainty_item = item.with_score(least_conf)
            if verbose or True:
                print(f"adding to uncertainty samples {new_high_uncertainty_item}")
            high_uncertainty_items.append(new_high_uncertainty_item)
            high_uncertainty_items.pop(-1)
            high_uncertainty_items.sort(reverse=True, key=lambda x: x.score)  # TODO: RIGHT

        if least_conf > 0.5:
            if len(model_based_outliers) < number_sampled_to_cache:
                outlier_item = item.with_score(outlier_score)
                if verbose or True:
                    print(f"adding an initial item {outlier_item} to outlier samples")
                # Added in order to prevent exception in the next line
                model_based_outliers.append(outlier_item)
            elif least_conf > model_based_outliers[-1].score:
                outlier_item = item.with_score(outlier_score)
                if verbose or True:
                    print(f"adding to outlier samples {outlier_item}")
                model_based_outliers.append(outlier_item)
                model_based_outliers.pop(-1)
                model_based_outliers.sort(reverse=True, key=lambda x: x.score)
        eel.sleep(0.1)


def create_feature_tables() -> None:
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
        


def record_missing_url(url:str) -> None:
    sql = 'INSERT OR REPLACE INTO url_missing (url) values(?)'
    feature_store.executemany(sql, [(url,)])


def url_is_missing(url:str) -> bool:
    with feature_store:
        data = feature_store.execute("SELECT * FROM url_missing WHERE url = '" + url + "'")
        for _row in data:
            return True  # it exists
        return False


def record_bad_image(url:str) -> None:
    sql = 'INSERT OR REPLACE INTO bad_image (url) values(?)'
    feature_store.executemany(sql, [(url,)])


def is_bad_image(url:str) -> bool:
    with feature_store:
        data = feature_store.execute("SELECT * FROM bad_image WHERE url = '" + url + "'")
        for row in data:
            return True  # it exists
        return False


def store_data_structure(structure_name: str, data: Any) -> None:
    sql = 'INSERT OR REPLACE INTO data_structure (name, data) values(?, ?)'
    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    compressed_data = zlib.compress(pickled_data)
    feature_store.executemany(sql, [(structure_name, compressed_data)])


def get_data_structure_store(structure_name: str) -> Optional[List[AnnotationData]]:
    with feature_store:
        cursor = feature_store.execute("SELECT name, data FROM data_structure WHERE name = '"+structure_name+"'")
        for row in cursor: 
            try:             
                compressed_data = row[1]                
                pickled_data = zlib.decompress(compressed_data)
                data = pickle.loads(pickled_data)
            except Exception as e:
                print("Couldn't load "+str(structure_name)+": "+str(e))

                return None
            return data
        return None


def add_to_feature_store(image_id: str, features: List[float], url: str = "", label: str = "") -> None:
    sql = 'INSERT OR REPLACE INTO feature (image_id, url, features, label) values(?, ?, ?, ?)'
    pickled_features = pickle.dumps(features, pickle.HIGHEST_PROTOCOL)
    compressed_features = zlib.compress(pickled_features)
    feature_store.executemany(sql, [(image_id, url, compressed_features, str(label))])


def get_features_in_feature_store() -> int:
    with feature_store:
        data = feature_store.execute(
            "SELECT count(*) FROM feature")
        for row in data:
            return row[0]
        #should be unreachable 
        return 0    

def get_features_from_store(image_id:str)->Optional[List[float]]:
    with feature_store:
        data = feature_store.execute("SELECT image_id, url, features, label FROM feature WHERE image_id = '"+image_id+"'")
        for row in data: 
            try:             
                compressed_features = row[2]                
                pickled_features = zlib.decompress(compressed_features)
                features = pickle.loads(pickled_features)
            except Exception as e:
                print("Couldn't load " + image_id + " : " + str(e))
                return None
            return features

        return None


def add_pending_annotations()->None:
    while True:
        # copy to avoid race conditions
        if len(pending_annotations) > 0 and verbose:
            print("adding pending annotations")
        
        found_annotation = None
        for annotation in pending_annotations:
            is_bicycle = annotation.is_bicycle
            if is_bicycle:
                if verbose:
                    print("prioritizing positive annotation")

                label = BICYCLE_CLASS
                url = annotation.url
                image_id = image_id_urls[url]
                
                # cache features for faster training later
                eel.sleep(0.01)  # allow other processes in
                feature_vector = make_feature_vector(image_id, url, label)
                eel.sleep(0.1)  # allow other processes in

                if feature_vector is not None:
                    append_data(NEW_TRAINING_DATA_PATH, [[image_id, url, label]])
                    new_training_data[url] = label
                    found_annotation = annotation

        if found_annotation:
            prior_num = len(pending_annotations)
            pending_annotations.remove(found_annotation) 
            after_num = len(pending_annotations)
            if after_num + 1 != prior_num:
                print("Warning did not remove item from list")
                   
        elif len(pending_annotations) > 0:
            label = NOT_BICYCLE_CLASS
            annotation = pending_annotations.pop()
            url = annotation.url
            image_id = image_id_urls[url]
                
            # cache features for faster training later
            eel.sleep(0.01)  # allow other processes in
            feature_vector = make_feature_vector(image_id, url, label)
            eel.sleep(0.1)  # allow other processes in
            if feature_vector is not None:
                append_data(NEW_TRAINING_DATA_PATH, [[image_id, url, label]])
                new_training_data[url] = label
        else:
            eel.sleep(1)
        


def append_data(filepath: str, data: List[List[str]]) -> None:
    with open(filepath, 'a', errors='replace', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()


@eel.expose
def training_loaded() -> bool:
    return len(unlabeled_items) > 0


@eel.expose
def validation_loaded() -> bool:
    return len(validation_annotations) > 0


@eel.expose
def get_current_accuracies() -> List[float]:
    return current_accuracies


@eel.expose
def estimate_processing_time()->float:
    total_pending = len(pending_annotations)
    
    if total_downloads == 0:
        return 0 # no info yet
    else:
        if verbose:
            print(f"Total Time: {total_time}, Total Downloads {total_downloads}, Total Pending {total_pending}")
        return (total_time / total_downloads) * total_pending

@eel.expose
def add_annotation(url:str, is_bicycle:bool)->None:
    if url not in validation_urls:
        if verbose:
            print(f"adding annotation for {url} {is_bicycle}")

        pending_annotations.append(PendingAnnotationData(url, is_bicycle))
        eel.sleep(0.01)
    else:
        if verbose:
            print("skipping validation: "+url)


@eel.expose
def get_next_image()->List[str]:
    if len(validation_annotations) == 0:
        return [] # not yet loaded
    if len(unlabeled_items) == 0:
        return get_validation_image()
    
    
    strategy = random.randint(0,9)
    if strategy == 0:
        return get_validation_image()
    elif strategy == 1 or len(high_uncertainty_items) == 0:
        return get_random_image()
    elif strategy < 9 or len(model_based_outliers) == 0:
        return get_uncertain_image()
    else:
        return get_outlier_image()
    

# get image with high uncertainty    
def get_uncertain_image()->List[str]:
    last_image = high_uncertainty_items.pop()
    return [last_image.url, last_image.thumbnail_url, last_image.label]

    
# get image that is model-based outlier and also uncertain
def get_outlier_image()->List[str]:
    last_image = model_based_outliers.pop()
    return [last_image.url, last_image.thumbnail_url, last_image.label]


def get_validation_image()->List[str]:
    shuffle(validation_annotations)
    label = random.randint(0,1)
    for item in validation_annotations:
        if str(item.label) != str(label):
            continue

        url = item.url
        if url_is_missing(url) or is_bad_image(url) or not test_if_url_ok(url):
            continue

        thumbnail_url = item.thumbnail_url
        if not test_if_url_ok(thumbnail_url):
            thumbnail_url = url
        
        return [url, thumbnail_url, str(label)]

    return []  # if there are no items


def get_random_image() -> List[str]: #type: ignore

    url = ""
    while url == "":
        item = random.choice(unlabeled_items)
        image_id = item.image_id
        url = item.url
        label = ""  # we're getting new labels so ignore OI ones
        thumbnail_url = item.thumbnail_url

        if url in new_training_data or url_is_missing(url) or is_bad_image(url):
            url = ""
            continue
        try:
            if not test_if_url_ok(url):
                url = ""
                continue
            if not test_if_url_ok(thumbnail_url):
                thumbnail_url = url

            return [url, thumbnail_url, label]

        except:
            print(f"error with image_id {image_id} url {url} thumb {thumbnail_url}")
            url = ""


def test_if_url_ok(url: str) -> bool:
    if len(url) == 0:
        return False
    response = requests.head(url)
    if response.status_code != 200:
        record_missing_url(url)
        return False
    return True 
  
    
create_feature_tables()




def load_data() -> None:
    global validation_annotations
    global evaluation_annotations
    global unlabeled_items
    global new_training_data

    print(f"There are {get_features_in_feature_store()} cached image features in feature_store")

    print("loading val")
    validation_annotations = load_annotations(VALIDATION_LABELS_PATH, VALIDATION_IMAGES_PATH, load_all=False)
    for item in validation_annotations:
        validation_urls.add(item.url)

    print("loading existing annotations")
    new_training_data = load_training_data(NEW_TRAINING_DATA_PATH)
    
    print("loading eval")
    evaluation_annotations = load_annotations(EVALUATION_LABELS_PATH, EVALUATION_IMAGES_PATH, load_all=False)

    print("loading train")
    unlabeled_items = load_annotations(TRAINING_LABELS_PATH, TRAINING_IMAGES_PATH, load_all=True)
    print("all data loaded")

    load_most_recent_model()


def continually_retrain() -> None:
    while True:
        train_model()
        eel.sleep(20) # Use eel.sleep(), not time.sleep()





# It takes a while to load the data, especially first time, so we parallelize it
eel.spawn(load_data)

# Separate thread to gradually download and extract COCO and ImageNet representations
eel.spawn(add_pending_annotations)

# Continually retrain the model and get predictions over unlabeled items
eel.spawn(continually_retrain)


eel.start('bicycle_detection.html', size=(1350, 900))




