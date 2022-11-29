from django.utils.html import conditional_escape
from django.http import JsonResponse
from django.http import HttpResponse, HttpResponseNotFound
from django.db.models import Case, When
import PIL
from PIL import Image
import boto3
from botocore.exceptions import ClientError
import urllib.request, urllib.error, urllib.parse
import io
import random
import sys
import os
import pickle, json, ast
import cv2 as cv
import numpy as np
import base64
from scipy.spatial import distance
from cropper.models import Images
import operator
import itertools 
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from django.conf import settings

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

import gif2numpy

s3client = boto3.client('s3')
s3= boto3.resource('s3')

# Initialize inceptionv3
model = InceptionV3(weights='imagenet')
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

# Initialize ORB and HOG grabbers
orb = cv.ORB_create(nfeatures=1000)
hog = cv.HOGDescriptor()

# # Instance segmentation
# ins = instanceSegmentation()

# print(os.path.join(settings.BASE_DIR, "static/models/mask_rcnn_coco.h5"))
# ins.load_model(os.path.join(settings.BASE_DIR, "static/models/mask_rcnn_coco.h5"))

# target_classes = ins.select_target_classes(person=True)

# Helpers
def read_kps(kp_dict_list):
    results = []

    for kp in kp_dict_list:
        results.append(cv.KeyPoint(x=kp['coords'][0],y=kp['coords'][1],size=kp['size'], angle=kp['angle'], response=kp['response'], octave=kp['octave'], class_id=kp['class_id']))

    return results

def get_img_url(file_path):
    if file_path is None:
        return ""
    url = s3client.generate_presigned_url(
                'get_object', {
                    'Bucket': 'cwps-media',
                    'Key': file_path
                }
            ),
    return url

# Return a list of narrowed down image ids/pks based on feature matching before running hog and embedding distance
def feature_match_flann(q_kp, q_des):
    # Minimum amount of matches to be considered a match
    MIN_MATCH_COUNT = 10
    q_kp = q_kp
    q_des = np.float32(q_des)

    # Return n set of ids where n is the number of orb kp/des sets
    results = []

    for i in range(0, len(q_des)):
        this_des = q_des[i]
        this_kp = q_kp[i]


# Main API call
def get_results(request):
    if request.method == 'POST':
        searchType = request.POST.get('searchType')

        inputImg = base64.b64decode(request.POST.get('img'))
        
        inputBbox = []
        
        counter = 0

        while len(request.POST.getlist(f'areas[{counter}][]')) > 0:
            inputBbox.append(request.POST.getlist(f'areas[{counter}][]'))
            counter += 1

        # Read image into openCV
        nparr = np.fromstring(inputImg, np.uint8)

        img = cv.imdecode(nparr, cv.IMREAD_COLOR)

        # # Add person segmentation here
        # result, output = ins.segmentImage(img, segment_target_classes=target_classes, output_image_name='../pic.jpg', extract_segmented_objects= True, save_extracted_objects=False)

        # background_only_img = img * ~result['masks']


        # Resize
        hog_img = cv.resize(img, (64, 128))

        # Get hog descriptors
        hog_desc = hog.compute(hog_img)
        hog_desc = np.nan_to_num(hog_desc, copy=False)


        # Resize
        embed_img = cv.resize(img, (299, 299))
        embed_img = np.array(embed_img) 
        embed_img = embed_img.reshape(1,299,299,3) 

        # Get embeddings
        imgx = preprocess_input(embed_img)
        img_features =  model.predict(imgx).tolist()[0]

        rois = []

        # Crop image
        for bbox in inputBbox:
            x1 = int(bbox[0])
            x2 = int(bbox[1])
            y1 = int(bbox[2])
            y2 = int(bbox[3])
            rois.append(img[y1:y2, x1:x2])
            
        # Get orb features from rois
        orb_kps = []
        orb_desc = []
        for roi in rois:
            roi_adj = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(roi_adj, None)
            orb_kps.append(kp)
            orb_desc.append(des)

        # Run feature detection on image and return results
        # Return separate list of kps for each roi feature match




        # Add and/or logic later?
        # If and, return intersection, if or return union
        if searchType == 1:
            print('AND Search')
        elif searchType == 2:
            print('OR Search')


        # Compute distances and return top 20 directories with highest similarity based on hog?
        photo = Images.objects.values('photo_id','image_embeddings_full_image')
        photo = list(photo)
        distances = {}

        for hogs in photo:
            hog_val = np.array(hogs['image_embeddings_full_image'].strip().rstrip(',').split(','), dtype=np.float32)
            
            hog_val = np.nan_to_num(hog_val, copy=False)

            distances[hogs['photo_id']] = round(float(distance.euclidean(img_features,hog_val)),5)


        distances=dict(sorted(distances.items(), key=operator.itemgetter(1)))

        # 20 lowest distances (hog)
        distances = dict(itertools.islice(distances.items(),20))

        top_photos = Images.objects.filter(photo_id__in=distances.keys())

        results = {}

        for index, photo_id in enumerate(distances.keys()): 
            photo_data = top_photos.filter(photo_id=photo_id).values()[0]

            results[index] = {}

            results[index]['photo_id'] = photo_id

            results[index]['url'] = get_img_url(photo_data['file_path'])
            # results[result['id']]['orb_score'] = result['orb_score']
            results[index]['hog_distance'] = distances[photo_id]
            # results[result['id']]['embedding_score'] = result['embedding_score']

        data = {'success': 'success', 'results': results}

        return JsonResponse(data)





