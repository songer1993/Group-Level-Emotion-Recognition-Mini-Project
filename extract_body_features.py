import openface
import cv2
import dlib
from glob import glob
from utils.imutils import *
from scipy.spatial.distance import euclidean
from phog_features.phog import PHogFeatures
import pandas as pd

IN_PATH = "./bodies/*.jpg"
FEATURES_OUT_PATH = "./features/"

phog = PHogFeatures()
phog_features_2 = dict()
phog_features_3 = dict()
phog_features_4 = dict()
image_ids2 = []
image_ids3 = []
image_ids4 = []

# Loop over images in the dataset to extract features
for image_path in glob(IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    num_faces = int(num_faces)
    face_id = int(face_id)
    #print num_faces, image_id, face_id

    if (num_faces == 2):
        phog_features_2[image_id] = list()
        image_ids2.append(image_id)
    elif (num_faces == 3):
        phog_features_3[image_id] = list()
        image_ids3.append(image_id)
    else:
        phog_features_4[image_id] = list()
        image_ids4.append(image_id)

# Generate a list of unique image ids
image_ids2 = sorted(set(image_ids2))
image_ids3 = sorted(set(image_ids3))
image_ids4 = sorted(set(image_ids4))


# Loop over images in the dataset to extract features
for image_path in glob(IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    #print num_faces, image_id, face_id
    num_faces = int(num_faces)
    face_id = int(face_id)

    phog_feature = phog.get_features(image_path, bins = 10, pyramid_levels= 3).tolist()

    if(num_faces == 2):
        phog_features_2[image_id].append(phog_feature)
    elif(num_faces == 3):
        phog_features_3[image_id].append(phog_feature)
    else:
        phog_features_4[image_id].append(phog_feature)



# Construct and save feature data
for image_id in image_ids2:
    phog_features_2[image_id] = [item for sublist in phog_features_2[image_id] for item in sublist]
for image_id in image_ids3:
    phog_features_3[image_id] = [item for sublist in phog_features_3[image_id] for item in sublist]
for image_id in image_ids4:
    phog_features_4[image_id] = [item for sublist in phog_features_4[image_id] for item in sublist]
df2 = pd.DataFrame(phog_features_2).T
df3 = pd.DataFrame(phog_features_3).T
df4 = pd.DataFrame(phog_features_4).T

print df2.shape
print df3.shape
print df4.shape

df2.to_csv((FEATURES_OUT_PATH+"2_body_phog_features.csv"))
df3.to_csv((FEATURES_OUT_PATH+"3_body_phog_features.csv"))
df4.to_csv((FEATURES_OUT_PATH+"4_body_phog_features.csv"))