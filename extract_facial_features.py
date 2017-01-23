import openface
import cv2
from descriptors.localbinarypatterns import LocalBinaryPatterns
from glob import glob
import pandas as pd

FACE_IN_PATH = "./faces/*.jpg"
LEFT_EYE_IN_PATH = "./left_eyes/*.jpg"
RIGHT_EYE_IN_PATH = "./right_eyes/*.jpg"
MOUTH_IN_PATH = "./mouths/*.jpg"
FEATURES_OUT_PATH = "./features/"

lbp = LocalBinaryPatterns(24, 8)
lbp_features_2 = dict()
lbp_features_3 = dict()
lbp_features_4 = dict()
image_ids2 = []
image_ids3 = []
image_ids4 = []

# Initialise for stroage
for image_path in glob(LEFT_EYE_IN_PATH):
    # Extract image id, face id, and number of faces
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    num_faces = int(num_faces)
    face_id = int(face_id)
    print num_faces, image_id, face_id

    if (num_faces == 2):
        lbp_features_2[image_id] = list()
        image_ids2.append(image_id)
    elif (num_faces == 3):
        lbp_features_3[image_id] = list()
        image_ids3.append(image_id)
    else:
        lbp_features_4[image_id] = list()
        image_ids4.append(image_id)

# Generate a list of unique image ids
image_ids2 = sorted(set(image_ids2))
image_ids3 = sorted(set(image_ids3))
image_ids4 = sorted(set(image_ids4))


# Loop over face images in the dataset to extract features
for image_path in glob(FACE_IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    #print num_faces, image_id, face_id
    num_faces = int(num_faces)
    face_id = int(face_id)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_feature = lbp.describe(gray).tolist()

    if(num_faces == 2):
        lbp_features_2[image_id].append(lbp_feature)
    elif(num_faces == 3):
        lbp_features_3[image_id].append(lbp_feature)
    else:
        lbp_features_4[image_id].append(lbp_feature)

# Loop over left eye images in the dataset to extract features
for image_path in glob(LEFT_EYE_IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    # print num_faces, image_id, face_id
    num_faces = int(num_faces)
    face_id = int(face_id)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_feature = lbp.describe(gray).tolist()

    if (num_faces == 2):
        lbp_features_2[image_id].append(lbp_feature)
    elif (num_faces == 3):
        lbp_features_3[image_id].append(lbp_feature)
    else:
        lbp_features_4[image_id].append(lbp_feature)

# Loop over left eye images in the dataset to extract features
for image_path in glob(RIGHT_EYE_IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    # print num_faces, image_id, face_id
    num_faces = int(num_faces)
    face_id = int(face_id)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_feature = lbp.describe(gray).tolist()

    if (num_faces == 2):
        lbp_features_2[image_id].append(lbp_feature)
    elif (num_faces == 3):
        lbp_features_3[image_id].append(lbp_feature)
    else:
        lbp_features_4[image_id].append(lbp_feature)

# Loop over mouth images in the dataset to extract features
for image_path in glob(MOUTH_IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    # print num_faces, image_id, face_id
    num_faces = int(num_faces)
    face_id = int(face_id)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_feature = lbp.describe(gray).tolist()

    if (num_faces == 2):
        lbp_features_2[image_id].append(lbp_feature)
    elif (num_faces == 3):
        lbp_features_3[image_id].append(lbp_feature)
    else:
        lbp_features_4[image_id].append(lbp_feature)

# Construct and save data
for image_id in image_ids2:
    lbp_features_2[image_id] = [item for sublist in lbp_features_2[image_id] for item in sublist]
for image_id in image_ids3:
    lbp_features_3[image_id] = [item for sublist in lbp_features_3[image_id] for item in sublist]
for image_id in image_ids4:
    lbp_features_4[image_id] = [item for sublist in lbp_features_4[image_id] for item in sublist]
df2 = pd.DataFrame(lbp_features_2).T
df3 = pd.DataFrame(lbp_features_3).T
df4 = pd.DataFrame(lbp_features_4).T

print df2.shape
print df3.shape
print df4.shape

df2.to_csv((FEATURES_OUT_PATH+"2_facial_lbp_features.csv"))
df3.to_csv((FEATURES_OUT_PATH+"3_facial_lbp_features.csv"))
df4.to_csv((FEATURES_OUT_PATH+"4_facial_lbp_features.csv"))