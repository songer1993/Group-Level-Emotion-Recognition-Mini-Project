import cv2
from glob import glob
from descriptors.centrist import CENTRIST
import numpy as np
import pandas as pd

IMAGE_IN_PATH = "./MultiEmoVA-images/images/all/*.jpg"
FACE_IN_PATH = "./faces/*.jpg"
FEATURES_OUT_PATH = "./features/"

centrist = CENTRIST(divide = 4)

num_faces_in_images = {}

# Loop over face images to obtatin
# image-number_of_faces dictionary
for image_path in glob(FACE_IN_PATH):
    # Extract image id, face id, and number of faces
    image_id = image_path.split("/")[-1][:-4]
    (num_faces, image_id, face_id) = image_id.split("_")
    num_faces_in_images[image_id] = int(num_faces)

centrist2 = []
centrist3 = []
centrist4 = []
for image_path in glob(IMAGE_IN_PATH):
    image_id = image_path.split("/")[-1][:-4]
    num_faces = num_faces_in_images[image_id]
    print(image_id)
    # Load in image and resize to (400, 400)
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (400, 400))
    # Convert to gray and extract CENTRIST
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    feature = centrist.describe(gray)
    # Store feature into different face number groups
    if(num_faces == 2):
        centrist2.append(feature)
    elif(num_faces==3):
        centrist3.append(feature)
    else:
        centrist4.append(feature)

# Construct data frames and store in CSV files
df2 = pd.DataFrame(centrist2)
df3 = pd.DataFrame(centrist3)
df4 = pd.DataFrame(centrist4)
df2.to_csv((FACE_IN_PATH+"2_CENTRIST.csv"))
df3.to_csv((FACE_IN_PATH+"3_CENTRIST.csv"))
df4.to_csv((FACE_IN_PATH+"4_CENTRIST.csv"))