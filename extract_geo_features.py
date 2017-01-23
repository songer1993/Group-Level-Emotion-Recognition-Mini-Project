import openface
import cv2
import dlib
from glob import glob
from utils.imutils import *
from scipy.spatial.distance import euclidean
import pandas as pd

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
INNER_EYES_MOUTH_AND_NOSE = openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
FACE_SIZE = 128
IN_PATH = "./MultiEmoVA-images/images/all/*.jpg"
FEATURES_OUT_PATH = "./features/"

# Create facial landmark detector using dlib model
align = openface.AlignDlib(PREDICTOR_PATH)

landmark_distances2 = []
landmark_distances3 = []
landmark_distances4 = []

# Loop over images in the dataset to extract features
for image_path in glob(IN_PATH):
    # Extract image name
    image_id = image_path.split("/")[-1][:-4]

    # Read in image and detect faces
    image = cv2.imread(image_path)
    faces = align.getAllFaceBoundingBoxes(image)

    # Select at most 4 faces closest to the image centre
    face_distances_from_image_centre = []
    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        face_centre = (float((x1 + x2)/2), float((y1+y2)/2))
        image_centre = (float(image.shape[1]/2), float(image.shape[0]/2))
        face_distances_from_image_centre.append(euclidean(face_centre, image_centre))
    idx_centre_faces = sorted(range(len(face_distances_from_image_centre)), key=lambda x: face_distances_from_image_centre[x])[:4]
    faces = [faces[i] for i in idx_centre_faces]
    # Order these faces from left to right
    dict_x = {i:faces[i].left() for i in range(len(faces))}
    close_to_left = sorted(dict_x.items(), key=lambda x: x[1])
    faces = [faces[close_to_left[i][0]] for i in range(len(close_to_left))]
    # Confirm number of faces
    num_faces = len(faces)
    if num_faces < 2:
        print "%s: just one face" % image_id

    landmark_distances = []
    for idx, face in enumerate(faces):
        # Align face and save it
        aligned_face = align.align(FACE_SIZE, image, bb = face, landmarkIndices= INNER_EYES_MOUTH_AND_NOSE, skipMulti=True)
        # Normalised landmarks in each face for computing
        rect = dlib.rectangle(left=0, right = FACE_SIZE, top = 0, bottom = FACE_SIZE)
        landmarks = align.findLandmarks(aligned_face, rect)
        for i in range(len(landmarks) - 1):
            for j in range(i + 1, len(landmarks)):
                landmark_distances.append(euclidean(landmarks[i], landmarks[j]))

        # Crop and save left eye and annotate it on the aligned face
        x1l, y1l, x2l, y2l = build_left_eye_bounding_box(FACE_SIZE)
        # Crop and save right eye and annotate it on the aligned face
        x1r, y1r, x2r, y2r = build_right_eye_bounding_box(FACE_SIZE)
        # Crop and save mouth and annotate it on the aligned face
        x1m, y1m, x2m, y2m = build_mouth_bounding_box(FACE_SIZE)
        # Crop and save body
        x1b, y1b, x2b, y2b = build_body_bounding_box(face, image)

    if num_faces == 2:
        landmark_distances2.append(landmark_distances)
    elif num_faces == 3:
        landmark_distances3.append(landmark_distances)
    else:
        landmark_distances4.append(landmark_distances)

df_geo2 = pd.DataFrame(landmark_distances2)
df_geo3 = pd.DataFrame(landmark_distances3)
df_geo4 = pd.DataFrame(landmark_distances4)
df_geo2.to_csv((FEATURES_OUT_PATH+"2_geo_features.csv"))
df_geo3.to_csv((FEATURES_OUT_PATH+"3_geo_features.csv"))
df_geo4.to_csv((FEATURES_OUT_PATH+"4_geo_features.csv"))
