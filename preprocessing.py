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
IMAGE_IN_PATH = "./MultiEmoVA-images/images/all/*.jpg"
FACE_OUT_PATH = "./faces/"
BODY_OUT_PATH = "./bodies/"
LEFT_EYE_OUT_PATH = "./left_eyes/"
RIGHT_EYE_OUT_PATH = "./right_eyes/"
MOUTH_OUT_PATH = "./mouths/"
ANNOTATED_FACE_OUT_PATH = "./annotated_faces/"
ANNOTATED_IMAGE_OUT_PATH = "./annotated_images/"
FEATURES_OUT_PATH = "./features/"

# Create facial landmark detector using dlib model
align = openface.AlignDlib(PREDICTOR_PATH)

# Initialise number of faces list
num_faces_in_images = []

# Loop over images in the dataset to extract features
for image_path in glob(IMAGE_IN_PATH):
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
        face_centre = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        image_centre = (float(image.shape[1] / 2), float(image.shape[0] / 2))
        face_distances_from_image_centre.append(euclidean(face_centre, image_centre))
    idx_centre_faces = sorted(range(len(face_distances_from_image_centre)),
                              key=lambda x: face_distances_from_image_centre[x])[:4]
    faces = [faces[i] for i in idx_centre_faces]
    # Order these faces from left to right
    dict_x = {i: faces[i].left() for i in range(len(faces))}
    close_to_left = sorted(dict_x.items(), key=lambda x: x[1])
    faces = [faces[close_to_left[i][0]] for i in range(len(close_to_left))]
    # Confirm number of faces
    num_faces = len(faces)
    if num_faces < 2:
        print "%s: just one face" % image_id

    num_faces_in_images.append(num_faces)

    all_landmarks_in_image = []
    bodies = []

    for idx, face in enumerate(faces):
        # Align face and save it
        aligned_face = align.align(FACE_SIZE, image, bb = face, landmarkIndices= INNER_EYES_MOUTH_AND_NOSE, skipMulti=True)
        cv2.imwrite((FACE_OUT_PATH+str(num_faces)+"_"+image_id+"_"+str(idx)+".jpg"), aligned_face)

        # Crop and save left eye and annotate it on the aligned face
        x1l, y1l, x2l, y2l = build_left_eye_bounding_box(FACE_SIZE)
        left_eye = aligned_face[y1l:y2l, x1l:x2l]
        cv2.imwrite((LEFT_EYE_OUT_PATH+str(num_faces)+"_"+image_id+"_"+str(idx)+".jpg"), left_eye)
        cv2.rectangle(aligned_face, (x1l, y1l), (x2l, y2l), (255, 0, 255), 2)

        # Crop and save right eye and annotate it on the aligned face
        x1r, y1r, x2r, y2r = build_right_eye_bounding_box(FACE_SIZE)
        right_eye = aligned_face[y1r:y2r, x1r:x2r]
        cv2.imwrite((RIGHT_EYE_OUT_PATH+str(num_faces)+"_"+image_id+"_"+str(idx)+".jpg"), right_eye)
        cv2.rectangle(aligned_face, (x1r, y1r), (x2r, y2r), (255, 0, 255), 2)

        # Crop and save mouth and annotate it on the aligned face
        x1m, y1m, x2m, y2m = build_mouth_bounding_box(FACE_SIZE)
        mouth = aligned_face[y1m:y2m, x1m:x2m]
        cv2.rectangle(aligned_face, (x1m, y1m), (x2m, y2m), (0, 0, 0), 2)
        cv2.imwrite((MOUTH_OUT_PATH+str(num_faces)+"_"+image_id+"_"+str(idx)+".jpg"), mouth)

        # Crop and save body
        x1b, y1b, x2b, y2b = build_body_bounding_box(face, image)
        body = image[y1b:y2b, x1b:x2b]
        bodies.append((x1b, y1b, x2b, y2b))
        cv2.imwrite((BODY_OUT_PATH+str(num_faces)+"_"+image_id+"_"+str(idx)+".jpg"), body)

        # Save aligned face with eyes and mouth annotated
        #cv2.imshow(str(idx), aligned_face)
        cv2.imwrite((ANNOTATED_FACE_OUT_PATH+str(num_faces)+"_"+image_id+"_"+str(idx)+".jpg"), aligned_face)

        # Original landmarks in the whole image for annotation
        landmarks_in_images = align.findLandmarks(aligned_face, face)
        all_landmarks_in_image.append(landmarks_in_images)

    # Annotating the full image with landmarks and faces
    annotated = annotate_all_faces_and_landmarks(image, faces, bodies, all_landmarks_in_image)
    cv2.imwrite((ANNOTATED_IMAGE_OUT_PATH+str(num_faces)+"_"+image_id + ".jpg"), annotated)
    #cv2.imshow("annotated image", annotated)
    #cv2.waitKey(0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

df = pd.DataFrame(num_faces_in_images)
df.to_csv((FEATURES_OUT_PATH+"number_of_faces_in_images.csv"))