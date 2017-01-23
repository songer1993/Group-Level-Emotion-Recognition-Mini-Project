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

num_faces_in_image = {}
context2 = []
context3 = []
context4 = []

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
    num_faces_in_image[image_id] = num_faces
    if num_faces < 2:
        print "%s: just one face" % image_id

    bodies = []
    # Obtain and store body coordinates from face coordinates
    for idx, face in enumerate(faces):
        x1b, y1b, x2b, y2b = build_body_bounding_box(face, image)
        bodies.append((x1b, y1b, x2b, y2b))

    # Calculate group coordinates, i.e. minimum bounding box
    # including all bodies
    x1g = min([bodies[i][0] for i in range(len(bodies))])
    x2g = max([bodies[i][2] for i in range(len(bodies))])
    y1g = min([bodies[i][1] for i in range(len(bodies))])
    y2g = max([bodies[i][3] for i in range(len(bodies))])

    # Compute width and height of the group bounding box
    wg = float(x2g - x1g)
    hg = float(y2g - y1g)

    # Calculate relative position and size in the group
    # and blurriness of each face
    loc_sca_var = []
    for idx, body in enumerate(bodies):
        body_centre_x = float(body[0]+body[2]) / 2
        body_centre_y = float(body[1]+body[3]) / 2
        wb = body[2] - body[0]
        hb = body[3] - body[1]
        body_loc_x = (body_centre_x - body[0]) / wg
        body_loc_y = (body_centre_y - body[1]) / hg
        body_sca_x = wb / wg
        body_sca_y = hb / hg
        face = faces[idx]
        x1f = face.left()
        x2f = face.right()
        y1f = face.top()
        y2f = face.bottom()
        face_crop = image[body[1]:body[3], body[0]:body[2]]
        face_var = variance_of_laplacian(face_crop)
        loc_sca_var.append([body_loc_x, body_loc_y,
                            body_sca_x, body_sca_y,
                            face_var])
    # flatten the list of lists
    loc_sca_var = [item for sublist in loc_sca_var for item in sublist]

    ious = []
    # Calculate body Intercept over Union IOU
    for i in range(len(bodies) - 1):
        for j in range(i + 1, len(bodies)):
            ious.append(intersection_over_union(bodies[i], bodies[j]))

    # Calculate relative position and scale of group
    # with respect to the whole image
    l_loc_x = float(x1g) / image.shape[1]
    l_loc_y = float(y1g) / image.shape[0]
    l_sca_x = float(wb) / image.shape[1]
    l_sca_y = float(hb) / image.shape[0]

    # Concatenate all features
    context = loc_sca_var + ious + [l_loc_x, l_loc_y, l_sca_x, l_sca_y]
    #print context

    # Append feature to different lists according to
    # the number of faces in the image
    if num_faces == 2:
        context2.append(context)
    elif num_faces == 3:
        context3.append(context)
    else:
        context4.append(context)

# Construct context feature data frame for different number of faces
df_con2 = pd.DataFrame(context2)
df_con3 = pd.DataFrame(context3)
df_con4 = pd.DataFrame(context4)
df_con2.to_csv((FEATURES_OUT_PATH+"2_context_features.csv"))
df_con3.to_csv((FEATURES_OUT_PATH+"3_context_features.csv"))
df_con4.to_csv((FEATURES_OUT_PATH+"4_context_features.csv"))


