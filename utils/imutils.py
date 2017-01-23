import openface
import dlib
import cv2
import numpy as np
import math
import dlib
from scipy.spatial import distance


def build_body_bounding_box(face, image):
    # coordinates of face bounding box
    x1f = face.left()
    x2f = face.right()
    y1f = face.top()
    y2f = face.bottom()
    # width and height of face
    wf = x2f - x1f
    hf = y2f - y1f
    # body will be 3 times larger than face
    # but cannot go beyond image
    hi = image.shape[0]
    wi = image.shape[1]
    x1b = clamp(int(x1f - wf), 0, wi)
    x2b = clamp(int(x2f + wf), 0, wi)
    y1b = clamp(int(y1f - 0.5 * hf), 0, hi)
    y2b = clamp(int(y2f + 1.5 * hf), 0, hi)
    return (x1b, y1b, x2b, y2b)


def build_left_eye_bounding_box(face_size):
    x1l = int(0.1 * face_size)
    x2l = int(0.4 * face_size)
    y1l = int(0.0 * face_size)
    y2l = int(0.3 * face_size)
    return (x1l, y1l, x2l, y2l)

def build_right_eye_bounding_box(face_size):
    x1r = int(0.6 * face_size)
    x2r = int(0.9 * face_size)
    y1r = int(0.0 * face_size)
    y2r = int(0.3 * face_size)
    return (x1r, y1r, x2r, y2r)

def build_mouth_bounding_box(face_size):
    x1m = int(0.2 * face_size)
    x2m = int(0.8 * face_size)
    y1m = int(0.5 * face_size)
    y2m = int(0.8 * face_size)
    return (x1m, y1m, x2m, y2m)

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def annotate_all_faces_and_landmarks(im, faces, bodies, all_landmarks):
    im = im.copy()
    # Annotate each face
    for idx, face in enumerate(faces):
        cv2.rectangle(im, (face.left(), face.top()),
                      (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(im, str(idx+1), (face.left(), face.top()),
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255))

    # Annotate each body
    for body in bodies:
        cv2.rectangle(im, (body[0], body[1]),
                      (body[2], body[3]), (255, 0, 0), 2)

    # Calculate group coordinates, i.e. minimum bounding box
    # including all bodies
    x1g = min([bodies[i][0] for i in range(len(bodies))])
    x2g = max([bodies[i][2] for i in range(len(bodies))])
    y1g = min([bodies[i][1] for i in range(len(bodies))])
    y2g = max([bodies[i][3] for i in range(len(bodies))])
    cv2.rectangle(im, (x1g, y1g),
                  (x2g, y2g), (255, 255, 255), 2)

    # Annotate each face' landmarks
    for landmarks in all_landmarks:
        for idx, point in enumerate(landmarks):
            pos = (point[0], point[1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou