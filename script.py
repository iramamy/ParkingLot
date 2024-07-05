import cv2
from helper import get_bboxes, is_empty, img_diff
import numpy as np

mask_path = './Data/mask.png'
video_path = './Data/data.mp4'

# Read mask
mask = cv2.imread(mask_path, 0)
width, height = mask.shape

# Read video
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_bboxes(connected_components)

# Playback speed (ms)
SPEED = 15 
STEP = 60
FRAME_NUMBER = 0
DIFFS = [None for _ in spots]
STATUS = [None for _ in spots]

previous_frame = None

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    
    if FRAME_NUMBER % STEP == 0 and previous_frame is not None:
        for idx, spot in enumerate(spots):
            x, y, w, h = spot
            spot_cropped = frame[y:y+h, x:x+w, :]
            DIFFS[idx] = img_diff(spot_cropped, previous_frame[y:y+h, x:x+w, :])
        
    if FRAME_NUMBER % STEP == 0:
        if previous_frame is None:
            _array = range(len(spots))
        else:
            array = [id for id in np.argsort(DIFFS) if DIFFS[id]/np.amax(DIFFS)>0.4][::-1]
        for idx in _array:
            x, y, w, h = spots[idx]
            spot_cropped = frame[y:y+h, x:x+w, :]
            spot_status = is_empty(spot_cropped)
            STATUS[idx] = spot_status

        previous_frame = frame.copy()

    for idx, spot in enumerate(spots):
        spot_status = STATUS[idx]
        x, y, w, h = spots[idx]
        color = (0, 255, 0) if spot_status else (0, 0, 255)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)


    cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
    cv2.putText(frame,
        'Available spots: {} / {}'.format(str(sum(STATUS)),
        str(len(STATUS))),
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(SPEED) & 0xFF == ord('q'):
        break

    FRAME_NUMBER += 1

cap.realese()
cv2.destroyAllWindows()
