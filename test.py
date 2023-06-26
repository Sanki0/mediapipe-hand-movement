import math
import cv2
import mediapipe as mp

video = cv2.VideoCapture('test.mp4')
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
contador = 0
check = True
hand_open = False
hand_closed = False
prev_hand_open = False
prev_hand_closed = False
hand_action_changed = False
action_start_frame = 0
action_end_frame = 0

while True:
    success, img = video.read()
    if not success:
        break
    results = Pose.process(img)
    points = results.pose_landmarks

    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)

    # detect if hand is open or closed
    h, w, _ = img.shape

    if points:
        indexDY = points.landmark[pose.PoseLandmark.RIGHT_INDEX].y
        indexDX = points.landmark[pose.PoseLandmark.RIGHT_INDEX].x
        indexIY = points.landmark[pose.PoseLandmark.LEFT_INDEX].y
        indexIX = points.landmark[pose.PoseLandmark.LEFT_INDEX].x
        pinkyDY = points.landmark[pose.PoseLandmark.RIGHT_PINKY].y
        pinkyDX = points.landmark[pose.PoseLandmark.RIGHT_PINKY].x
        pinkyIY = points.landmark[pose.PoseLandmark.LEFT_PINKY].y
        pinkyIX = points.landmark[pose.PoseLandmark.LEFT_PINKY].x

        wristDY = points.landmark[pose.PoseLandmark.RIGHT_WRIST].y
        wristDX = points.landmark[pose.PoseLandmark.RIGHT_WRIST].x
        wristIY = points.landmark[pose.PoseLandmark.LEFT_WRIST].y
        wristIX = points.landmark[pose.PoseLandmark.LEFT_WRIST].x

        thumbDY = points.landmark[pose.PoseLandmark.RIGHT_THUMB].y
        thumbDX = points.landmark[pose.PoseLandmark.RIGHT_THUMB].x
        thumbIY = points.landmark[pose.PoseLandmark.LEFT_THUMB].y
        thumbIX = points.landmark[pose.PoseLandmark.LEFT_THUMB].x

        dist_index_pinky = math.hypot(indexDX - pinkyDX, indexDY - pinkyDY)
        dist_index_thumb = math.hypot(indexDX - thumbDX, indexDY - thumbDY)
        dist_index_wrist = math.hypot(indexDX - wristDX, indexDY - wristDY)
        dist_pinky_thumb = math.hypot(pinkyDX - thumbDX, pinkyDY - thumbDY)
        dist_pinky_wrist = math.hypot(pinkyDX - wristDX, pinkyDY - wristDY)
        dist_thumb_wrist = math.hypot(thumbDX - wristDX, thumbDY - wristDY)

        print(f'index_pinky:{dist_index_pinky} index_thumb:{dist_index_thumb} index_wrist:{dist_index_wrist} pinky_thumb:{dist_pinky_thumb} pinky_wrist:{dist_pinky_wrist} thumb_wrist:{dist_thumb_wrist}')

        if dist_index_pinky > 0.020 and dist_index_wrist > 0.04 :
            hand_open = True
            hand_closed = False
        else:
            hand_open = False
            hand_closed = True

        # check for hand action change
        if hand_open != prev_hand_open or hand_closed != prev_hand_closed:
            if not hand_action_changed:
                hand_action_changed = True
                if hand_open:
                    action_start_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    action_end_frame = video.get(cv2.CAP_PROP_POS_FRAMES) - 1

        # update previous hand states
        prev_hand_open = hand_open
        prev_hand_closed = hand_closed

        # print(f'hand_open:{hand_open} hand_closed:{hand_closed}')

        texto = f'CANT.: {contador}'
        cv2.rectangle(img, (20, 240), (340, 120), (255, 0, 0), -1)
        cv2.putText(img, texto, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    cv2.imshow('Resultado', img)
    cv2.waitKey(40)

    # check if action has completed
    if hand_action_changed:

        if video.get(cv2.CAP_PROP_POS_FRAMES) > action_end_frame:
            hand_action_changed = False
            print("Hand action started at frame", int(action_start_frame))
            print("Hand action ended at frame", int(action_end_frame))
        
        contador += 1
