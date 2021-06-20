import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic#使用地标检测模型

cap = cv2.VideoCapture(1)#电脑自带摄像头
#机器模型
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        #给反馈图像上色
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        #制作检测
        results = holistic.process(image)
        #print(results.face_landmarks)
        #脸的地标，姿势的地标，左手的地标，右手的地标

        #将image的RGB转换成BGR，方便后面读取
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #1.画出脸部地标
        mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACE_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1))

        #右手的地标
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2))

        #左手的地标
        mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))

        #姿势检测
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

        cv2.imshow('draw feed',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()