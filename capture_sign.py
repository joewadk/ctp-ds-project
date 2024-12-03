import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def create_frame_landmarks(results,frame, xyz):

    xyz_skel=xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    
    face=pd.DataFrame()
    pose=pd.DataFrame()
    left_hand=pd.DataFrame()
    right_hand=pd.DataFrame()
    if results.face_landmarks:
        for i,point in enumerate(results.face_landmarks.landmark):
            face.loc[i,['x','y','z']]=[point.x,point.y,point.z]
    if results.pose_landmarks:
        for i,point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i,['x','y','z']]=[point.x,point.y,point.z]
    if results.left_hand_landmarks:
        for i,point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i,['x','y','z']]=[point.x,point.y,point.z]   
    if results.right_hand_landmarks:
        for i,point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i,['x','y','z']]=[point.x,point.y,point.z]
    face=face.reset_index().rename(columns={'index':'landmark_index'}).assign(type='face')
    pose=pose.reset_index().rename(columns={'index':'landmark_index'}).assign(type='pose')
    left_hand=left_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='left_hand')
    right_hand=right_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='right_hand')
    landmarks=pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True)
    #print(landmarks.columns,xyz_skel.columns)
    landmarks=xyz_skel.merge(landmarks,how='left',on=['type','landmark_index'])
    landmarks=landmarks.assign(frame=frame)
    return landmarks


def capture(xyz):
    all_landmarks=[]

    cap=cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        frame=0
        while cap.isOpened():
            frame+=1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)         
            results = holistic.process(image)

            landmarks=create_frame_landmarks(results,frame,xyz)
            all_landmarks.append(landmarks)


            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image, 
                results.face_landmarks, 
                mp_holistic.FACEMESH_CONTOURS, 
                landmark_drawing_spec=None, 
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
             #in theory these should work. ill have to test it. base functionality with pose and face works. ---edit: it works. flawlessly now too
            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                            
        cap.release()
        cv2.destroyAllWindows()
    return all_landmarks
def capture_sign():
    BASE_DIR = 'data/asl-signs/'
    train = pd.read_csv(f'{BASE_DIR}/train.csv')
    xyz=pd.read_parquet(f'{BASE_DIR}/train_landmark_files/16069/695046.parquet')
    all_landmarks=capture(xyz)
    all_landmarks=pd.concat(all_landmarks).reset_index(drop=True)
    all_landmarks.to_parquet('landmarks.parquet')
    return

if __name__ == "__main__":
    capture_sign()
    print('Landmarks saved')



