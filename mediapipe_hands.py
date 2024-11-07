import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#load mediapipe stuff
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#mediapipe image detection
def mediapipe_detect(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

#mediapipe drawing
def draw_landmarks(image, results):
    #face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 155, 0), thickness=1, circle_radius=1))

    # pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 0), thickness=2, circle_radius=2))

    #left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

    #right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        

def basic_mediapipe():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Use the holistic model for detection
        image, results = mediapipe_detect(frame, holistic)

        # Draw landmarks on the image
        draw_landmarks(image, results)

        # Display the frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#extract keypoints
def extract_keypoints(results):
    pose= np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face= np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])


import os
dataset_path= os.path.join('data')
actions= np.array(['hello','thanks','iloveyou'])
no_sequences= 30
sequence_length= 30
def make_dirs():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(dataset_path, action, str(sequence)))
            except FileExistsError:
                pass

def data_load():
    #run only to make the dataset
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break
                image, results = mediapipe_detect(frame, holistic)

                #draw the landmarks
                draw_landmarks(image, results)
                if frame_num==0: #dataset collection starting   
                    cv2.putText(image, 'STARTING COLLECTION',(120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),(15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('Camera', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),(15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('Camera', image)
                keypoints= extract_keypoints(results) # save the keypoints 
                npy_path= os.path.join(dataset_path, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                #cv break 
                if cv2.waitKey(10) & 0xFF == ord('q'):   
                    break

    #RELEASE MEEEEE
    cap.release()
    cv2.destroyAllWindows()
def new_model(): #run only if you want to make a new model
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard
    sequences, labels= [], []
    label_map= {label:num for num, label in enumerate(actions)}
    for action in actions:
        for sequence in range(no_sequences):
            window=[]
            for frame_num in range(sequence_length):
                res= np.load(os.path.join(dataset_path, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X= np.array(sequences) #sklearn
    y= to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.05)
    #tensorboard stuff, if you have it installed you can run 
    ''' 
    tensorboard --logdir=.
    '''
    log_dir= os.path.join('Logs') #training logs from tensorboard
    tb_callback= TensorBoard(log_dir=log_dir)
    

    model= Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
    return model

def load_model(): #run only if you have action.h5
    from tensorflow.keras.models import load_model
    model= load_model('action.h5')
    return model

def insights(model, X_train, y_train): #2 cm's, accuracy metrics
    from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix
    yhat= model.predict(X_train)
    y_true= np.argmax(y_train, axis=1).tolist()
    yhat=np.argmax(yhat, axis=1).tolist()
    print(multilabel_confusion_matrix(y_true, yhat))
    print(confusion_matrix(y_true, yhat))
    print(accuracy_score(y_true, yhat))

def ml_model():
    #model=new_model(X_train, y_train)
    model= load_model()

    cap = cv2.VideoCapture(0)
    sequence= []
    sentence= []
    threshold= 0.8
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        #image manipulation
        image, results = mediapipe_detect(frame, holistic)
        draw_landmarks(image, results)

        #prediction logic
        keypoints= extract_keypoints(results)
        sequence.append(keypoints)
        sequence= sequence[-30:]
        if len(sequence)==30:
            res= model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            #sentence.append(actions[np.argmax(res)])

            # visualize the prediction
            if np.max(res)>threshold:
                if len(sentence)>0:
                    if actions[np.argmax(res)]!=sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence)>5:
                sentence= sentence[-5:]
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)  

        # Display the frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #uncomment as needed
    #basic_mediapipe()
    ml_model()
