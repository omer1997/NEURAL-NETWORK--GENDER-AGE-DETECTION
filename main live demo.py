# we need to have or install opencv for this project : do pip install opencv-python
# we can use har cascade also but better to use this
import cv2



def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)  # binary large object to deal with videos   (frame, scalefactor, size of img, mean of rgb img,swapRB)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    #print(detection)   #  GIVES AN array multidimensional array
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]  #   boundary boxes of detection confidence value or probability
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs


# used to detect our face we can also use har cascade classifier as well
faceProto = "opencv_face_detector.pbtxt"          #.pbtxt file that text graph definition in protobuf format
faceModel = "opencv_face_detector_uint8.pb"    #  the protbuf file contains the graph definition as well as the weights of the model. 
                                                #Thus, a pb file is all you need to be able to run a given trained model. We can use harcascade also

# ageProto = "age_deploy.prototxt"
ageModel = "C:/Users/13098/OneDrive/Desktop/Assignments and quizes/Fall-21/CSCI 680 (NN-CV)/opencv project/age_model.onnx"

# genderProto = "gender_deploy.prototxt"
genderModel = "C:/Users/13098/OneDrive/Desktop/Assignments and quizes/Fall-21/CSCI 680 (NN-CV)/opencv project/gender_model.onnx"



faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel) #loading age model
genderNet=cv2.dnn.readNet(genderModel)     # loading gender model

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
genderList = ['Male', 'Female']


video=cv2.VideoCapture(0)

padding=20

while True:
    ret,frame=video.read()   # create video object
    frame,bboxs=faceBox(faceNet,frame)   #iterate through boundary boxes
    for bbox in bboxs:
        # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]    #1 mean x1 , x2 means 2 like indexing 
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]   # to resolve imge resize error as we move here and there
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)   # scale factor , mean value because pre trained model
        genderNet.setInput(blob)
        genderPred=genderNet.forward()  # save predictions in a variable
        gender=genderList[genderPred[0].argmax()] # takes max value from the genderpred


        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]


        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)    # boundary boxes,font,thickness,color,color thickness
    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):        # if q is pressed then exit... ord() function returns the Unicode code from a given character.
        break
video.release()
cv2.destroyAllWindows()