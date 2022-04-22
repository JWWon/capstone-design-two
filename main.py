import cv2
import os
from keras.models import load_model
import numpy as np
import time

def main():
  face = cv2.CascadeClassifier('pre_trained/haarcascade_frontalface_alt.xml')
  # 얼굴 검출기
  leye = cv2.CascadeClassifier('pre_trained/haarcascade_lefteye_2splits.xml')
  # 왼쪽 눈 검출
  reye = cv2.CascadeClassifier('pre_trained/haarcascade_righteye_2splits.xml')

  model = load_model('model/weight.h5')
  path = os.getcwd()

  lbl=['Close','Open'] # 눈을 감았는지 떳는지 두가지로 분류하면 댐

  cap = cv2.VideoCapture(0)
  # 실시간 웹캠 영상 실행 (0이 실시간이라는 뜻)
  
  font = cv2.FONT_HERSHEY_COMPLEX_SMALL
  
  count=0
  score=0
  thicc=2
  rpred=[99]
  lpred=[99]

  while(True):
      ret, frame = cap.read()
      height,width = frame.shape[:2] 

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
      left_eye = leye.detectMultiScale(gray)
      right_eye =  reye.detectMultiScale(gray)

      cv2.rectangle(frame, (0,height-50) , (200,height) , (255,0,0) , thickness=cv2.FILLED )

      for (x,y,w,h) in faces: #얼굴 ROI 영역 생성
          cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 1 )

      for (x,y,w,h) in right_eye:# 오른쪽 ROI 영역 생성
          r_eye=frame[y:y+h,x:x+w]
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
          count=count+1
          r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
          r_eye = cv2.resize(r_eye,(24,24)) #학습시킨 데이터를 (24,24)로 처리했기 때문에 
          r_eye= r_eye/255
          r_eye=  r_eye.reshape(24,24,-1)
          r_eye = np.expand_dims(r_eye,axis=0)
          rpred = model.predict_classes(r_eye)
          
          if(rpred[0]==1):
              lbl='Open' 
          if(rpred[0]==0):
              lbl='Closed'
          break

      for (x,y,w,h) in left_eye:# 왼쪽 ROI 영역 생성
          l_eye=frame[y:y+h,x:x+w]
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
          count=count+1
          l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
          l_eye = cv2.resize(l_eye,(24,24))
          l_eye= l_eye/255
          l_eye=l_eye.reshape(24,24,-1)
          l_eye = np.expand_dims(l_eye,axis=0)
          lpred = model.predict_classes(l_eye)
          
          if(lpred[0]==1):
              lbl='Open'   
          if(lpred[0]==0):
              lbl='Closed'
          break
      

      if(rpred[0]==0 and lpred[0]==0): 
          #왼쪽눈과 오른쪽눈 다 닫혀있을 경우
          score=score+1
          cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
      # if(rpred[0]==1 or lpred[0]==1):
      else:
          score=score-1
          cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


      if score < 0:
          #눈을 뜨고 있을 경우
          score = 0
      cv2.putText(frame,'Point:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
      if score > 15:
          #눈을 일정 시간 감고 있을 경우
          
          cv2.putText(frame, "YOU ARE SLEEPING", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          #you are sleeping 멘트 출력
          # try:
          #   sound.play() #알람 재생
          # except:  # isplaying = False
          #     pass
          if thicc < 16:
              thicc = thicc+2
          else:
              thicc = thicc-2
              if thicc < 2:
                  thicc=2
          cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
          # 15이상 됬을 경우 바탕에 빨간 영역 생성
          if(score > 100):
              # sound.stop()
              print()
      
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()    

if __name__ == '__main__':
  main()