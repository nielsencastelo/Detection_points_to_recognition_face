# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 22:03:04 2021

@author: Nielsen Castelo Damasceno
"""

import cv2
import time
import imutils
import dlib

DIMENSAO = 800
totalFrames = 0
skip_frames = 1
url = 0

cap = cv2.VideoCapture(url)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output_real_time.avi',fourcc, 10.0, (800,600))

detector_face = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')


time.sleep(1.0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=DIMENSAO)
    
    
    if not (ret):
        print('Erro no frame')
        st = time.time()
        cap = cv2.VideoCapture(url)
        #cap = cv2.VideoCapture(0)
        print("tempo perdido devido à inicialização  : ",time.time()-st)
        continue
    
    if ret == True:
        if totalFrames % skip_frames == 0:
            #out.write(frame)
            
            # detectar as faces
            deteccoes = detector_face(frame, 1)
            for face in deteccoes:
              pontos = detector_pontos(frame, face)
              for ponto in pontos.parts():
                cv2.circle(frame, (ponto.x, ponto.y), 2, (0,255,0), 1)
            
            cv2.imshow('Detection Points Real time', frame)
            out.write(frame)
    
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()