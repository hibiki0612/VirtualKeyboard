import cv2
import mediapipe as mp
import math
from time import sleep
import numpy as np
from pynput.keyboard import Controller

class HandDetector:


    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if  self.results.multi_hand_landmarks:
            for handType,handLms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
                myHand={}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] =  (cx, cy)

                if flipType:
                    if handType.classification[0].label =="Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img,myHand["type"],(bbox[0] - 30, bbox[1] - 30),cv2.FONT_HERSHEY_PLAIN,
                                2,(255, 0, 255),2)
        if draw:
            return allHands,img
        else:
            return allHands

    def fingersUp(self,myHand):

        myHandType =myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self,p1, p2, img=None):

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        
        cv2.rectangle(img, button.pos, (x + w, y + h), (224, 255, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
    return img

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        
        

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 2560)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    
    keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
    finalText = ""

    keyboard = Controller()
    
    buttonList = []
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([200 * j + 10, 150 * i + 10], key))

    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
        img = drawAll(img, buttonList)

        if hands:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmark points
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                centerPoint1 = hand1['center']  # center of the hand cx,cy
                handType1 = hand1["type"]  # Handtype Left or Right

                fingers1 = detector.fingersUp(hand1)

                if handType1 == 'Left' and x < lmList1[8][0] < x + w and y < lmList1[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5),(250,235,215), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    l, _, _ = detector.findDistance(lmList1[8], lmList1[12], img)
                    if l < 50:
                        keyboard.press(button.text)
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += button.text
                        
                        sleep(0.15)
                        
                        
    
        cv2.rectangle(img, (100, 1000), (1800, 600), (224, 255, 255), cv2.FILLED)
        space_x = 0
        space_y = 0
        cnt = 0
        for text in finalText:
            if cnt >=34:
                space_y += 80
                space_x = 0
                cnt = 0
            cv2.putText(img, text, (100+space_x, 700+space_y),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
            
            space_x += 50
            cnt +=1
            
            
        '''
        cv2.putText(img, finalText, (100, 700),
                     cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
        '''
       

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()        
                    
        
                   


if __name__ == "__main__":
    main()
