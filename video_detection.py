from operator import is_
import cv2
from object_detection import ObjectDetection
import numpy as np
import time
from cart import Cart

stream = cv2.VideoCapture(0)
detector = ObjectDetection()

all_items = {
    'Juice':15,'Milk':29.5,'Softdrink':21,'Teabottle':18,'Waterbottle':7,'Apple':25,'Banana':5,'Lime':4,'Onion':13,'Orange':20
}

up = 0
y_prev = 0
i = 10
is_added = False
blink_timer = time.time()

testCart = Cart(all_items)

while(1):
    # print('UP: {} and y: {}'.format(up,y_prev))
    ret, frame = stream.read()
    #print(frame.shape)
    if ret:
        bbox_list = detector.detect(frame)
        #print('BBOX LIST:', bbox_list)
        
        if len(bbox_list) == 1:
            # cv2.rectangle(frame,(bbox_list[0][0], bbox_list[0][1]),(bbox_list[0][0] + bbox_list[0][2], bbox_list[0][1] + bbox_list[0][3]),(126, 65, 64),thickness=2)
            cv2.rectangle(frame,(int(bbox_list[0][0] - bbox_list[0][2]/2), int(bbox_list[0][1] - bbox_list[0][3]/2)),(int(bbox_list[0][0] + bbox_list[0][2]/2), int(bbox_list[0][1] + bbox_list[0][3]/2)),(126, 65, 64),thickness=2)
                    
                    
            # center text according to the face frame
            textSize = cv2.getTextSize(bbox_list[0][4], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = bbox_list[0][0] + bbox_list[0][2] // 2 - textSize[0] // 2
                    
            # draw prediction label
            cv2.putText(frame,bbox_list[0][4],(textX, bbox_list[0][1]-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)   

            y = bbox_list[0][1]
            obj_name = bbox_list[0][4]
        
            if y-y_prev > 0:
                up += 1
            elif y-y_prev < 0:
                up -= 1

            

            if up > i and not is_added:
                #print('in')
                testCart.addItem(obj_name)
                testCart.printCart()
                is_added = True
            elif up < -i and not is_added:
                #print('out')
                testCart.removeItem(obj_name)
                testCart.printCart()
                is_added = True

            y_prev = y
            blink_timer = time.time()

        else:
            if time.time() - blink_timer > 1:
                up = 0
                is_added = False

        

  
    cv2.imshow('main', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break