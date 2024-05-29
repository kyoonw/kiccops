#!/usr/bin/env python
# coding: utf-8

# 킥보드 클래스 
# 변수 : 
#     trun - 킥보드 개체가 탐지되는지 확인하는 변수 
#     loc - 킥보드 영역의 픽셀값을 저장하는 배열 
#     ex_cnt - 영역 밖으로 나갔는지 확인하는 프레임 카운트 
#     cnt - 영역에 머무르는 시간을 프레임 카운트 
#     cent_x , cent_y, cent_xy - 해당객체의 센터좌표
#     pt1, p2 - 객체 사각형의 양끝점 (pt1- 좌측하단, pt2 - 우측상단)
#     

# In[1]:


#킥보드 클래스 
class Kickboard :
    def __init__(self,pt1,pt2): # 생성자 함수 
        self.trun = False  # 인지여부 확인 
        self.loc = [[]]  # 킥보드 영역 픽셀값 저장 배열 
        self.ex_cnt = 100 # 영역 밖으로 나갔는지 확인하는 프레임 카운트 
        self.cnt = 0 # 영역 머무르는 시간 프레임 카운트 
        self.cent_x = 0 # 
        self.cent_y = 0
        self.cent_xy = (self.cent_x,self.cent_y)
        self.pt1 = pt1
        self.pt2 = pt2
        self.start_time = time.thread_time()
        self.stay_time = 0
     
    def frame_exp(self):  # 객체가 영역에 없을경우 새는 카운트 
        self.ex_cnt -= 1
        if self.ex_cnt == 0 :  # ex_cnt가 0이 될 경우 객체 삭제 
            return True
        return False
        
    def frame_cnt(self) : # 객체가 영역에 존재하는 시간 카운트 
        self.cnt += 1
        self.ex_cnt =100  # 영역내에서 카운트가 되면 객체가 나간것이 아니므로 나가는 카운트 리셋 
        self.stay_time = int((time.thread_time() - self.start_time)*2)
        if(((time.thread_time() - self.start_time)*2) == 3.0):
            self.sound('voice_test3.mp3')
        if(((time.thread_time() - self.start_time)*2) == 6.0):
            self.sound('voice_test22.mp3')
        
    def tracking(self,new) : # 객체가 다음프레임에 어디로 갔는지 추적 
        if (self.pt2[0]<new[0]<self.pt1[0]) and (self.pt1[1]<new[1]<self.pt2[1]) :
            return True  # 탐지된 객체의 중심좌표가 현 객체의 영역에 위치하면 동일 객체로 인식 
        return False 
        
    def kick_copy(self,new_pt1,new_pt2): # 동일객체로 인식된 경우 새로운 좌표값을 입력해준다. 
        self.pt1 = new_pt1
        self.pt2 = new_pt2
        
    def sound(self,sound_file):
        wn = Audio(sound_file, autoplay=True)
        display(wn)
        
    def __del__(self):
        print("클래스 소멸자 호출됨!")


# In[2]:


def apply_gradient_top(image, start_point, gradient_height):
    # 그라데이션을 적용할 이미지와 동일한 크기의 빈 공간 준비 
    height, width, _ = image.shape
    gradient = np.zeros_like(image, dtype=np.uint8)
    
    # 위쪽 그라데이션 생성
    for y in range(start_point[1], start_point[1] - gradient_height, -1):
        intensity = 1 - (start_point[1] - y) / gradient_height # 색상 강도 결정(0~1)
        gradient[y, :, 0] = 255 * intensity  # 파란색 채널, y좌표에 해당하는 행 전체를 이 색으로 결정함

    # 그라데이션을 이미지에 적용
    mask = np.zeros_like(image, dtype=np.uint8) # 그라데이션을 적용할 영역 표시하기 위해, 빈 이미지 생성
    mask[start_point[1] - gradient_height:start_point[1], :] = gradient[start_point[1] - gradient_height:start_point[1], :]
    blended = cv2.addWeighted(image, 1.0, mask, 0.5, 0)
    
    return blended

def light(img, start_point, end_point,height):
    red_color = (0, 0, 255, 0.5)
    overlay = img.copy()  # 이미지를 복사하여 오버레이 생성
    cv2.rectangle(overlay, start_point, end_point, red_color, -1)  # 오버레이에 사각형 그리기
    cv2.addWeighted(overlay, 0.5, img, 0.9, 0, img) # 이미지 + 오버레이 합치기
    
    # 위쪽 그라데이션 적용
    img = apply_gradient_top(img, start_point,height)
    
    # 아래쪽 영역을 파란색으로 채우되, 원본 이미지는 유지하기 위함
    blue_area = np.zeros_like(img)
    blue_area[end_point[1]:, start_point[0]:end_point[0]] = (255, 0, 0)
    img = cv2.addWeighted(img, 0.8, blue_area, 0.9, 0)
    
    return img


# In[4]:


def video_mode():
    start_point = (0, 525) # 주차금지 영역 
    end_point = (561, 700) 
    kicks = [] #킥보드 객체 배열 
    model = YOLO('best_1.0.pt')
    
    warnning = False 
    warn_key = False 
    try:
        cap = cv2.VideoCapture('test_video2.mp4') # 실시간 영상대신 동영상으로 
        print('동영상 읽기 성공')
    except:
        print('동영상 읽기 실패')
        
    while cap.isOpened():
        ret, img = cap.read()           # 다음 프레임 읽기
        img = cv2.resize(img,(561,1000))
        if ret:
            # 사각형 그리기
            color = (0, 255, 0)  # 선의 색상 (B, G, R)
            thickness = 2  # 선의 두께
            results = model(img) # 딥러닝 모델가동 및 킥보드 탐지 
            img = results[0].plot() #탐지한 킥보드를 영상에 사각형 출력  
         
            for result in results: # 모델이 탐지한 킥보드들 하나씩 분석 
                for box, label in zip(result.boxes.xyxy, result.names):
                    x1, y1, x2, y2 = box.tolist()  # 박스의 (x1, y1, x2, y2) 좌표를 가져옵니다.
                    pt1 = (int(x2),int(y1))
                    pt2 = (int(x1),int(y2))
                    #cv2.rectangle(img,pt1,pt2,color)
                 
                    #센터값 구하기 (세로로는 하위 20퍼)
                    x_cent = int((x1+x2)/2)
                    y_cent = int(((y2-y1)*0.8)+y1)
                    cent_xy = (x_cent,y_cent)
                    cv2.circle(img,cent_xy,10,(255,255,255),3,cv2.LINE_AA) 

                    if y_cent >start_point[1] and y_cent < end_point[1]: # 탐지한 킥보드가 영역 내부로 들어올경우
                        if len(kicks) == 0:   #기존객체가 없을경우 바로 새로운객체로 등록 
                            kicks.append(Kickboard(pt1,pt2))
                        else:
                            new_kick = True
                           
                            for i in range(len(kicks)):       #기존 객체에서 같은 객체가 있는지 확인 
                                if kicks[i].tracking(cent_xy) : 
                                    new_kick = False
                                    kicks[i].frame_cnt()
                                    kicks[i].trun = True
                                    kicks[i].kick_copy(pt1,pt2)
                            if new_kick :
                                kicks.append(Kickboard(pt1,pt2))
                                
                        warn_key = True
            for i in range(len(kicks)):  # 영역내에 탐지안된 객체들을 ex_cnt를 감소시키고 0인객체는 제거 
                if i >= (len(kicks)-1):
                    break
                if kicks[i].trun == False : 
                    if kicks[i].frame_exp():
                        del kicks[i]
                        i -= 1
                kicks[i].trun = False  
            if(warn_key): 
                warnning = True
                warn_key = False

            if (warnning):
                cv2.putText(img, "warnning!!!", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(img, str(len(kicks) ), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(img, str(kicks[i].stay_time ), (0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                img = light(img, start_point, end_point,300)
                # bounding box 그리기
                color = (0,0,255)
                warnning = False
                wran_key = False
            else:
                color = (0,255,0)
            
            cv2.rectangle(img, start_point, end_point, color, thickness)
            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            if cv2.waitKey(1) &0xFF == ord(' '):    # 1ms 동안 키 입력 대기
                return                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break
            
    return


# In[5]:


def detecting_mode(cap):
    start_point = (0, 120)
    end_point = (640, 360)
    kicks = [] 
    model = YOLO('best_1.0.pt')
    
    warnning = False 
    warn_key = False 
    while True:
        ret, img = cap.read()    
        if ret:
            # 사각형 그리기
            color = (0, 255, 0)  # 선의 색상 (B, G, R)
            thickness = 2  # 선의 두께
            results = model(img)
            img = results[0].plot()

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()  # 박스의 (x1, y1, x2, y2) 좌표를 가져옵니다.
                    pt1 = (int(x2),int(y1))
                    pt2 = (int(x1),int(y2))
                    #cv2.rectangle(img,pt1,pt2,color)
                    
                    #센터값 구하기 
                    x_cent = (x1+x2)/2
                    y_cent = (y1+y2)/2
                    cent_xy = (x_cent,y_cent)

                    if y_cent >start_point[1] and y_cent < end_point[1]:
                        if len(kicks) == 0:
                            kicks.append(Kickboard(pt1,pt2))
                        else:
                            new_kick = True
                           
                            for i in range(len(kicks)):
                                if kicks[i].tracking(cent_xy) :
                                    new_kick = False
                                    kicks[i].frame_cnt()
                                    kicks[i].trun = True
                                    kicks[i].kick_copy(pt1,pt2)
                                    print(i)
                        
                            if new_kick :
                                kicks.append(Kickboard(pt1,pt2))
                                
                        warn_key = True
            for i in range(len(kicks)):
                if i >= (len(kicks)-1):
                    break
                if kicks[i].trun == False : 
                    if kicks[i].frame_exp():
                        del kicks[i]
                        i -= 1
                kicks[i].trun = False
            print(len(kicks))    
            if(warn_key):
                warnning = True
                warn_key = False

            if (warnning):
                cv2.putText(img, "warnning!!!", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(img, str(len(kicks) ), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                # bounding box 그리기
                color = (255,0,0)
                warnning = False
                wran_key = False
            else:
                color = (0,255,0)
            
            cv2.rectangle(img, end_point, start_point, color, thickness)
            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            if cv2.waitKey(1) &0xFF == ord(' '):    # 1ms 동안 키 입력 대기
                return                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break
    return


# In[6]:


# 모드선택 
def select_model(cap):
    while True:
        ret, img = cap.read()           # 다음 프레임 읽기
        if ret:
            if cv2.waitKey(1) &0xFF == ord('d'):    # 1ms 동안 키 입력 대기
                detecting_mode(cap)
            if cv2.waitKey(1) &0xFF == ord('v'):    # 1ms 동안 키 입력 대기
                video_mode()
                cap = cv2.VideoCapture(0)
            if cv2.waitKey(1) &0xFF == ord('f'):    # 1ms 동안 키 입력 대기
                break                   # 아무 키라도 입력이 있으면 중지
            cv2.putText(img, "---Menu---", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, "Detecting mode(D),TestVideo(V), Finish(F)", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('camera', img)
    cap.release()                  # 자원 해제
    cv2.destroyAllWindows()        # 모든 창 닫기

# In[7]:


# 찐코드
import cv2
import numpy as np
import time 
from IPython.display import Audio, display
from ultralytics import YOLO
cap = cv2.VideoCapture(0)               # 0번 카메라 장치 연결 


if cap.isOpened():                      # 캡쳐 객체 연결 확인
    print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    select_model(cap)
    
else:
    print("can't open camera.")



# In[57]:


#영역선택 
def click_and_crop(event, x, y, flags, param):
    # refPt와 cropping 변수를 global로 만듭니다.
    global refPt, cropping , img
    global t_region, p_region ,mode

    # 왼쪽 마우스가 클릭되면 (x, y) 좌표 기록을 시작하고
    # cropping = True로 만들어 줍니다.
    if event == cv2.EVENT_LBUTTONDOWN:
        
        refPt = [(x, y)]
        cropping = True
 
    # 왼쪽 마우스 버튼이 놓여지면 (x, y) 좌표 기록을 하고 cropping 작업을 끝냅니다.
    # 이 때 crop한 영역을 보여줍니다.
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        if mode == 't':
            t_region.append((retPt[0],retPt[1]))

        if mode == 'p':
            t_region.append((retPt[0],retPt[1]))
            
        cv2.putText(img, "dragging", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow('camera', img)
        return 
    
def select_region (cap):
    global t_region, p_region ,mode ,img
    cv2.setMouseCallback('camera', click_and_crop)
    t_color=(0,0,255)
    t_color=(255,0,0)
    mode = None
    
    t_region =[]
    p_region = []
    if cap.isOpened():
        
        while True:
            ret, img = cap.read()           # 다음 프레임 읽기
            if ret:
                if cv2.waitKey(1) &0xFF == ord('t'):    # 1ms 동안 키 입력 대기
                    t_region.clear()
                    cv2.putText(img, "tmode", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    mode = 't'
                if cv2.waitKey(1) &0xFF == ord('p'):    # 1ms 동안 키 입력 대기
                    p_region.clear()
                    mode = 'p'


                for i in range(len(t_region)):    
                    cv2.rectangle(img, t_region[i][0], t_region[i][1], t_color, thickness)
                for i in range(len(p_region)):    
                    cv2.rectangle(img, start_point, end_point, p_color, thickness)

            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            if cv2.waitKey(1) &0xFF == ord(' '):    # 1ms 동안 키 입력 대기
                return                   # 아무 키라도 입력이 있으면 중지

    
