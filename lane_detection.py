#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocess_cv2 import *
import rospy
import sys, os
from std_msgs.msg import Float64

currentPath = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class LaneDetection():
    def __init__(self, webcam_port=0):
        global currentPath
        # setting port
        self.cap = cv2.VideoCapture(webcam_port)
        self.now = datetime.datetime.now() # estimate time
        self.current = 0
        
        # model preparation(ex. U-Net)
        self.model_path = currentPath + '/log/UNet_last.pth'
        self.model = LaneNet(arch='UNet')
        self.state_dict = torch.load(self.model_path)
        self.model.load_state_dict(self.state_dict)
        
        # plotting 
        self.y_center = []
        self.x_count = 0
        self.fromCenter = [0]
        self.detected = 0
        
        # recording
        self.f = open("./records/{}.txt".format(self.now.strftime('%Y%m%d_%H%M%S')), 'w')
        
        
    def getFrame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        
    def detect(self):
        current = time.time()
        
        img = self.getFrame()
        img = cv2.resize(img, (320, 180))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        color, bordered_color, binary = getImages(img, self.model_path, self.model, self.state_dict)
        dst = binary.astype(np.float32)
        dst = perspective_warp(dst, dst_size=(320, 180))
        inv = inv_perspective_warp(dst, dst_size=(320, 180))
        # pipe = pipeline(img)
        out_img, curves, lanes, ploty = sliding_window(dst)
        
        # print("img :", img.shape)
        # print("colored :", colfromCenter[-1]or.shape)
        # print("dst :", dst.shape)
        # print("inv :", inv.shape)
        # print("pipe :", pipe.shape)
        # print("out_img :", out_img.shape)
        
        img_ = draw_lanes(color, curves[0], curves[1])
        img_ = cv2.resize(img_, (640, 360))
        
        try:
            curverad = get_curve(img, curves[0], curves[1])
            centered, isOutliner = keepCenter(self.fromCenter, curverad[2])

            if isOutliner == 1:
                self.fromCenter.append(centered)
                self.y_center.append(self.fromCenter[-1])
                self.x_count += 1
            elif isOutliner == -1:
                self.y_center.append(self.fromCenter[-1])
                self.x_count += 1
            
            cv2.putText(img_, text="Center : {}".format(curverad[3]), org=(20, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
            
            self.detected += 1
        except:
            self.y_center.append(self.fromCenter[-1])
            self.x_count += 1
                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        cv2.imshow("img", img)
        cv2.imshow("detect", img_)
        
        self.center = self.fromCenter[-1]
        
        print("\nCenter : {}".format(self.fromCenter[-1]))
        print("\nTime : {}s".format(time.time() - current))
        print("\nFrame : {}s\n\n\n".format(float(1 / (time.time() - current))))

        self.f.write("\nCenter : {}".format(self.fromCenter[-1]))
        self.f.write("\nTime : {}s\n\n".format(time.time() - current))
        
        if len(self.fromCenter) > 5:
            self.fromCenter = self.fromCenter[-5:]
    
if __name__ == "__main__":
    
    LD = LaneDetection(0) # default webcam port = 0 -> refer to definition
    
    rospy.init_node("lane_detection")
        
    while not rospy.is_shutdown():        
        ld_pub = rospy.Publisher('lane_center', Float64, queue_size=1)

        LD.detect()

        ld_pub.publish(LD.center)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    print("Detected : {}%".format(LD.detected / LD.x_count * 100))
    print("Not detected : {}%".format(100 - LD.detected / LD.x_count * 100))
    
    LD.cap.release()
    cv2.destroyAllWindows()
    
    LD.f.close()
    
    plt.scatter(range(LD.x_count), LD.y_center)
    plt.xlabel("frames")
    plt.ylabel("center")
    plt.title("Center Position")
    plt.savefig("./plots/{}_scatter.jpg".format(LD.now.strftime('%Y%m%d_%H%M%S%f')))
    plt.show()
    plt.close()
    
    