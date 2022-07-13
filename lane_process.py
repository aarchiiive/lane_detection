#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/src/sensor/camera/lane_detection")
from lane_detection import LaneDetection
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float64

if __name__ == "__main__":
    
    LD = LaneDetection() # 웹캠 포트 기본값 = 0 -> 수정 가능 (정의 참고)
        
    rospy.init_node("lane_dection")
        
    while not rospy.is_shutdown():
        ld_pub = rospy.Publisher('center', Float64, queue_size=1)
        lane_msg = Float64()
        LD.detect()
        
        lane_msg.center = float(LD.center)
        ld_pub.publish(lane_msg)
        
    print("Detected : {}%".format(LD.detected / LD.x_count * 100))
    print("Not detected : {}%".format(100 - LD.detected / LD.x_count * 100))