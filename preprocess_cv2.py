#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
from re import A
from matplotlib import image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import numpy as np
# import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
# from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
import matplotlib.cm as cm
import matplotlib.animation as animation
import time 
from model.lanenet.LaneNet import LaneNet
import torch
from preprocess_lanenet import getImages

def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))


def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


undistort_img()


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack(
        (np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_warp(img,
                     dst_size=(320, 180),
                     src=np.float32([(0.28, 0.63), (0.72, 0.63), (0.01, 0.75), (0.99, 0.75)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
                    # (0.35, 0.63), (0.65, 0.63), (0.08, 0.75), (0.92, 0.75) alomst best but not in highly curved lane
                    # (0.395, 0.58), (0.605, 0.58), (0.06, 0.75), (0.94, 0.75)
    # img = region(img)
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def inv_perspective_warp(img,
                         dst_size=(320, 180),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.28, 0.63), (0.72, 0.63), (0.01, 0.75), (0.99, 0.75)])):
                                        # (0.415, 0.58), (0.585, 0.58), (0.06, 0.75), (0.94, 0.75)
                                        # (0.36, 0.54), (0.64, 0.54), (0, 0.85), (1, 0.85)
                                        # 0.35, 0.55
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:, :], axis=0)
    return hist

def sliding_window(img, nwindows=9, margin=40, minpix=1, draw_windows=True):
    # global left_a, left_b, left_c,right_a, right_b, right_c
    left_a, left_b, left_c = [], [], []
    right_a, right_b, right_c = [], [], []

    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    leftDetected = False
    rightDetected = False
    direction = 0 # 1 : right, -1 : left
    balanced = 0 # 1 : right, -1 : left
    
    """
    ::TODO::
    
    left_fit[0]과 right_fit[0] 이 같은 부호라면 detected = 1 아니라면 0
    
    """
    
    # Fit a second order polynomial to each
    # left fit
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        
        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])
        
        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])
        
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        
        leftDetected = True
    except:
        pass
    
    # right fit
    try:
        right_fit = np.polyfit(righty, rightx, 2)

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])
        
        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]] = [0, 100, 255]

        rightDetected = True
    except:
        pass
        
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    # print("lefty :", lefty)
    # print("leftx :", leftx)
    # print("righty :", righty)
    # print("rightx :", rightx)
        
    # Which direction the curved lane made
    if leftDetected and rightDetected:
        if left_fit[0] * right_fit[0] > 0:
            if left_fit[0] > 0:
                direction = -1 
            else:
                direction = 1
                
        if 150 < max(lefty) < 160 and 150 < max(righty) < 160:
            if abs(leftx[0] - (320 - rightx[0])) < 10:
                pass
            elif leftx[0] > (320 - rightx[0]):
                balanced = 320 - rightx[0] # have to MOVE left
            elif leftx[0] < (320 - rightx[0]):
                balanced = -leftx[0] # have to MOVE right
                
        elif 150 < max(lefty) < 160:
            balanced = -leftx[0]
        elif 150 < max(righty) < 160:
            balanced = 320 - rightx[0]
        else:
            if abs(max(lefty) - max(righty)) < 10:
                pass
            elif max(lefty) - max(righty) > 0:
                balanced = -leftx[0]
            elif max(lefty) - max(righty) < 0:
                balanced = 320 - rightx[0]
                
        if max(leftx) - min(leftx) != 0 and max(rightx) - min(rightx) != 0:
            a = (max(lefty) - min(lefty)) / (min(leftx) - max(leftx)) # + : right, - : wrong
            b = (min(righty) - max(righty)) / (min(rightx) - max(rightx)) # + : right, - : wrong
            
            if a < b:
                k = -a
            else:
                k = b
                
            if a != 0 and b != 0:
                _balanced = 1 / b + 1 / a
                # if a > b:
                #     balanced = 1 / b - 1 / a
                # else:
                #     balanced = 1 / (b - a)
            else:
                _balanced = 0 
        else:
            k = 0
            _balanced = 0 

        balanced += 50 * _balanced
            
        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty, direction, balanced, "Both detected : {}".format(balanced) # "a : {:2f}\n b : {:2f}".format(float(a), float(b))
    
    elif leftDetected: # or rightDetected
        if leftx[0] > leftx[-1]:
            direction = -1
        else:
            direction = 1
        
        # if left_fit[0] > 0:
        #     direction = -1
        # else:
        #     direction = 1
            
        if leftx[0] > 30:
            balanced = -leftx[0]
            
        if max(leftx) - min(leftx) != 0:
            k = (min(lefty) - max(lefty)) / (max(leftx) - min(leftx)) # + : right, - : wrong
        else:
            k = 0
        
        # if k < 0:
        #     balanced = -balanced
        
        if k != 0:
            _balanced = 1 / k
        else:
            _balanced = 0
        
        if k > 0:   
            # _balanced = -k
            balanced -= 100 * _balanced
        else:
            # _balanced = k
            # balanced = 320 - leftx[0] - 100 * _balanced
            # balanced = 320 - max(leftx) - 100 * _balanced
            balanced = -85
        
        return out_img, (left_fitx, 0), (left_fit_, 0), ploty, direction, balanced, "Left detected : {}".format(balanced)
    
    elif rightDetected:
        if rightx[0] < rightx[-1]:
            direction = -1
        else:
            direction = 1
        
        if rightx[0] < 290:
            balanced = 320 - rightx[0]
        
        if max(rightx) - min(rightx) != 0:
            k = (min(righty) - max(righty)) / (min(rightx) - max(rightx)) # + : right, - : wrong
        else:
            k = 0
        
        if k != 0:
            _balanced = 1 / k
        else:
            _balanced = 0
        
        if k > 0:   
            # _balanced = k
            balanced += 100 * _balanced
        else:
            # _balanced = k
            # balanced = 320 - rightx[0] + 100 * _balanced
            # balanced = 320 - min(rightx) + 100 * _balanced
            balanced = 85
        
        return out_img, (0, right_fitx), (0, right_fit_), ploty, direction, balanced, "Right detected : {}".format(balanced)
    else:
        return out_img, (0, 0), (0, 0), ploty, direction, balanced, "Not detected"
        


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 14/360  # 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 1.75/360  # 3.7/720 # meters per pixel in x dimension

    try:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                        left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                        right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + \
            left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + \
            right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        # print("car_pos :", car_pos)
        # print(l_fit_x_int, r_fit_x_int)
        # center = (l_fit_x_int + r_fit_x_int) / 100
        
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters
        if center > 0:
            # print("LEFTLEFTLEFTLEFTLEFTLEFTLEFTLEFT")
            direction = "LEFT"
        elif center < 0:
            # print("RIGHTRIGHTRIGHTRIGHTRIGHTRIGHTRIGHT")
            direction = "RIGHT"
        else:
            # print("CENTERCENTERCENTERCENTERCENTERCENTER")
            direction = "CENTER"
        
        return (left_curverad, right_curverad, center, direction)
    except:
        return ([0], [0], 0)


def draw_lanes(img, left_fit, right_fit):
    try:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        color_img = np.zeros_like(img)

        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
        
        inv_perspective = inv_perspective_warp(color_img)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
        return inv_perspective
    except:
        print("Draw Failed!")
        inv_perspective = inv_perspective_warp(img)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
        return inv_perspective

def keepCenter(center, now, file=None):
    if len(center) < 1:
        return center[-1], 1
    else:
        # center.append(now)
        # arr = np.array([center])
        # arr = np.asarray(center, dtype=np.float16)    
        
        
        # print("\n\n\narr[-1] : {}\n\n\n".format(normalized[0]))
        # print("mean :", np.mean(center))
        # print("std :", np.std(center))
        
        if file is not None:
            file.write("mean :{}\n".format(np.mean(center)))
            file.write("std :{}\n\n".format(np.std(center)))
        
        # normalized = normalize(arr)
        # if abs(normalized[0][-1]) > 0.0025:
        if abs(center[-1] - now) > 0.3: # important
                # print("Im AAAAAAAAAAAAAAAA", center[-1])
                return center[-1], -1
            
        if abs(now) > 1:
            if now > 0:
                # print("Im BBBBBBBBBBBBBBBB", 1)
                # return center[-2], -1 # return normalized[-2], -1
                return 1, 1
            else:
                # print("Im CCCCCCCCCCCCCCCC", -1)
                return -1, 1
        else:
            return now, 1  # return normalized[-1], 1

def keepBalance(balance, now, file=None):
    if len(balance) == 1:
        return now, 1
    else:
        # center.append(now)
        # arr = np.array([center])
        # arr = np.asarray(center, dtype=np.float16)    
        
        
        # print("\n\n\narr[-1] : {}\n\n\n".format(normalized[0]))
        # print("mean :", np.mean(balance))
        # print("std :", np.std(balance))
        
        if file is not None:
            file.write("mean :{}\n".format(np.mean(balance)))
            file.write("std :{}\n\n".format(np.std(balance)))
        
        # normalized = normalize(arr)
        # if abs(normalized[0][-1]) > 0.0025:
        
        # if abs(balance[-1] - now) < 50 and abs(now) > 95:
        #     if now > 0:
        #         return 100, 1
        #     else:
        #         return -100, 1
        
        # if balance[-1] * now < 0 and 30 < abs(balance[-1]) < 50:
        #     return now, 1
        
        # if abs(balance[-1] - now) > 35: # and abs(now) > 25 and len(balance) > 500: important
        #         # print("Im AAAAAAAAAAAAAAAA", center[-1])
        #         return balance[-1], -1
            
        # balance = np.array(balance[-10:]) ########## modify ##########
        # left = np.where(balance < -70) ########## modify ##########
        # right = np.where(balance > 70) ########## modify ##########
         
        # print("balance :", balance)
        # print("left :", np.shape(left))
        # print("right :", np.shape(right))
        # # print("len(left) :", left.shape)
        # # print("len(right) :", right.shape)
        
        # if np.shape(left)[1] - np.shape(right)[1] >= 0 and np.shape(left)[1] > 3: ########## modify ##########
        #     print("+++++++++++++++++++++++++++++++++++++++++")
        #     temp = balance[left]
            
        #     if np.mean(temp) < -100:
        #         return -99, 1
        #     else:
        #         return np.mean(temp), 1
                
        # elif np.shape(right)[1] - np.shape(left)[1] >= 0 and np.shape(right)[1] > 3: ########## modify ##########
        #     print("-----------------------------------------")
        #     temp = balance[right]
            
        #     if np.mean(temp) > 100:
        #         return 99, 1
        #     else:
        #         return np.mean(temp), 1
        # else:
        #     return 0, 1
            
        ############################################################
        if abs(now) > 100:
            if now > 0:
                # print("Im BBBBBBBBBBBBBBBB", 1)
                # return center[-2], -1 # return normalized[-2], -1
                return 99, 1
            else:
                # print("Im CCCCCCCCCCCCCCCC", -1)
                return -99, 1
        else:
            return now, 1  # return normalized[-1], 1
        ############################################################
