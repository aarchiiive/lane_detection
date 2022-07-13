#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
import sys
from copy import deepcopy
import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor
from model.utils.cli_helper_test import parse_args
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import cv2
from datetime import datetime
# from steering import estimate
import line_fit_video
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
# from advanced import WarpPerspective

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def load_real_time_data(img, transform):
    img = Image.fromarray(img)
    img = transform(img)
    return img


def region(image):
    h, w = image.shape

    # rectangle = np.array([[(w // 8, h), (w // 8 * 7, h),
    #                        (w // 3 * 2, h // 3), (w // 3, h // 3)]])
    rectangle = np.array([[(0, h // 4 * 3), (0, h), (w, h), (w, h // 4 * 3),
                           (w // 4 * 3, int(h * 0.63)), (w // 4, int(h * 0.63))]])

    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, rectangle, 255)

    mask = cv2.bitwise_and(image, mask)
    return mask


def getImages(img, model_path, model, state_dict):
    # if os.path.isdir(os.getcwd() + "/SNU_DATASET/test") == False:
    #     os.mkdir(os.getcwd() + "/SNU_DATASET/test")

    ########################################################
    # args = parse_args()
    img_path = "./SNU_DATASET"
    resize_height = 256 # args.height
    resize_width = 512 # args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # model_path = './log/best_model.pth'  # args.model
    # model = LaneNet(arch='ENet')
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    ########################################################


    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow("test", img)

    # img를 frame -> transfrom 함수로 고쳐야함
    dummy_input = load_real_time_data(img, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.fromarray(img)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(
        outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(
        outputs['binary_seg_pred']).to('cpu').numpy() * 255

    #################### Preprocessing : crop ########################
    binary = np.array(binary_pred, dtype=np.uint8)
    mask = region(binary)
    input = cv2.resize(input, (320, 180))
    
    black = [0, 0, 0]
    
    constant = cv2.copyMakeBorder(mask, 135, 135, 240, 240, cv2.BORDER_CONSTANT, value=black)
    constant = cv2.resize(constant, (320, 180))

    bordered_color = cv2.copyMakeBorder(input, 135, 135, 240, 240, cv2.BORDER_CONSTANT, value=black)
    bordered_color = cv2.resize(bordered_color, (320, 180))
    bordered_color = cv2.cvtColor(bordered_color, cv2.COLOR_BGR2RGB)
    
    # cv2.imwrite("./bordered_color.jpg", bordered_color)
    # cv2.imwrite("./constant.jpg", constant)
    
    return input, bordered_color, constant

        

def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    count = 0

    # w = 1024
    # h = 512
    # mask = np.zeros((h, w))
    # print("Mask shape :", mask.shape)
    # rectangle = np.array([[w // 8, h], [w // 8 * 7, h],
    #                     [w // 3 * 2, h // 3], [w // 3, h // 3]])
    # mask = cv2.fillPoly(mask, [rectangle], 1)

    for img in os.listdir(img_path):
        current = time.time()

        dummy_input = load_test_data(
            img_path + "/" + img, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = model(dummy_input)

        input = Image.open(img_path + '/' + img)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        instance_pred = torch.squeeze(
            outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
        binary_pred = torch.squeeze(
            outputs['binary_seg_pred']).to('cpu').numpy() * 255

        #################### Preprocessing : crop ########################

        binary = np.array(binary_pred, dtype=np.uint8)
        mask = region(binary)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        warped, _, _, _ = perspective_transform(mask)

        estimate(input, binary)

        # cv2.imshow("advanced", advanced)
        # WarpPerspective(input)
        # cv2.imshow("warped", warped)
        cv2.imshow("input", input)
        cv2.imshow("binary", binary)
        cv2.imshow("mask", mask)
        cv2.waitKey(30)
        # time.sleep(2)
        ##################################################################
        cv2.imwrite(os.path.join('./test_dataset/kcity01_output',
                    'input_{}.jpg'.format(img[:6])), input)
        cv2.imwrite(os.path.join('./test_dataset/kcity01_output',
                    'instance_output_{}.jpg'.format(img[:6])), instance_pred.transpose((1, 2, 0)))
        cv2.imwrite(os.path.join('./test_dataset/kcity01_output',
                    'binary_output_{}.jpg'.format(img[:6])), mask)
        # cv2.imwrite(os.path.join('./test_dataset/kcity01_output', 'binary_output_{}.jpg'.format(img[:6])), binary_pred)

        count += 1

        print("Time : {}".format(time.time() - current))
        print("Iterations : {}".format(count))
        
        time.sleep(0.1)

    cv2.destroyAllWindows()


def test_realtime():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    count = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        current = time.time()

        if not ret:
            break

        dummy_input = load_real_time_data(frame, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = model(dummy_input)

        input = Image.fromarray(frame)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        instance_pred = torch.squeeze(
            outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
        binary_pred = torch.squeeze(
            outputs['binary_seg_pred']).to('cpu').numpy() * 255

        #################### Preprocessing : crop ########################

        binary = np.array(binary_pred, dtype=np.uint8)
        mask = region(binary)
        cv2.imshow("input", input)
        cv2.imshow("binary", binary)
        cv2.imshow("mask", mask)
        cv2.waitKey(10)

        ##################################################################
        # cv2.imwrite(os.path.join('./test_dataset/kcity02_output',
        #             'input_{}.jpg'.format(img[:6])), input)
        # cv2.imwrite(os.path.join('./test_dataset/kcity02_output',
        #             'instance_output_{}.jpg'.format(img[:6])), instance_pred.transpose((1, 2, 0)))
        # cv2.imwrite(os.path.join('./test_dataset/kcity02_output',
        #             'binary_output_{}.jpg'.format(img[:6])), mask)
        # cv2.imwrite(os.path.join('./test_dataset/kcity01_output', 'binary_output_{}.jpg'.format(img[:6])), binary_pred)

        count += 1

        print("Time : {}".format(time.time() - current))
        print("Iterations : {}".format(count))

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def postprocess(img):
    """ 
    advanced lane detection 
    """

    with open(os.getcwd() + '/calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    window_size = 5  # how many frames for line smoothing
    left_line = Line(n=window_size)
    right_line = Line(n=window_size)
    detected = False  # did the fast line fit detect the lines?
    left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
    left_lane_inds, right_lane_inds = None, None  # for calculating curvature
    

def perspective_transform(img):
    """
	Execute perspective transform
	"""
    img_size = (img.shape[1], img.shape[0])
    h, w = img.shape
    
    src = np.float32(
        [[int(w * 0.23), h],
         [int(w * 0.77), h],
         [int(w * 0.42), int(h * 0.625)],
         [int(w * 0.58), int(h * 0.625)]])
    dst = np.float32(
        [[int(w * 0.35), h],
         [int(w * 0.65), h],
		 [int(w * 0.35), 0],
		 [int(w * 0.65), 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    return warped, unwarped, m, m_inv

def test_webcam():
    if os.path.isdir(os.getcwd() + "/SNU_DATASET/test") == False:
        os.mkdir(os.getcwd() + "/SNU_DATASET/test")

    ########################################################
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    ########################################################

    cap = cv2.VideoCapture(0)  # set port number

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000 / fps)

    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # out = cv2.VideoWriter(datetime.now().strftime(
    #     '%Y%m%d_%H%M%S') + "_output.mp4", fourcc, fps, (w, h))

    count = 0

    while True:
        ret, frame = cap.read()
        current = time.time()
        count += 1

        if ret:
            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow("test", img)

            # img를 frame -> transfrom 함수로 고쳐야함
            dummy_input = load_real_time_data(frame, data_transform).to(DEVICE)
            dummy_input = torch.unsqueeze(dummy_input, dim=0)
            outputs = model(dummy_input)

            input = Image.fromarray(frame)
            input = input.resize((resize_width, resize_height))
            input = np.array(input)

            instance_pred = torch.squeeze(
                outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
            binary_pred = torch.squeeze(
                outputs['binary_seg_pred']).to('cpu').numpy() * 255

            #################### Preprocessing : crop ########################
            binary = np.array(binary_pred, dtype=np.uint8)
            mask = region(binary)
            # advanced = line_fit_video.annotate_image(input)
            # print("TYPE :", type(advanced))
            # print("SHAPE :",advanced.shape)
            warped, _, _, _ = perspective_transform(mask)
            
            estimate(input, binary)
            # WarpPerspective(input)
            # cv2.imshow("advanced", advanced)
            # cv2.imshow("warped", warped)
            cv2.imshow("input", input)
            cv2.imshow("binary", binary)
            cv2.imshow("mask", mask)
            # cv2.imshow("frame", frame)
            ##################################################################

            ####################### Speed Estimations ########################
            print("Time : {}".format(time.time() - current))
            print("Iterations : {}".format(count))
            ##################################################################

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# COMMAND : python test.py --img ./test_dataset/kcity01 (example)

if __name__ == "__main__":
    test()
    # test_webcam()
    # test_realtime()
