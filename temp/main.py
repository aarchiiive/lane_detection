from preprocess_cv2 import *

mode = "images"
# mode = "images_with_plotting"
# mode = "real_time"
# mode = "real_time_with_plotting"

# model_path = './log/ENet_last.pth'  # args.model
# model = LaneNet(arch='ENet')
model_path = './log/UNet_last.pth'  # args.model
model = LaneNet(arch='UNet')
# model_path = './log/DeepLabv3+_last.pth'  # args.model
# model = LaneNet(arch='DeepLabv3+')

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
    
    
if mode == "images":
    images = sorted(os.listdir("./SNU_DATASET2"))  # set image path
    fromCenter = [0]
    x_count = 0
    y_center = []
    detected = 0
    now = datetime.datetime.now()

    f = open("./{}.txt".format(now.strftime('%Y%m%d_%H%M%S')), 'w')

    for i in range(len(images)):
        current = time.time()
        # img = random.choice(images)
        img = images[i]
        imgName = img
        img = cv2.imread("./SNU_DATASET2/" + img)
        
        img = cv2.resize(img, (640, 360))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        color, bordered_color, binary = getImages(img, model_path, model, state_dict)
        dst = binary.astype(np.float32)
        dst = perspective_warp(dst, dst_size=(640, 360))
        inv = inv_perspective_warp(dst, dst_size=(640, 360))
        pipe = pipeline(img)
        out_img, curves, lanes, ploty = sliding_window(dst)
        
        # print("img :", img.shape)
        # print("colored :", color.shape)
        # print("dst :", dst.shape)
        # print("inv :", inv.shape)
        # print("pipe :", pipe.shape)
        # print("out_img :", out_img.shape)
        
        img_ = draw_lanes(color, curves[0], curves[1])

        # Visualize undistortion
        # fig = plt.figure(figsize=(20, 10))
        # ax1 = fig.add_subplot(2, 3, 1)
        # ax2 = fig.add_subplot(2, 3, 2)
        # ax3 = fig.add_subplot(2, 3, 3)
        # ax4 = fig.add_subplot(2, 3, 4)
        # ax5 = fig.add_subplot(2, 3, 5)
        # ax6 = fig.add_subplot(2, 3, 6)

        # ax1.imshow(binary)
        # ax1.set_title('Binary', fontsize=20)
        # ax2.imshow(dst, cmap='gray')
        # ax2.set_title('Warped Image', fontsize=20)
        # ax3.imshow(inv)
        # ax3.set_title('Inverse', fontsize=20)
        # ax4.imshow(pipe)
        # ax4.set_title('Pipeline', fontsize=20)
        # ax5.imshow(out_img)

        color = cv2.resize(color, (640, 360))
        
        try:
            # ax5.set_title('Sliding Windows', fontsize=20)
            # ax5.plot(curves[0], ploty, color='yellow', linewidth=1)
            # ax5.plot(curves[1], ploty, color='yellow', linewidth=1)
            curverad = get_curve(img, curves[0], curves[1])
            centered, isOutliner = keepCenter(fromCenter, curverad[2], f)
            print("centered, isOutliner :", centered, isOutliner)

            if isOutliner == 1:
                fromCenter.append(centered)
                y_center.append(fromCenter[-1])
            elif isOutliner == -1:
                y_center.append(fromCenter[-1])

            detected += 1
            
            cv2.putText(color, text="Center : {}".format(curverad[3]), org=(20, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        except:
            y_center.append(fromCenter[-1])
        
        # img_ = img_.astype(np.float32)
        # cv2.putText(img_, "center : {}".format(fromCenter[-1]), org=(30, 50),fontFace=cv2.FONT_HERSHEY_COMPLEX,
        #             fontScale=0.2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        # cv2.cvtColor(img_, cv2.COLOR_HSlen(x_count)
        # ax6.set_title('center : {}'.format(fromCenter[-1]), fontsize=20)
        # plt.savefig("./figures_snu_2/{}".format(imgName))
        # plt.close(fig)
        # fig.clear()
        
        
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        cv2.putText(color, text="Center : {}".format(fromCenter[-1]), org=(20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=0.6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
       
        cv2.imshow("img", color)
        
        cv2.waitKey(10)
        
        print("\nCenter : {}".format(fromCenter[-1]))
        print("\n\nTime : {}s\n\n\n".format(time.time() - current))
        f.write("\nCenter : {}".format(fromCenter[-1]))
        f.write("\nTime : {}s\n\n".format(time.time() - current))

        fromCenter = fromCenter[-5:]
        # if i == 500: break

    print(len(y_center))
    print(x_count)

    print("Detected : {}%".format(detected / len(images) * 100))
    print("Not detected : {}%".format(100 - detected / len(images) * 100))

    cv2.destroyAllWindows()
    f.close()

    plt.scatter(range(x_count), y_center)
    plt.xlabel("frames")
    plt.ylabel("center")
    plt.show()
    plt.savefig("scatter_{}.jpg".format(now.strftime('%Y%m%d_%H%M%S%f')))

elif mode == "images_with_plotting":
    images = os.listdir("./SNU_DATASET2")  # set image path
    fromCenter = [0]
    x_length = range(len(images))
    y_center = []
    detected = 0
    now = datetime.datetime.now()

    f = open("./{}.txt".format(now.strftime('%Y%m%d_%H%M%S')), 'w')

    for i in range(len(images)):
        current = time.time()
        # img = random.choice(images)
        img = images[i]
        imgName = img

        img = cv2.imread("./SNU_DATASET2/" + img)
        img = cv2.resize(img, (640, 360))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colored, _, binary = getImages(img, model_path, model, state_dict)

        dst = binary.astype(np.float32)

        dst = perspective_warp(dst, dst_size=(640, 360))
        inv = inv_perspective_warp(dst, dst_size=(640, 360))
        pipe = pipeline(img)
        out_img, curves, lanes, ploty = sliding_window(dst)

        img_ = draw_lanes(colored, curves[0], curves[1])

        ################################################################################
        # try:
        #     curverad = get_curve(img, curves[0], curves[1])
        #     centered, isOutliner = keepCenter(fromCenter, curverad[2], f)

        #     if isOutliner == 1:
        #         fromCenter.append(centered)
        #         y_center.append(fromCenter[-1])
        #     elif isOutliner == -1:
        #         y_center.append(fromCenter[-1])

        #     detected += 1
        # except:
        #     y_center.append(fromCenter[-1])
        #     pass
        ################################################################################

        print("\nCenter : {}".format(fromCenter[-1]))
        print("\n\nTime : {}s\n\n\n".format(time.time() - current))
        f.write("\nCenter : {}".format(fromCenter[-1]))
        f.write("\nTime : {}s\n\n".format(time.time() - current))

        fromCenter = fromCenter[-5:]
        # if i == 500: break

    print(len(y_center))
    print(len(x_length))

    print("Detected : {}%".format(detected / len(images) * 100))
    print("Not detected : {}%".format(100 - detected / len(images) * 100))

    f.close()

    plt.scatter(x_length, y_center)
    plt.show()
    plt.savefig("scatter_{}.jpg".format(now.strftime('%Y%m%d_%H%M%S%f')))

elif mode == "real_time":
    fromCenter = [0]
    x_count = 0
    y_center = []
    detected = 0
    now = datetime.datetime.now()

    f = open("./{}.txt".format(now.strftime('%Y%m%d_%H%M%S')), 'w')
    
    cap = cv2.VideoCapture(0)

    # w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    # delay = round(1000/fps)
    
    # out = cv2.VideoWriter('{}_output.avi'.format(now.strftime('%Y%m%d_%H%M%S')), fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        current = time.time()

        if ret:
            img = frame
            img = cv2.resize(img, (320, 180))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            colored, _, binary = getImages(img, model_path, model, state_dict)
            dst = binary.astype(np.float32)
            dst = perspective_warp(dst, dst_size=(320, 180))
            inv = inv_perspective_warp(dst, dst_size=(320, 180))
            pipe = pipeline(img)
            out_img, curves, lanes, ploty = sliding_window(dst)
            print("img :", img.shape)
            print("colored :", colored.shape)
            print("dst :", dst.shape)
            print("inv :", inv.shape)
            print("pipe :", pipe.shape)
            print("out_img :", out_img.shape)
            img_ = draw_lanes(colored, curves[0], curves[1])

            try:
                curverad = get_curve(img, curves[0], curves[1])
                centered, isOutliner = keepCenter(fromCenter, curverad[2], f)

                if isOutliner == 1:
                    fromCenter.append(centered)
                    y_center.append(fromCenter[-1])
                    x_count += 1
                elif isOutliner == -1:
                    y_center.append(fromCenter[-1])
                    x_count += 1

                detected += 1
            except:
                y_center.append(fromCenter[-1])
                x_count += 1

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 360))
            img_ = cv2.resize(img_, (640, 360))
            cv2.imshow("img", img)
            cv2.imshow("detect", img_)

            print("\nCenter : {}".format(fromCenter[-1]))
            print("\nTime : {}s".format(time.time() - current))
            print("\nFrame : {}s\n\n\n".format(
                float(1 / (time.time() - current))))

            f.write("\nCenter : {}".format(fromCenter[-1]))
            f.write("\nTime : {}s\n\n".format(time.time() - current))

            fromCenter = fromCenter[-5:]

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(len(y_center))
    print(len(x_count))

    print("Detected : {}%".format(detected / x_count * 100))
    print("Not detected : {}%".format(100 - detected / x_count * 100))

    f.close()

    plt.scatter(range(x_count), y_center)
    plt.xlabel("frames")
    plt.ylabel("center")
    plt.show()
    plt.savefig("scatter_{}.jpg".format(now.strftime('%Y%m%d_%H%M%S%f')))

elif mode == "real_time_with_plotting":
    fromCenter = [0]
    x_count = 0
    y_center = []
    detected = 0
    now = datetime.datetime.now()

    f = open("./{}.txt".format(now.strftime('%Y%m%d_%H%M%S')), 'w')

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        current = time.time()

        if ret:
            img = frame
            img = cv2.resize(img, (1280, 720))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            colored, _, binary = getImages(img, model_path, model, state_dict)

            dst = binary.astype(np.float32)
            dst = perspective_warp(dst, dst_size=(1280, 720))
            inv = inv_perspective_warp(dst, dst_size=(1280, 720))
            pipe = pipeline(img)
            out_img, curves, lanes, ploty = sliding_window(dst)
            img_ = draw_lanes(colored, curves[0], curves[1])

            # Visualize undistortion
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(2, 3, 1)
            ax2 = fig.add_subplot(2, 3, 2)
            ax3 = fig.add_subplot(2, 3, 3)
            ax4 = fig.add_subplot(2, 3, 4)
            ax5 = fig.add_subplot(2, 3, 5)
            ax6 = fig.add_subplot(2, 3, 6)

            ax1.imshow(binary)
            ax1.set_title('Binary', fontsize=20)
            ax2.imshow(dst, cmap='gray')
            ax2.set_title('Warped Image', fontsize=20)
            ax3.imshow(inv)
            ax3.set_title('Inverse', fontsize=20)
            ax4.imshow(pipe)
            ax4.set_title('Pipeline', fontsize=20)
            ax5.imshow(out_img)

            try:
                ax5.set_title('Sliding Windows', fontsize=20)
                ax5.plot(curves[0], ploty, color='yellow', linewidth=1)
                ax5.plot(curves[1], ploty, color='yellow', linewidth=1)
                curverad = get_curve(img, curves[0], curves[1])
                centered, isOutliner = keepCenter(fromCenter, curverad[2], f)
                print("centered, isOutliner :", centered, isOutliner)

                if isOutliner == 1:
                    fromCenter.append(centered)
                    y_center.append(fromCenter[-1])
                    x_count += 1
                elif isOutliner == -1:
                    y_center.append(fromCenter[-1])
                    x_count += 1

                detected += 1
            except:
                y_center.append(fromCenter[-1])
                x_count += 1

            ax6.imshow(img_, cmap='hsv')
            ax6.set_title('center : {}'.format(fromCenter[-1]), fontsize=20)
            plt.savefig(
                "./test_webcam/{}.jpg".format(now.strftime('%Y%m%d_%H%M%S%f')))
            plt.close(fig)
            fig.clear()

            cv2.imshow("img", img)
            cv2.imshow("detect", img_)

            print("\nCenter : {}".format(fromCenter[-1]))
            print("\nTime : {}s".format(time.time() - current))
            print("\nFrame : {}s\n\n\n".format(
                float(1 / (time.time() - current))))

            f.write("\nCenter : {}".format(fromCenter[-1]))
            f.write("\nTime : {}s\n\n".format(time.time() - current))

            fromCenter = fromCenter[-5:]

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(len(y_center))
    print(x_count)

    print("Detected : {}%".format(detected / x_count * 100))
    print("Not detected : {}%".format(100 - detected / x_count * 100))

    f.close()

    plt.scatter(range(x_count), y_center)
    plt.xlabel("frames")
    plt.ylabel("center")
    plt.show()
    plt.savefig("scatter_{}.jpg".format(now.strftime('%Y%m%d_%H%M%S%f')))
