from preprocess_cv2 import *

# TODO : 경로설정해야 되는 부분 싹 다시

class lane_detection():
    def __init__(self):
        self.center = 0
    
    def getCenter(self):
        return self.center
    
    def main(self):
        # model_path = './log/ENet_last.pth'  # args.model
        # model = LaneNet(arch='ENet')
        model_path = './log/UNet_last.pth'  # args.model
        model = LaneNet(arch='UNet')
        # model_path = './log/DeepLabv3+_last.pth'  # args.model
        # model = LaneNet(arch='DeepLabv3+')

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

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
                
                color, bordered_color, binary = getImages(img, model_path, model, state_dict)
                dst = binary.astype(np.float32)
                dst = perspective_warp(dst, dst_size=(320, 180))
                inv = inv_perspective_warp(dst, dst_size=(320, 180))
                pipe = pipeline(img)
                out_img, curves, lanes, ploty = sliding_window(dst)
                
                # print("img :", img.shape)
                # print("colored :", colfromCenter[-1]or.shape)
                # print("dst :", dst.shape)
                # print("inv :", inv.shape)
                # print("pipe :", pipe.shape)
                # print("out_img :", out_img.shape)
                
                img_ = draw_lanes(color, curves[0], curves[1])
                # color = cv2.resize(color, (640, 360))
                img_ = cv2.resize(img_, (640, 360))
                
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
                    
                    cv2.putText(img_, text="Center : {}".format(curverad[3]), org=(20, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=0.6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
                except:

                    y_center.append(fromCenter[-1])
                    x_count += 1
                    
                cv2.putText(img_, text="Center : {}".format(fromCenter[-1]), org=(20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                        fontScale=0.6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
                
                # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 360))
                
                
                # cv2.imshow("img", img)
                cv2.imshow("detect", img_)
                # cv2.imshow("img", color)
                
                # cv2.waitKey(10)
                
                self.center = fromCenter[-1]
                # TODO : or publish self.center
                
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
        plt.title("lane_test")
        plt.savefig("{}_scatter.png".format(now.strftime('%Y%m%d_%H%M%S%f')))
        plt.show()
        plt.close()