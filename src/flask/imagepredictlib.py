import cv2
import socket


class Camera(object):
    def __init__(self):
        self.CAMERA_ID = 0
        self.video = cv2.VideoCapture(self.CAMERA_ID)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, frame = cv2.imencode('.jpg', image)
        return frame


class Cameras(object):
    def __init__(self):
        self.camidlist = self.num_cam()
        self.CAMERA_NUM = len(self.camidlist)
        self.video = [cv2.VideoCapture(camid) for camid in self.camidlist]

    def __del__(self):
        for i in range(self.CAMERA_NUM):
            self.video[i].release()

    def num_cam(self):
        SEARCH_NUM = 20
        found_cam_id_list = []
        for i1 in range(0, SEARCH_NUM):
            cap1 = cv2.VideoCapture(i1)
            if cap1.isOpened(): 
                # print("VideoCapture(", i1, ") : Found")
                found_cam_id_list.append(i1)
            cap1.release()
        return found_cam_id_list

    def get_frame(self):
        imagelist = []
        for i in range(self.CAMERA_NUM):
            success, image = self.video[i].read()
            cv2.putText(image,
                        text="CamID: " + str(self.camidlist[i]),
                        org=(0, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_4)
            imagelist.append(image)
        h_min = min(image.shape[0] for im in imagelist)
        im_list_resize = [cv2.resize(image, (int(image.shape[1] * h_min / image.shape[0]), h_min)) for image in imagelist]
        image = cv2.hconcat(im_list_resize)
        ret, frame = cv2.imencode('.jpg', image)
        return frame


class PLC(object):
    def __init__(self, ip, port):
        super().__init__()  # 必須
        self.ipadress = ip
        self.port_no = port
        # HEAD_3E = "500000FF03FF00"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((ip, port))
            s.settimeout(2.0)
            message = bytes("500000FF03FF00000C002801010000", "ascii")
            s.send(message)  # CPU形名読み出し0101-0000
            response = s.recv(4096)
            response = response.decode("ascii")
            # print(response)
            self.code = response[-4:]
            self.model = response[-20:-4].replace(" ", "")
            self.series = self.model[0]
            # 以下確認のための出力
            # print("code:" + self.code)
            print("model:" + self.model)
            # print("series:" + self.series)

    def __dell__(self):
        pass

    def read(self, dev_code, dev_points, access_points):    # 一括読出し0401
        HEAD = "500000FF03FF00"
        series = self.series
        dev_code = dev_code.upper()
        if series == "Q":
            if dev_code in ["X", "Y", "M", "L", "B"]:
                data = "001004010001" + str("{:*<2}".format(dev_code)) + str("{:0=6}".format(dev_points)) + str("{:0=4x}".format(access_points))
            else:
                data = "001004010000" + str("{:*<2}".format(dev_code)) + str("{:0=6}".format(dev_points)) + str("{:0=4x}".format(access_points))
        elif series == "R":
            if dev_code in ["X", "Y", "M", "L", "B"]:
                data = "04010003" + str("{:*<4}".format(dev_code)) + str("{:0=4}".format(dev_points)) + str("{:0=2x}".format(access_points))
            else:
                data = "04010002" + str("{:*<4}".format(dev_code)) + str("{:0=4}".format(dev_points)) + str("{:0=2x}".format(access_points))
        elif series == "F":
            if dev_code in ["X", "Y", "M", "L", "B"]:
                data = "000004010001" + str("{:*<2}".format(dev_code)) + str("{:0=6}".format(dev_points)) + str("{:0=4x}".format(access_points))
            else:
                data = "000004010000" + str("{:*<2}".format(dev_code)) + str("{:0=6}".format(dev_points)) + str("{:0=4x}".format(access_points))
        data_len = str("{:04X}".format(len(data)))
        send_data = HEAD + data_len + data
        # print(send_data)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((self.ipadress, self.port_no))
            s.settimeout(1.0)
            s.send(bytes(send_data, "ascii"))
            response = s.recv(4096).decode("ascii")
            # 確認のための出力3行
            # print("response:" + response)
            # print("end code:" + response[21:24])
            # print("announce data:" + response[24:])
            # print("受信部：" + response[14:])
            # print("応答データ長："+ str(int(response[14:18], 16)))
            # print("終了コード：" + response[18:22])
            # print("受信データ：" + response[22:])
            if response[18:22] == "0000":  # エラーコード0(エラーなし)
                return response[22:]
            else:
                return False

    def write(self, dev_code, dev_points, write_data):
        HEAD = "500000FF03FF00"
        series = self.series
        dev_code = dev_code.upper()
        if series == "Q":
            if dev_code in ["X", "Y", "M", "L", "B"]:
                data = "001014010001" + "{:*<2}".format(dev_code) + str("{:0=6}".format(dev_points)) + "{:0=4x}".format(len(str(write_data)))
            elif (len(write_data) % 4 == 0):
                data = "001014010000" + str("{:*<2}".format(dev_code)) + str("{:0=6}".format(dev_points)) + str("{:0=4x}".format(int(len(str(write_data))/4)))
            else:
                print("error:書き込みデータが不正です")
        elif series == "R":
            if dev_code in ["X", "Y", "M", "L", "B"]:
                data = "001014010003" + "{:*<4}".format(dev_code) + "{:0=6}".format(dev_points) + "{:0=4x}".format(len(str(write_data)))
            elif (len(write_data) % 4 == 0):
                data = "001014010002" + "{:*<4}".format(dev_code) + "{:0=6}".format(dev_points) + "{:0=4x}".format(len(str(write_data))/4)
            else:
                print("error:書き込みデータが不正です")
        elif series == "F":
            if dev_code in ["X", "Y", "M", "L", "B"]:
                data = "000014010001" + "{:*<2}".format(dev_code) + "{:0=6}".format(dev_points) + "{:0=4x}".format(len(str(write_data)))
            elif (len(write_data) % 4 == 0):
                data = "000014010000" + "{:*<2}".format(dev_code) + "{:0=6}".format(dev_points) + "{:0=4x}".format(len(str(write_data))/4)
            else:
                print("error:書き込みデータが不正です")
        data_len = str("{:04X}".format(len(data + str(write_data))))
        send_data = HEAD + data_len + data + str(write_data)
        # 確認のための出力
        print("送信データ：" + send_data)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((self.ipadress, self.port_no))
            s.settimeout(1.0)
            s.send(bytes(send_data, "ascii"))
            # response = s.recv(4096).decode("ascii")
