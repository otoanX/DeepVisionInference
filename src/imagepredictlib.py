# import cv2


# def serchcam():
#     SEARCHCAMNUM = 20
#     for i1 in range(0, SEARCHCAMNUM): 
#         cap1 = cv2.VideoCapture( i1, cv2.CAP_DSHOW )
#         if cap1.isOpened():
#             print("VideoCapture(", i1, ") : Found")
#         else:
#             print("VideoCapture(", i1, ") : None")
#         cap1.release()
#     return 

# if __name__ == "__main__":
#     serchcam()

# import subprocess
# import re

# output = (
#     subprocess.check_output(["powershell.exe", 'Get-PnpDevice -Class "Image"'])
#     .decode("utf-8")
# )

# camera_infos = output.splitlines()[3:-2]

# cameras = [re.split(r"\s{2,}", camera)[2] for camera in camera_infos]
# print(cameras)


import cv2


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
