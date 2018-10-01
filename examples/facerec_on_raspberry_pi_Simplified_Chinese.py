# 这是一个在树莓派上运行人脸识别的案例
# 本案例会在命令行控制面板上输出识别出的人脸数量和身份结果。

# 你需要一个2代以上的树莓派，并在树莓派上安装face_recognition，并连接上picamera摄像头
# 并确保picamera这个模块已经安装（树莓派一般会内置安装）
# 你可以参考这个教程配制你的树莓派：
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import picamera
import numpy as np

# 你需要在sudo raspi-config中把camera功能打开
camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)

# 载入样本图片（奥巴马和拜登）
print("Loading known face image(s)")
obama_image = face_recognition.load_image_file("obama_small.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# 初始化变量
face_locations = []
face_encodings = []

while True:
    print("Capturing image.")
    # 以numpy array的数据结构从picamera摄像头中获取一帧图片
    camera.capture(output, format="rgb")

    # 获得所有人脸的位置以及它们的编码
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)

    # 将每一个人脸与已知样本图片比对
    for face_encoding in face_encodings:
        # 看是否属于奥巴马或者拜登
        match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
        name = "<Unknown Person>"

        if match[0]:
            name = "Barack Obama"

        print("I see someone named {}!".format(name))
