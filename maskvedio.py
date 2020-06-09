import os
import shutil
import cv2
face_cascade = \
    cv2.CascadeClassifier('./haarcascades/cascade.xml')

vid = cv2.VideoCapture("./maskwith.mp4")
# vid = cv2.VideoCapture(0)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vid.get(cv2.CAP_PROP_FPS)
print("width %d,height %d" % (width, height))
print("fps %f" % (fps))
grabbed, frame = vid.read()

if os.path.exists("./output1"):
    shutil.rmtree("./output1")
os.mkdir("./output1")
fourcc = cv2.VideoWriter.fourcc(*"XVID")
videowriter = cv2.VideoWriter("./output1.avi", fourcc,
                              fps,
                              (int(width), int(height)))

while grabbed:
    frame = cv2.medianBlur(frame, 5)

    frame = cv2.addWeighted(frame, 0.7, frame, 0.3, 0)  # 9_1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 22

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 22

    for (x, y, w, h) in faces:  # 22
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cv2.putText(frame, 'Detected Face:', (top_left[0], top_left[1] - 10),
        #             cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, 1)

    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    videowriter.write(frame)

    grabbed, frame = vid.read()

vid.release()
videowriter.release()
