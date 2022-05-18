
import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import time
import datetime as dt
import argparse
import numpy as np


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.ok = False

        # timer
        self.timer = ElapsedTimeClock(self.window)

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(
            window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tk.Button(
            window, bg='blue', fg='white', text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT)

        # video control buttons

        self.btn_start = tk.Button(
            window, bg='blue', fg='white', text='Start', command=self.open_camera)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = tk.Button(
            window, bg='blue', fg='white', text='Stop', command=self.close_camera)
        self.btn_stop.pack(side=tk.LEFT)

        # hsv button
        self.btn_hsv = tk.Button(
            window, bg='blue', fg='white', text='HSV', command=self.hsv)
        self.btn_hsv.pack(side=tk.LEFT)

        # face detection btn
        self.btn_face = tk.Button(
            window, bg='blue', fg='white', text='Face Detection', command=self.faceDetect)
        self.btn_face.pack(side=tk.LEFT)

        # quit button
        self.btn_quit = tk.Button(
            window, bg='blue', fg='white', text='Quit', command=self.quit_app)
        self.btn_quit.pack(side=tk.LEFT)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("capture-"+time.strftime("%d-%m-%Y-%H-%M-%S") +
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def open_camera(self):
        self.ok = True
        self.timer.start()
        print("camera opened => Recording")

    def close_camera(self):
        self.ok = False
        self.timer.stop()
        print("camera closed => Not Recording")

    def quit_app(self):
        self.window.destroy()
        print('Quit application')

    def update(self):

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def hsv(self, video_source=0):
        cap = cv2.VideoCapture(self.video_source)
        while True:
            # Cap will read each frame
            _, frame = cap.read()
            # Convert frames to format call hsv
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Red color
            low_red = np.array([161, 155, 84])
            high_red = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv_frame, low_red, high_red)
            red = cv2.bitwise_and(frame, frame, mask=red_mask)
            # Blue color
            low_blue = np.array([94, 80, 2])
            high_blue = np.array([126, 255, 255])
            blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
            blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

            # Green color
            low_green = np.array([25, 52, 72])
            high_green = np.array([102, 255, 255])
            green_mask = cv2.inRange(hsv_frame, low_green, high_green)
            # Using bitwise_and to convert bitwise to rgb
            # np.bitwise_and(array1, array2, out)
            green = cv2.bitwise_and(frame, frame, mask=green_mask)

            # Every color except white
            low = np.array([0, 42, 0])
            high = np.array([179, 255, 255])
            mask = cv2.inRange(hsv_frame, low, high)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("Frame", frame)
            cv2.imshow("Red", red)
            cv2.imshow("Blue", blue)
            cv2.imshow("Green", green)
            cv2.imshow("Result", result)
            # Using to wait for a specific time interval and then close the active image window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def faceDetect(self, video_source=0):
        cap = cv2.VideoCapture(0)

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
            )

            print("Found {0} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detects eyes of different sizes in the input image
                eyes = eye_cascade.detectMultiScale(roi_gray)

                # To draw a rectangle in eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 127, 255), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
            # cap.release()
        cv2.destroyAllWindows()


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args = CommandLineParser().args

        # create videowriter

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc = VIDEO_TYPE[args.type[0]]

        # 2. Video Dimension
        STD_DIMENSIONS = {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res = STD_DIMENSIONS[args.res[0]]
        print(args.name, self.fourcc, res)
        self.out = cv2.VideoWriter(
            args.name[0]+'.'+args.type[0], self.fourcc, 10, res)

        # set video sourec width and height
        self.vid.set(3, res[0])
        self.vid.set(4, res[1])

        # Get video source width and height
        self.width, self.height = res

    # To get frames

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.out.release()
            cv2.destroyAllWindows()


class ElapsedTimeClock:
    def __init__(self, window):
        self.T = tk.Label(window, text='00:00:00', font=(
            'times', 20, 'bold'), bg='blue', fg='white')
        self.T.pack(fill=tk.BOTH, expand=1)
        self.elapsedTime = dt.datetime(1, 1, 1)
        self.running = 0
        self.lastTime = ''
        t = time.localtime()
        self.zeroTime = dt.timedelta(hours=t[3], minutes=t[4], seconds=t[5])
        # self.tick()

    def tick(self):
        # get the current local time from the PC
        self.now = dt.datetime(1, 1, 1).now()
        self.elapsedTime = self.now - self.zeroTime
        self.time2 = self.elapsedTime.strftime('%H:%M:%S')
        # if time string has changed, update it
        if self.time2 != self.lastTime:
            self.lastTime = self.time2
            self.T.config(text=self.time2)
        # calls itself every 200 milliseconds
        # to update the time display as needed
        # could use >200 ms, but display gets jerky
        self.updwin = self.T.after(100, self.tick)

    def start(self):
        if not self.running:
            self.zeroTime = dt.datetime(1, 1, 1).now()-self.elapsedTime
            self.tick()
            self.running = 1

    def stop(self):
        if self.running:
            self.T.after_cancel(self.updwin)
            self.elapsedTime = dt.datetime(1, 1, 1).now()-self.zeroTime
            self.time2 = self.elapsedTime
            self.running = 0


class CommandLineParser:

    def __init__(self):

        # Create object of the Argument Parser
        parser = argparse.ArgumentParser(description='Script to record videos')

        # Create a group for requirement
        # for now no required arguments
        # required_arguments=parser.add_argument_group('Required command line arguments')

        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=[
                            'avi'], type=str, help='Type of the video output: for now we have only AVI & MP4')

        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=[
                            '480p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')

        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument(
            '--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()


def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(), 'Project Python')


main()
