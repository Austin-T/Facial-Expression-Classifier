"""Here is the main class for the "Facial Expression Recognition" program. This class contains code which
controls video capture, face detection, facial expression analysis, and image display. After creating an
instance of the ExpressionAnalyst class, a call to the run() method will allow the program to execute as
designed.
"""

import cv2
import tensorflow
import numpy as np


class ExpressionAnalyst:

    def __init__(self):
        """Initializes an instance of the ExpressionAnalyst Class."""

        # Static class attributes
        self.ret = None
        self.frame = None
        self.gray_frame = None
        self.faces = None
        self.labels = []
        self.label_dict = {
            0: "Anger/Disgust",  # More anger than disgust
            1: "Anger/Disgust",  # More disgust than anger
            2: "Fear/Surprise",  # More fear than surprise
            3: "Happiness",  # Only Happiness
            4: "Sadness",  # Only Sadness
            5: "Fear/Surprise",  # More surprise than fear
            6: "Neutral"  # Only Neutral
        }

        # Adjustable class attributes
        self.scale_factor = 1.5  # Used for detection of faces with haarcascade object
        self.min_neighbors = 5  # Used for detection of faces with haarcascade object

        self.roi_border_width = 2
        self.roi_border_color = (255, 0, 0)  # BGR

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_size = 2
        self.font_color = (255, 255, 255)
        self.font_stroke = 2

        # Load haarcascade object for facial detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

        # Initialize video capture from the default video camera
        self.capture = cv2.VideoCapture(0)

        # Load deep learning model
        self.model = tensorflow.keras.models.model_from_json(open("model.json", "r").read())
        self.model.load_weights("model.h5")

    def run(self):
        """Executes video capture, face detection, facial expression analysis, and image display in sequence. Checks
        to see if the program should be terminated (happens when the user holds down the 'Q' key while the cursor
        is over the window).
        """

        running = True

        while running:
            self.capture_frame()
            self.detect_faces()
            self.detect_expressions()
            self.display_window()

            if self.check_close_clicked():
                running = False

    def capture_frame(self):
        """Extracts a frame from 'self.capture', the video capture from the default camera. Creates a grayscale
        version of the original frame.
        """
        
        # Extract a frame
        self.ret, self.frame = self.capture.read()

        # Create a grayscale image from the original
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        """Detects regions of interest (i.e. faces) using the Haarcascade object. Resets the labels
        for each region of interest.
        """

        # Detect the faces
        self.faces = self.face_cascade.detectMultiScale(self.gray_frame, self.scale_factor, self.min_neighbors)

        # Reset the face labels
        self.labels = []

    def detect_expressions(self):
        """Creates grayscale images that represent each region of interest in 'self.faces'. Formats
        images for processing by deep learning model. The deep learning model then predicts which
        facial expression is being emoted by the face. The prediction is added to 'self.labels'.
        """

        for (x, y, w, h) in self.faces:

            # Get a grayscale image of the face
            end_coord_x = x + w
            end_coord_y = y + h
            roi = self.gray_frame[y:end_coord_y, x:end_coord_x]

            # Format the grayscale image for processing with deep learning model
            roi = cv2.resize(roi, (48, 48))
            img_pixels = tensorflow.keras.preprocessing.image.img_to_array(roi)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # Get a prediction from the deep learning model
            prediction = self.model.predict(img_pixels)

            # Convert prediction to a user-readable string.
            max_index = np.argmax(prediction)
            label_string = self.label_dict[max_index]

            # Add the string to 'self.labels'
            self.labels.append(label_string)

    def display_window(self):
        """Displays a window which will contain the live video capture. This window will also display a
        rectangle around each face that is detected within it. The rectangle will have a label which shows
        the expression that is emoted by each face, as predicted by the deep learning model.
        """

        # Draw regions of interest
        for (x, y, w, h), label in zip(self.faces, self.labels):
            end_coord_x = x + w
            end_coord_y = y + h

            # Display roi rectangle
            cv2.rectangle(self.frame, (x, y), (end_coord_x, end_coord_y), self.roi_border_color, self.roi_border_width)

            # Display roi label
            cv2.putText(self.frame, label, (x, y), self.font, self.font_size, self.font_color, self.font_stroke, cv2.LINE_AA)

        # Display the colored frame (not grayscale)
        cv2.imshow('frame', self.frame)

    def check_close_clicked(self):
        """Checks to see if the 'Q' key has been pressed. The cursor must be over top of the window for this to
        be detected. If the key is pressed, terminate the cv2 window and live video capture.

        :return: True if 'q' has been pressed. False otherwise.
        """

        # Check for close-window command
        if cv2.waitKey(20) & 0xFF == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            return True
        else:
            return False
