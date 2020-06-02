# Facial-Expression-Classifier
This is a repository for real-time detection of facial expressions, using OpenCV and Keras.

As your computer's default camera captures video, this program scans each frame to detect faces. The detected faces are then passed to a 
deep learning model, which categorizes the facial expressions into one of five categories: Happiness, Sadness, Neutral, Anger/Disgust, and 
Fear/Surprise. The program then displays a window containing the live video capture. Within the window, each face is outlined by a blue 
rectangle, and a label above the rectangle indicates the expression that is being emoted by the face.

In training, the deep learning model has functioned with up to 91.7% accuracy. However, results will vary during real-time detection. For 
the best results, a face should be fully illuminated from both sides, and a high resolution web-camera should be used for capturing video.

All code is avalable under 'GNU General Public License v3.0'. Enjoy!

Austin
