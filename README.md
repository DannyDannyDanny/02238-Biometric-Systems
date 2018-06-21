# Face Quality Enhancement (FQE)
> 02238-Biometric-Systems

# Basically
The file `./pythonmess/seven.py` is the final implementation and spits out the file called `./pythonmess/seven.csv`.

Read the entire report here:

# Research papers
The scope of this project report is:
* Analysis of a set of face enhancements and their influence on comparison scores
* Graphs showing the effect of various enhancements on the biometric performance

Facial detection
With the advances and spread of both Surveillance cameras and Face Detection technology, this research paper aims to analyze the limitations of combining these technologies for mass surveillance. A preliminary investigation on the overall impact of sample quality on detection ability is tested by applying OpenCV feature detection to a live camera stream.

The 720p front-facing camera of a laptop revealed that even when the subject is centered and consumes a majority portion of the screen, OpenCV feature detection quality depends primarily on direct or neutral lighting and prefers a frontal-full-face at camera level with no forward/backward and side-to-side tilt.

In the case of a security camera watching over a stream of pedestrians, a subject may have an easily detectable head posture in good lighting for a single frame or two.

This report investigates to what extent lighting, face angle and distance (simulated through resampling) affect image quality.

# Data
The MIT-CBCL face recognition database consists of 3240 images synthesized from high definition 3D models of 10 individuals heads with varying lighting angles and differing-facing-angles. Both of which are parameters that pose a challenge OpenCV's facial detection system.

Due to the synthesized nature of the images in the database some images show signs of digital aliasing. Furthermore, the parameter range for light and face angle cause some faces to be unrecognizable by all classifiers even at original resolution.
[Link to page](http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html)

* 0002_-32_0_0_75_0_1.pgm
* 0009_0_0_0_0_75_45_1.pgm
* 0008_-32_0_0_0_90_30_1.pgm
* 0006_-32_0_0_0_75_0_1.pgm


## Detection Method
OpenCV's python library `cv2`, performs face detection. The process involves analyzing differences between neighboring regions of in an image, to find certain features. The features may vary in complexity from diagonal and vertical edges to human faces and number plates.

The features are detected using *Haar feature-based Cascade Classifiers* which are classifier models pre-trained to specific features using machine learning and large sets of images.

The OpenCV library contains a range of Haar cascades ranging for detecting features like Russian license plates, cat faces and sub-features of human faces. To test how the image quality effects detection accuracy, several cascades are chosen - but not too many to keep computational time reasonable.

[OpenCV](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)

### Choosing OpenCV Cascades
The initial scanning yields the following OpenCV cascades as the top performers:
* 2928:haarcascade_frontalface_alt.xml
* 3221:haarcascade_frontalface_alt2.xml
* 2887:haarcascade_frontalface_default.xml
* 1813:haarcascade_profileface.xml

Several feature cascades are omitted due to irregular performance:
* *haarcascade_smile* yields high false negative rates occasionally misclassifying a thin eyebrow as a smile
* *lefteye_2splits* and *righteye_2splits* are asymmetric features and underperform on slightly side-facing images or uneven lighting.

## Optimization
https://medium.com/machine-learning-world/how-to-replicate-lets-enhance-service-without-even-coding-3b6b31b0fa2e

# Notes to self
Use [Open-CV](https://github.com/pypa/pipenv/issues/1313#issuecomment-358188962) if installing with pipenv
