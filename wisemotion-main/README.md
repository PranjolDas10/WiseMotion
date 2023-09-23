# Wisemotion
An all in one platform designed for safe driving

# About
This code analyzes the posture of a sitting driver to determine whether or not they are sitting ergonomically.
Reminders are given to those driving in bad postures.

The process is completed through the use of a local webcam. Distinctive points (ex. hips, shoulder) are identified and then angles are calculated between such points. Using the RULA (Rapid Upper Limb Assessment) developed by ergonomist Dr. Lynn McAtamney and Professor E. Nigel Corlett from the University of Nottingham, the analysis reaches high accuracy and precision.

This model mainly relies on Google’s machine learning models and MediaPipe

The other main aspect of this code analyzes a person’s face and eyes to determine whether they are drowsy or not.
Verbal reminders/prompts are then sent to the driver if they are too drowsy to be driving.

This process also uses a local webcam. The program uses the Haar Cascade classifiers to identify faces and then identifies the eyes within the face region. The Dlib toolkit is then used to locate facial landmarks within each eye region. This program mainly uses pre-trained models from Dlib for facial mapping. An Ear Aspect Ratio (EAR) is then calculated for each eye and an average of both eyes is used to calculate the overall drowsiness of a driver.

# Setup
part 1 with anaconda:

conda create -n ergonomy python=3.7

conda activate ergonomy

python -m pip install mediapipe


part 2 with pip

pip install opencv-python

pip install dlib

pip install numpy

pip install playsound

python -m pip install scipy
