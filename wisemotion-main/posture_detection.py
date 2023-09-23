"""
This code will determine whether or not a driver in a car is driving with correct posture
"""

#import necessary packages
import cv2
import mediapipe as mp
import numpy as np
import time

#pose estimation on the camera
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

state = [0]

class Ergonomy:
	def __init__(self):
		self.trunk_angle=0

	def update_joints(self, landmarks_3d):
		try:
			#all the marked joints
			left_shoulder = np.array([landmarks_3d.landmark[11].x, landmarks_3d.landmark[11].y, landmarks_3d.landmark[11].z])
			right_shoulder = np.array([landmarks_3d.landmark[12].x, landmarks_3d.landmark[12].y, landmarks_3d.landmark[12].z])
			left_hip = np.array([landmarks_3d.landmark[23].x, landmarks_3d.landmark[23].y, landmarks_3d.landmark[23].z])
			right_hip = np.array([landmarks_3d.landmark[24].x, landmarks_3d.landmark[24].y, landmarks_3d.landmark[24].z])
			left_knee = np.array([landmarks_3d.landmark[25].x, landmarks_3d.landmark[25].y, landmarks_3d.landmark[25].z])
			right_knee = np.array([landmarks_3d.landmark[26].x, landmarks_3d.landmark[26].y, landmarks_3d.landmark[26].z])
			
			#joints to measure
			mid_shoulder = (left_shoulder + right_shoulder) / 2
			mid_hip = (left_hip + right_hip) / 2
			mid_knee = (left_knee + right_knee) / 2

			#angles
			self.trunk_angle = self.get_angle(mid_knee, mid_hip, mid_shoulder, mid_hip, adjust=True)

		except:
			# could not retrieve all needed joints
			pass
	
	#this gets the angles between the body parts
	def get_angle(self, a, b, c, d, adjust):
		#returns angle between two vectors 
		vec1 = a - b
		vec2 = c - d

		cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
		angle = np.arccos(cosine_angle)
		#this returns the angles
		return int(abs(np.degrees(angle)))

	#this function determines what colour to show on screen based on posture angles 
	def get_trunk_color(self, state):
		#returns (B, G, R) colours for visualization
		#angles greater than 75 and less than 125 turn green
		if self.trunk_angle >= 75 and self.trunk_angle <= 125: 
			state.append(0)
			return (0, 255, 0)
		#everything else is a bad posture 
		else:
			if state[-1] != 1: 
				print("Your posture is bad!")
				time.sleep(1)
				state.append(1)
			return(0, 0, 255)
	
if __name__ == '__main__':
	MyErgonomy = Ergonomy()
	#this starts capturing a video from the webcam
	cap = cv2.VideoCapture(0)
	#this starts posing the landmarks 
	with mp_pose.Pose(
		model_complexity=1,
		smooth_landmarks=True,
		min_detection_confidence=0.3,
		min_tracking_confidence=0.3) as pose:
		while cap.isOpened():
			success, image = cap.read()
			if not success:
				continue

			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			results = pose.process(image)
			landmarks_3d = results.pose_world_landmarks

			#this draws the pose annotation on the image
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			mp_drawing.draw_landmarks(
				image,
				results.pose_landmarks,
				mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

			#this updates everything with the joints locations
			MyErgonomy.update_joints(landmarks_3d)

			#this displays whether or not a posture is good
			image = cv2.putText(image, text = "Posture:"+str(MyErgonomy.trunk_angle), 
				org=(5,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=MyErgonomy.get_trunk_color(state), thickness=3)
			image = cv2.rectangle(image, (5,5), (145*2, 30), color=(255,255,255), thickness=-1)
			image = cv2.rectangle(image, (5,5), (145*2-(MyErgonomy.trunk_angle * 2), 30), color=MyErgonomy.get_trunk_color(state), thickness=-1)

			#this creates a window with the camera named Pose Detector 
			cv2.imshow('Pose Detector', image)

			if cv2.waitKey(5) & 0xFF == 27:
				break
	cap.release()