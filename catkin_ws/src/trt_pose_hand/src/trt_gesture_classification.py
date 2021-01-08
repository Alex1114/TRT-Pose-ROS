#!/usr/bin/env python3

import os 
import cv2
import math
import time
import json
import numpy as np
import operator
import pickle 
# import traitlets
import PIL.Image
import trt_pose.coco
import warnings
warnings.filterwarnings("ignore")

# ROS
import rospy
import roslib
import rospkg
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

# Torch
import torch
import torch2trt
from torch2trt import TRTModule
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# trt_model
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from preprocessdata import preprocessdata
from gesture_classifier import gesture_classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class trt_gesture_classification(object):
	def __init__(self):
		self.bridge = CvBridge()

		r = rospkg.RosPack()
		self.path = r.get_path("trt_pose_hand")

		# Json path
		with open(os.path.join(self.path, "preprocess/hand_pose.json"), "r") as f:
			self.hand_pose = json.load(f)

		self.topology = trt_pose.coco.coco_category_to_topology(self.hand_pose)

		self.num_parts = len(self.hand_pose['keypoints'])
		self.num_links = len(self.hand_pose['skeleton'])

		self.model = trt_pose.models.resnet18_baseline_att(self.num_parts, 2 * self.num_links).cuda().eval()

		self.WIDTH = 224
		self.HEIGHT = 224
		self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()

		# Model path
		if not os.path.exists(os.path.join(self.path, "model/hand_pose_resnet18_att_244_244_trt.pth")):
			MODEL_WEIGHTS = 'model/hand_pose_resnet18_att_244_244.pth'
			self.model.load_state_dict(torch.load(os.path.join(self.path, MODEL_WEIGHTS)))
			self.model_trt = torch2trt.torch2trt(self.model, [self.data], fp16_mode=True, max_workspace_size=1<<25)
			OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'
			torch.save(self.model_trt.state_dict(), os.path.join(self.path, OPTIMIZED_MODEL))

		OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'

		self.model_trt = TRTModule()
		self.model_trt.load_state_dict(torch.load(os.path.join(self.path, OPTIMIZED_MODEL)))

		self.parse_objects = ParseObjects(self.topology,cmap_threshold=0.15, link_threshold=0.15)
		self.draw_objects = DrawObjects(self.topology)

		self.preprocessdata = preprocessdata(self.topology, self.num_parts)

		self.gesture_classifier = gesture_classifier()

		# Gesture

		self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))

		filename = 'svmmodel.sav'
		self.clf = pickle.load(open(os.path.join(self.path, filename), 'rb'))

		with open(os.path.join(self.path, "preprocess/gesture.json"), 'r') as f:
			self.gesture = json.load(f)
		self.gesture_type = self.gesture["classes"]

		# Publisher
		self.predict_images = rospy.Publisher("trt_gesture_classification/prediction_image", Image, queue_size = 1)

		# Mssage filter 
		image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
		depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 1)
		ts.registerCallback(self.callback)

	def callback(self, rgb, depth):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
			cv_depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
		except CvBridgeError as e:
			print(e)
		
		cv_image = cv2.resize(np.array(cv_image), (224, 224))
		data = self.preprocess(cv_image)
		cmap, paf = self.model_trt(data)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		counts, objects, peaks = self.parse_objects(cmap, paf)
		joints = self.preprocessdata.joints_inference(cv_image, counts, objects, peaks)
		self.draw_joints(cv_image, joints)
		# self.draw_objects(cv_image, counts, objects, peaks)# try this for multiple hand pose prediction 
		
		dist_bn_joints = self.preprocessdata.find_distance(joints)
		gesture = self.clf.predict([dist_bn_joints,[0]*self.num_parts*self.num_parts])
		gesture_joints = gesture[0]
		self.preprocessdata.prev_queue.append(gesture_joints)
		self.preprocessdata.prev_queue.pop(0)
		self.preprocessdata.print_label(cv_image, self.preprocessdata.prev_queue, self.gesture_type)

		self.predict_images.publish(self.bridge.cv2_to_imgmsg(cv_image, "8UC3"))

	def draw_joints(self, image, joints):
		count = 0
		for i in joints:
			if i==[0,0]:
				count+=1
		if count>= 3:
			return 
		for i in joints:
			cv2.circle(image, (i[0],i[1]), 2, (0,0,255), 1)
		cv2.circle(image, (joints[0][0],joints[0][1]), 2, (255,0,255), 1)
		for i in self.hand_pose['skeleton']:
			if joints[i[0]-1][0]==0 or joints[i[1]-1][0] == 0:
				break
			cv2.line(image, (joints[i[0]-1][0],joints[i[0]-1][1]), (joints[i[1]-1][0],joints[i[1]-1][1]), (0,255,0), 1)

	def preprocess(self, image):
		mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
		std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
		global device
		device = torch.device('cuda')
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = PIL.Image.fromarray(image)
		image = transforms.functional.to_tensor(image).to(device)
		image.sub_(mean[:, None, None]).div_(std[:, None, None])
		return image[None, ...]


	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	
if __name__ == '__main__': 
	rospy.init_node('trt_gesture_classification',anonymous=False)
	trt_gesture_classification = trt_gesture_classification()
	rospy.on_shutdown(trt_gesture_classification.onShutdown)
	rospy.spin()
