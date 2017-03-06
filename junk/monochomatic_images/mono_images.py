import sys
import getopt
import csv
import os
import re
import numpy as np
import random
import math
import time
from sklearn.cluster import MiniBatchKMeans

sys.path.append('/usr/local/apps/opencv/opencv-2.4.11/lib/python2.7/site-packages/')

import cv2


def get_clusters(image, cluster_num):
	kmeans = MiniBatchKMeans(n_clusters=cluster_num, max_iter=2000)

	print(image.shape)

	pixels = image.reshape((int(image.shape[0] * image.shape[1]), 3)).astype("float64") / 255.0
	
	np.random.shuffle(pixels)
	
	kmeans.fit(pixels)
	
	return kmeans


def color_monochromatic(image, kmeans):
	
	out_im = np.zeros(shape=image.shape, dtype="uint8")

	colors = [30, 90, 150]

	kmeans_im = image.reshape((int(image.shape[0] * image.shape[1]), 3)).astype("float64") / 255.0

	labels = kmeans.predict(kmeans_im)

	labels = labels.reshape((image.shape[0], image.shape[1]))

	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			
			out_im[y, x, :] = colors[labels[y, x]]

	return out_im


def main(argv):
	
	opts, args = getopt.getopt(argv, "i:o:", ["input_image=", "out="])

	out_name = ""
	in_name = ""

	for opt, arg in opts:
		if opt in ("--input_image", "-i"):
			print("here")
			in_name = arg
		elif opt in ("--out", "-o"):
			out_name = arg
	print("In name")
	print(in_name)
	image = cv2.imread(in_name)

	kmeans_cluster = get_clusters(image, 3)

	out_image = color_monochromatic(image, kmeans_cluster)

	cv2.imwrite(out_name, out_image)

if __name__ == "__main__":
	main(sys.argv[1:])
