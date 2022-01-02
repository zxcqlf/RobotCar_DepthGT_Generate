################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
from collections import Counter
import cv2

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, default='~SEQ~/2014-12-09-13-21-02_stereo_left_04/2014-12-09-13-21-02/stereo/left/',help='Directory containing images')
parser.add_argument('--min_depth',default=1e-3, type=float, help='min depth 1e-3')
parser.add_argument('--max_depth',default=60, type=float, help='max_depth 70')
parser.add_argument('--laser_dir', type=str, default='~SEQ~/Front-2D lidar/2014-12-09-13-21-02_lms_front_04/2014-12-09-13-21-02/lms_front/',help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, default='./vo.csv',help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, default='camera-models/',help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str,default='extrinsics/' ,help='Directory containing sensor extrinsics')
parser.add_argument('--predicted_disp_path', type=str,default='./encoder.npy' , help='predicted_disparity .npy file path')
parser.add_argument('--test_file_time_stamps',default='./tset_splits/test_files.txt' ,type=str,help='test_images timestamps')
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')


args = parser.parse_args()

'''def load_predictions(pred_file,image_shape):

	predicted_disp=np.load(args.predicted_disp_path)

	print image_shape

	disp_all=[]

	for disp in predicted_disp:
		
		re_disp=cv2.resize(disp,(image_shape[1],image_shape[0]), interpolation=cv2.INTER_LINEAR)


		#plt.imshow(re_disp,cmap='gray')

		#plt.show()

		re_disp*=re_disp.shape[1]

		re_disp=(964.82*0.239)/re_disp

		disp_all.append(re_disp)

	return disp_all'''
def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = float(min_disp) + float(max_disp - min_disp) * disp
    depth = float(1.) / scaled_disp
    return depth

def load_predictions(pred_file,image_shape):

	predicted_disp=np.load(args.predicted_disp_path)

	print (image_shape)

	disp_all=[]

	for disp in predicted_disp:
		
		re_disp=cv2.resize(disp,(image_shape[1],image_shape[0]), interpolation=cv2.INTER_LINEAR)


		#plt.imshow(re_disp,cmap='gray')

		#plt.show()

		#re_disp*=re_disp.shape[1]

		#re_disp=(983.044006*0.239)/re_disp

		#re_disp=(1.)/re_disp

		disp_all.append(disp_to_depth(re_disp,0.1,100.0))

	return disp_all

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__=='__main__':

	model = CameraModel(args.models_dir, args.image_dir)

	extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')

	with open(extrinsics_path) as extrinsics_file:

		extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

	G_camera_vehicle = build_se3_transform(extrinsics)
	G_camera_posesource = None

	poses_type =re.search('(vo|ins)\.csv', args.poses_file).group(1)

	if poses_type == 'ins':
		with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
		    extrinsics = next(extrinsics_file)
		    G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
	else:
		# VO frame and vehicleS frame are the same
		G_camera_posesource = G_camera_vehicle


	timestamps_path = args.test_file_time_stamps

	timestamp = 0

	init_flag=True

	depth_all=[]

	timestamp_list=[1418757429014730]

	with open(timestamps_path) as timestamps_file:

		for line in (timestamps_file):

			print (line[:-5])

			timestamp=int(line[:-5])

			if not timestamp in timestamp_list:

				pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file, args.extrinsics_dir,
				                           timestamp - 1e7, timestamp + 1e7, timestamp)

				pointcloud = np.dot(G_camera_posesource, pointcloud)

				if init_flag:

					image_path = os.path.join(args.image_dir, str(timestamp) + '.png')

					image = load_image(image_path, model)

					im_shape=image.shape

					init_flag=False

				velo_pts_im, velo_depth = model.project(pointcloud, image.shape)

				velo_pts_im=velo_pts_im.T


				velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) -1
				velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) -1

				depth = np.zeros((im_shape[0],im_shape[1]))

				depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = np.expand_dims(velo_depth,1)

				depth[depth<0] = 0

				#plt.figure(1)

				#plt.imshow(depth)

				#plt.show()

				crop_height=int(im_shape[0]*0.75)

				depth=depth[:crop_height,:]

				depth_all.append(depth)
			else:

				depth_all.append([])
	
		np.save('GT/test_DAY3D_disps.npy',depth_all)
	

	disp_all=load_predictions(args.predicted_disp_path,(crop_height,im_shape[1]))

	num_samples=len(depth_all)

	print (num_samples)

	rms     = np.zeros(num_samples, np.float32)
	log_rms = np.zeros(num_samples, np.float32)
	abs_rel = np.zeros(num_samples, np.float32)
	sq_rel  = np.zeros(num_samples, np.float32)
	d1_all  = np.zeros(num_samples, np.float32)
	a1      = np.zeros(num_samples, np.float32)
	a2      = np.zeros(num_samples, np.float32)
	a3      = np.zeros(num_samples, np.float32)
    
	for i in range(num_samples):

		if len(depth_all[i])>0:

			gt_depth = depth_all[i]


			pred_depth = disp_all[i]

			pred_depth[pred_depth < args.min_depth] = args.min_depth
			pred_depth[pred_depth > args.max_depth] = args.max_depth


			mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

			scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])

			pred_depth[mask] *= scalor

			 
			#gt_height, gt_width = gt_depth.shape

			if len(gt_depth[mask])>1:	
				abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
			else:

				print ('skipping',i)

	print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
	print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

	
