import os
import sys
from random import shuffle, choice
from math import floor
from scipy import sparse

from time import time
import tensorflow as tf
import numpy as np
import smpl
import snug_utils as utils


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')


class Data:
    def __init__(self,  batch_size=10, mode='train'):
        """
        Args:
        - poses: path to .npy file with poses
        - shape: SMPL shape parameters for the subject
        - gender: 0 = female, 1 = male
        - batch_size: batch size
        - shuffle: shuffle
        """
        # Read sample list
        self._pose_path = "assets/CMU"
        # self._poses, self._trans, self._trans_vel, self._shape = self._get_pose()
        self._poses, self._trans, self._trans_vel = self._get_pose()
        if self._poses.dtype == np.float64: self._poses = np.float32(self._poses)
        self._n_samples = self._poses.shape[0]
        # smpl
        self.SMPL = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

        # self._shape = np.zeros(10, dtype=np.float32)
        # TF Dataset
        ds = tf.data.Dataset.from_tensor_slices((self._poses,self._trans,self._trans_vel))
        if mode == 'train': ds = ds.shuffle(self._n_samples)
        # ds = ds.map(self.tf_map, num_parallel_calls=batch_size)
        ds = ds.batch(batch_size=batch_size)
        self._iterator = ds
        # print("sys.path", sys.path) # 17299's PATH in windows

    def _get_pose(self):
        #print("_get_pose CALLED") #called
        Batch_size = 2

        poses_array = np.zeros((1,72))
        trans_array = np.zeros((1,3))
        trans_vel_array = np.zeros((1,3))
        # betas_array = np.zeros((10,))
        betas = np.zeros((10,))

        folder_num = 10
        file_num = 100
        totfile_num = 2
        k_folder = 0
        k_file = 0
        k_totfile = 0

        for folder in os.listdir(self._pose_path):
            k_folder+=1
            k_file = 0
            # if folder == '07':
            for file in os.listdir(os.path.join(self._pose_path,folder)):
                
                file_path = os.path.join(self._pose_path, folder, file)
                # print(file_path, folder)
                # if file == '07_02_poses.npz':
                poses, trans, trans_vel, betas = utils.load_motion_train_With_betas(file_path)
                print("file {}-{}".format(k_file, poses.shape[0]), file)
                # print("")
                # print(file_path)
                # print(poses.shape[0])
                # print(trans.dtype, trans.shape)
                # print(trans_vel.dtype, trans_vel.shape)
                # print(betas.dtype, betas.shape)
                # assets/CMUb
                # float32 (688, 72)
                # float32 (688, 3)
                # float32 (688, 3)
                # float64 (10,)
                # print("")
                poses_array = np.concatenate((poses_array,poses),axis=0)
                trans_array = np.concatenate((trans_array, trans), axis=0)
                trans_vel_array = np.concatenate((trans_vel_array, trans_vel), axis=0)
                # betas_array = np.concatenate((betas_array, betas), axis=0)
                k_file += 1
                k_totfile += 1
                if k_file>=file_num:
                    break
                if k_totfile >= totfile_num:
                    break

            if k_folder >=folder_num:
                break
            if k_totfile >= totfile_num:
                break

        #delete zeros
        poses_array = np.delete(poses_array,0,axis=0)
        trans_array = np.delete(trans_array,0,axis=0)
        trans_vel_array = np.delete(trans_vel_array,0,axis=0)
        # betas_array = np.delete(betas_array,0,axis=0)
        #去掉最后几帧
        remainder = poses_array.shape[0]%3
        poses_array = np.delete(poses_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]),axis=0)
        trans_array = np.delete(trans_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
        trans_vel_array = np.delete(trans_vel_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
                
        poses_array = poses_array.reshape((-1,3,poses_array.shape[-1]))        
        trans_array = trans_array.reshape((-1, 3, trans_array.shape[-1]))        
        trans_vel_array = trans_vel_array.reshape((-1, 3, trans_vel_array.shape[-1]))
        
        #还得去掉最后几帧
        remainder = poses_array.shape[0]%2
        poses_array = np.delete(poses_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]),axis=0)
        trans_array = np.delete(trans_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
        trans_vel_array = np.delete(trans_vel_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)


        # print("--------------------------")
        # print(poses_array.shape, trans_array.shape, trans_vel_array.shape)
        # #(7089, 3, 72) (7089, 3, 3) (7089, 3, 3)
        # print("--------------------------")
        #(229, 3, 72) (229, 3, 3) (229, 3, 3)
        print("tot", poses_array.shape[0]/2)
        # return poses_array, trans_array, trans_vel_array, betas
        return poses_array, trans_array, trans_vel_array



    def _next(self, pose):
        # print("_next CALLED")

        # compute body
        # while computing SMPL should be part of PBNS,
        # if it is in Data, it can be efficiently parallelized without overloading GPU

        G, B = self.SMPL.set_params(pose=pose.numpy(), beta=self._shape, with_body=True)

        return pose, G, B

    def tf_map(self, pose):
        # print("tf_map CALLED")
        return tf.py_function(func=self._next, inp=[pose], Tout=[tf.float32, tf.float32, tf.float32])