import argparse
import os

import tensorflow as tf
import numpy as np

import smpl
import snug_utils as utils 
# from train import *

if __name__ == "__main__":
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion",
        type=str,
        default="assets/CMU/07/07_02_poses.npz",
        help="path of the motion to use as input"
    )

    parser.add_argument(
        "--garment",
        type=str,
        default="tshirt",
        help="name of the garment (tshirt, tank, top, pants or shorts)"
    )

    parser.add_argument(
        "--savedir",
        type=str,
        default="tmp",
        help="path to save the result"
    )

    args = parser.parse_args()
    
    # Load smpl
    body = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

    # Load garment model
    model_path, template_path = utils.get_model_path(args.garment)
    # snug = tf.saved_model.load(model_path)
    mysnug = tf.saved_model.load(model_path)
    _, f_garment = utils.load_obj(template_path)
   
    # # Load motion   
    # poses, trans, trans_vel = utils.load_motion(args.motion)
    # # Body shape 
    # betas = np.zeros(10, dtype=np.float32)

    poses, trans, trans_vel, betas = utils.load_motion_train_With_betas(args.motion)

    # Run model
    hidden_states = [
        tf.zeros((1, 256), dtype=tf.float32), # State 0
        tf.zeros((1, 256), dtype=tf.float32), # State 1
        tf.zeros((1, 256), dtype=tf.float32), # State 2
        tf.zeros((1, 256), dtype=tf.float32), # State 3
    ]

    # print(len(poses)) 688

    for frame in range(len(poses)):
        pose = tf.reshape(poses[frame], [1, 1, 72])
        shape = tf.reshape(betas, (1, 1, 10))
        translation = tf.reshape(trans[frame], (1, 1, 3))
        translation_vel = tf.reshape(trans_vel[frame], (1, 1, 3))

        # Eval body
        v_body, tensor_dict = body(
            shape=tf.reshape(shape, [-1, 10]), 
            pose=tf.reshape(pose, [-1, 72]),
            translation=tf.reshape(translation, [-1, 3]),
        )

        # Eval SNUG
        shape = tf.cast(shape, tf.float32)
        v_garment, hidden_states = mysnug([ 
            pose,
            translation_vel,
            shape,

            # State of the recurrent hidden layers
            hidden_states[0],
            hidden_states[1],
            hidden_states[2],
            hidden_states[3],

            # Additional inputs for LBS and collision postprocess
            translation,
            tensor_dict["shape_blendshape"],
            tensor_dict["joint_transforms"],
            tensor_dict["vertices"],
            tensor_dict["vertex_normals"]
        ])
        
        # pose_ = tf.reshape(poses[frame:frame+3], [1, 3, 72])
        # # shape_ = tf.reshape(betas, (1, 1, 10)).astype(float32)
        # shape_ = tf.cast(shape, tf.float32)
        # translation_ = tf.reshape(trans[frame:frame+3], (1, 3, 3))
        # translation_vel_ = tf.reshape(trans_vel[frame:frame+3], (1, 3, 3))

        # v_garment_skinning0, v_garment_skinning1, v_garment_skinning2, v_body2, tensor_dict2 = mysnug([ 
        #     # [betas, poses, trans, trans_vel, hidden_states[0], hidden_states[1],\
        #     #                  hidden_states[2], hidden_states[3]]
        #     shape_,
        #     pose_,
        #     translation_,
        #     translation_vel_,

        #     # State of the recurrent hidden layers
        #     hidden_states[0],
        #     hidden_states[1],
        #     hidden_states[2],
        #     hidden_states[3],

        #     # Additional inputs for LBS and collision postprocess
        #     # tensor_dict["shape_blendshape"],
        #     # tensor_dict["joint_transforms"],
        #     # tensor_dict["vertices"],
        #     # tensor_dict["vertex_normals"]
        # ])

        # v_garment = v_garment_skinning0
        
        # hidden_states

        body_path = os.path.join(args.savedir, f"{frame:04d}_body.obj")
        utils.save_obj(body_path, v_body, body.faces)

        garment_path = os.path.join(args.savedir, f"{frame:04d}_{args.garment}.obj")
        utils.save_obj(garment_path, v_garment, f_garment)
