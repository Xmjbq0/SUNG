import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import GRU,Dense
from tensorflow.python.keras import Sequential
import numpy as np
import math
import smpl
import utils as utils2
import snug_utils as utils
import argparse
from outfit import Outfit
from cloth import * 
from physics import *
from material import *

import Data
import time
import gc
import psutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
          .format(total, used, free))
    return total, used, free

# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--motion",
#     type=str,
#     default="assets/CMU/07_02_poses.npz",#改了
#     help="path of the motion to use as input"
# )

# parser.add_argument(
#     "--garment",
#     type=str,
#     default="tshirt",
#     help="name of the garment (tshirt, tank, top, pants or shorts)"
# )

# parser.add_argument(
#     "--savedir",
#     type=str,
#     default="tmp",
#     help="path to save the result"
# )

# args = parser.parse_args()

# Load smpl
body = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
v_Tbody = body.template_vertices
# Load garment model
# model_path, template_path = utils.get_model_path(args.garment)
# print(model_path)
# print(template_path)
# models/SNUG-Tshirt
# assets/meshes/tshirt.obj

# snug = tf.saved_model.load(model_path)

# Fabric material parameters
thickness = 0.00047 # (m)
bulk_density = 426  # (kg / m3)
area_density = thickness * bulk_density

material = Material(
    density=area_density, # Fabric density (kg / m2)
    thickness=thickness,  # Fabric thickness (m)
    young_modulus=0.7e5, 
    poisson_ratio=0.485,
    stretch_multiplier=1,
    bending_multiplier=50
)

print(f"Lame mu {material.lame_mu:.2E}, Lame lambda: {material.lame_lambda:.2E}")

# Initialize structs
cloth = Cloth(
    path="assets/meshes/tshirt.obj",
    material=material,
    dtype=tf.float32
)


# cloth = Cloth(path = template_path, material = Material(2.36e4, 4.44e4, 426,))
v_Tgarment = cloth.v_template
f_garment = cloth.f
# print(v_Tgarment.shape, f_garment.shape)
# v_Tgarment_tmp, f_garment_tmp = utils.load_obj(template_path)
# print(v_Tgarment.shape, f_garment.shape)
# (4424, 3) (8710, 3)


# closest_vertices = find_nearest_neighbour(v_Tgarment, v_Tbody)
# print("closest_vertices",closest_vertices)
# D
# (4424, 3) (6890, 3)
# print("D")
# print(v_Tgarment.shape, v_Tbody.shape)
closest_vertices = tf.convert_to_tensor(utils2.find_nearest_neighbour(v_Tgarment, v_Tbody), dtype = tf.int64)
print("closest_vertices",closest_vertices)

# #计算服装蒙皮权重
# # garment_skinning_weights = np.zeros((v_Tgarment.shape[0],24))
garment_skinning_weights = tf.gather(body.skinning_weights, indices=closest_vertices, axis=0)

class Deformation_model(tf.keras.Model):
    def __init__(self, output):
        super(Deformation_model, self).__init__()
        self.v_TGRAMENT = v_Tgarment
        self.g_skinweight = garment_skinning_weights
        self.gru1 = GRU(256, return_state=True, return_sequences=True, activation='tanh')
        self.gru2 = GRU(256, return_state=True, return_sequences=True, activation='tanh')
        self.gru3 = GRU(256, return_state=True, return_sequences=True, activation='tanh')
        self.gru4 = GRU(256, return_state=True, return_sequences=True, activation='tanh')
        self.linear = Dense(output)

    def pairwise_distance(self, A, B):
        rA = tf.reduce_sum(tf.square(A), axis=2)
        rB = tf.reduce_sum(tf.square(B), axis=2)
        distances = - 2*tf.matmul(A, tf.transpose(B, perm = [0,2,1])) + rA[:,:, tf.newaxis] + rB[:,tf.newaxis,:]
        return distances


    def find_nearest_neighbour(self, A, B, dtype=tf.int32):
        nearest_neighbour = tf.argmin(self.pairwise_distance(A, B), axis=2)
        return tf.cast(nearest_neighbour, dtype)
    
    def fix_collisions(self, vc, vb, nb, eps=0.002):
        """
        Fix the collisions between the clothing and the body by projecting
        the clothing's vertices outside the body's surface
        """

        # For each vertex of the cloth, find the closest vertices in the body's surface
        closest_vertices = self.find_nearest_neighbour(vc, vb)
        # print(vb.shape)
        vb = tf.gather(vb, closest_vertices, axis = 1, batch_dims=1) 
        nb = tf.gather(nb, closest_vertices, axis = 1, batch_dims=1) 

        # Test penetrations
        penetrations = tf.reduce_sum(nb*(vc - vb), axis=2) - eps
        penetrations = tf.minimum(penetrations, 0)

        # Fix the clothing
        corrective_offset = -tf.multiply(nb, penetrations[:,:,tf.newaxis]) #scalar multiplication with Vector
        vc_fixed = vc + corrective_offset

        return vc_fixed

    def call(self, inputs):
        pose, translation_vel, shape, hidden_states0, hidden_states1, hidden_states2, hidden_states3, translation, shape_blendshape, joint_transforms, vertices, vertex_normals = inputs
        # (1, 1, 10) (1, 3, 72) (1, 3, 3) (1, 3, 3) (1, 256) (1, 256) (1, 256) (1, 256)
        # print(shape.shape, pose.shape, translation.shape)
        # print("---------------------------------")
        x = tf.concat([shape, pose, translation],axis=-1)
        # print(x.shape)
        x, hidden_states0 = self.gru1(x, initial_state=hidden_states0)
        # print(x.shape)
        x, hidden_states1 = self.gru2(x, initial_state=hidden_states1)
        # print(x.shape)
        x, hidden_states2 = self.gru3(x, initial_state=hidden_states2)
        # print(x.shape)
        x, hidden_states3 = self.gru4(x, initial_state=hidden_states3)
        # print(x.shape)s
        x = self.linear(x)
        # print(x.shape)

        verts_num = v_Tgarment.shape[0]
        # print("1",v_Tgarment)
        x = tf.reshape(x,[-1 ,verts_num, 3])#(1, 4424, 3) Num of GARMENT, not Human.
        # print(x.shape)
        # (16, 1, 85)
        # (16, 1, 256)
        # (16, 1, 256)
        # (16, 1, 256)
        # (16, 1, 256)
        # (16, 1, 13272)
        # (16, 4424, 3)
        v_garment = x + tf.reshape(self.v_TGRAMENT,[-1,3])
        # print(v_garment.shape)

        v_garment = smpl.LBS()(v_garment, joint_transforms, self.g_skinweight)
        # print(v_garment.shape)

        # v_garment += translation[:, tf.newaxis, :]
        v_garment += translation
        # print(translation.shape)
        # print(v_garment.shape)

        v_garment = tf.reshape(v_garment,[-1,verts_num,3])
        #post process
        
        # print(v_garment.shape)
        v_garment = self.fix_collisions(v_garment, vertices, vertex_normals)
        
        return v_garment, tf.stack([hidden_states0, hidden_states1, hidden_states2, hidden_states3])
    


model = Deformation_model(v_Tgarment.shape[0] * 3)
# optimizer = tf.optimizers.SGD(learning_rate=0.00001)
optimizer = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='Adam')
optimizer2 = tf.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='Adam')
optimizer3 = tf.optimizers.Adam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='Adam')
optimizer4 = tf.optimizers.Adam(learning_rate=0.00000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='Adam')

def train():

    batch_size = 2
    tr_data = Data.Data(batch_size=batch_size)
    tr_steps = np.floor(tr_data._n_samples / batch_size)
    num_epochs = 40
    train_steps = int(tr_steps * num_epochs)
    # lr_fn = tf.optimizers.schedules.ExponentialDecay(1e-4, train_steps, 0.0001)
    # opt = tf.optimizers.Adam(lr_fn)
    step = 0

# print(tr_steps)
# print(hidden_states.shape)
#  443.0
# (4, 16, 256)  
    writer_1 = tf.summary.create_file_writer("./mylogs")  # （1）创建一个 SummaryWriter 对象，生成的日志将储存到 "./mylogs" 路径中
    
    hidden_states = tf.stack([
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 0
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 1
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 2
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 3
            ])
    
    save_model_dir = 'train_model_save'

    with writer_1.as_default():  # （2）使用 writer_1 记录with包裹的context中，进行 summary 写入的操作
        for epoch in range(num_epochs):
            print("")
            print("Epoch " + str(epoch + 1))
            print("--------------------------")

            """ TRAIN """
            print("Training...")
            # betas = np.zeros(10, dtype=np.float32)
            # betas = tr_data._shape
            # betas = tf.reshape(betas, (1, 1, 10))
            total_time = 0
            step = 0
            metrics = [0] * 4  # Edge, Bend, Gravity, Collisions
            start = time.time()
            batch_iter = 0
            # print(tr_data._iterator)
            for poses, trans, trans_vel in tr_data._iterator:
                trans = tf.cast(trans, tf.float32)
                betas = tf.random.uniform(shape=[batch_size,10], minval=-3, maxval=3, dtype=tf.float32)[:,tf.newaxis,:]
                #print("for poses, trans, trans_vel in tr_data._iterator:", poses.shape, trans.shape, trans_vel.shape, betas.shape)
                #(1, 3, 72) (1, 3, 3) (1, 3, 3)
                """ Train step """
                with tf.GradientTape() as tape:

                    trans_vel0 = tf.reshape(trans_vel[:,0,:], [-1, 1, trans_vel.shape[2]]) 
                    trans_vel1 = tf.reshape(trans_vel[:,1,:], [-1, 1, trans_vel.shape[2]]) 
                    trans_vel2 = tf.reshape(trans_vel[:,2,:], [-1, 1, trans_vel.shape[2]]) 
                    # print(x0.shape, x1.shape, x2.shape)
                    # (1, 4424, 3) (1, 4424, 3) (1, 4424, 3)
                    # exit
                    pose0 = tf.reshape(poses[:,0,:], [-1, 1, poses.shape[2]]) 
                    pose1 = tf.reshape(poses[:,1,:], [-1, 1, poses.shape[2]]) 
                    pose2 = tf.reshape(poses[:,2,:], [-1, 1, poses.shape[2]]) 
                    trans0 = tf.reshape(trans[:,0,:], [-1, 1, trans.shape[2]]) 
                    trans1 = tf.reshape(trans[:,1,:], [-1, 1, trans.shape[2]]) 
                    trans2 = tf.reshape(trans[:,2,:], [-1, 1, trans.shape[2]]) 
                    
                    # print("OK")
                    v_body0, tensor_dict0 = body(
                        shape=tf.reshape(betas, [-1, 10]),
                        pose=tf.reshape(pose0, [-1, 72]),
                        translation=tf.reshape(trans0, [-1, 3]),
                    )
                    v_body1, tensor_dict1 = body(
                        shape=tf.reshape(betas, [-1, 10]),
                        pose=tf.reshape(pose1, [-1, 72]),
                        translation=tf.reshape(trans1, [-1, 3]),
                    )
                    v_body2, tensor_dict2 = body(
                        shape=tf.reshape(betas, [-1, 10]),
                        pose=tf.reshape(pose2, [-1, 72]),
                        translation=tf.reshape(trans2, [-1, 3]),
                    )

                    # print(v_body0.shape)
                    # print(tensor_dict0["vertex_normals"].shape)
                    # print(tensor_dict0["vertex_normals"])
                    # print(tensor_dict1["vertex_normals"])
                    # print(tensor_dict2["vertex_normals"])

                    # print(pose0.shape, trans_vel0.shape, betas.shape)
                    v_garment0 , hidden_states = model([
                        pose0,
                        trans_vel0,
                        betas,

                        # State of the recurrent hidden layers
                        hidden_states[0],
                        hidden_states[1],
                        hidden_states[2],
                        hidden_states[3],

                        # Additional inputs for LBS and collision postprocess
                        trans0,
                        tensor_dict0["shape_blendshape"],
                        tensor_dict0["joint_transforms"],
                        tensor_dict0["vertices"],
                        tensor_dict0["vertex_normals"]
                    ])
                    v_garment1 , hidden_states = model([
                        pose1,
                        trans_vel1,
                        betas,

                        # State of the recurrent hidden layers
                        hidden_states[0],
                        hidden_states[1],
                        hidden_states[2],
                        hidden_states[3],

                        # Additional inputs for LBS and collision postprocess
                        trans1,
                        tensor_dict1["shape_blendshape"],
                        tensor_dict1["joint_transforms"],
                        tensor_dict1["vertices"],
                        tensor_dict1["vertex_normals"]
                    ])
                    v_garment2 , hidden_states = model([
                        pose2,
                        trans_vel2,
                        betas,

                        # State of the recurrent hidden layers
                        hidden_states[0],
                        hidden_states[1],
                        hidden_states[2],
                        hidden_states[3],

                        # Additional inputs for LBS and collision postprocess
                        trans2,
                        tensor_dict2["shape_blendshape"],
                        tensor_dict2["joint_transforms"],
                        tensor_dict2["vertices"],
                        tensor_dict2["vertex_normals"]
                    ])

                    v_grament = tf.concat([v_garment0[:,tf.newaxis,:,:], v_garment1[:,tf.newaxis,:,:], v_garment2[:,tf.newaxis,:,:]], axis=1)
                    L_inertia = inertial_term_sequence(v_grament ,mass= cloth.v_mass, time_step= 1.0/30)
                    L_strain = stretching_energy(v_garment2, cloth)
                    L_bend = bending_energy(v_garment2, cloth)
                    L_gravity = gravitational_energy(v_garment2, cloth.v_mass)
                    L_collision = collision_penalty(v_garment2, v_body2, tensor_dict2["vertex_normals"])

                    L_bend_New = bending_energy_Willmore(v_garment2, cloth, tensor_dict2["vertex_normals"])
                    # fn = FaceNormals(dtype=v_garment2.dtype, normalize=False)(v_garment2, cloth.f)
                    # print("--------------------")
                    # print(v_garment2.shape)
                    # print(cloth.f.dtype, cloth.f.shape)
                    # print(fn.dtype, fn.shape)

                    loss = L_inertia + L_strain + L_bend + L_gravity + L_collision
                    """ Backprop """
                    # print("1")
                grads = tape.gradient(loss, model.trainable_variables)
                for i in range(0,len(grads)):
                    # print("Grads",i," ",tf.norm(grads[i]), grads[i].shape)
                    tf.summary.scalar("Grads {}".format(i), tf.norm(grads[i]), step=epoch)

                # print(tf.norm(grads[1]))
                # print(tf.convert_to_tensor(grads, dtype = tf.float32))
                # print(tf.norm(tf.convert_to_tensor(grads, dtype = tf.float32)))
                if epoch < 10:
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    # tf.summary.scalar("Current Lr", tf.norm(grads[i]), step=epoch)
                    # print(optimizer._decayed_lr(tf.float32))
                else: 
                    if epoch < 20:
                        optimizer2.apply_gradients(zip(grads, model.trainable_variables))
                        # print(optimizer2._decayed_lr(tf.float32))
                    else:
                        if epoch < 30:
                            optimizer3.apply_gradients(zip(grads, model.trainable_variables))
                            # print(optimizer3._decayed_lr(tf.float32))
                        else:
                            optimizer4.apply_gradients(zip(grads, model.trainable_variables))
                            # print(optimizer4._decayed_lr(tf.float32))
                
                # opt.apply_gradients(zip(grads, model.trainable_variables))
                # print(opt._decayed_lr('float32').numpy())
                # # print(opt.lr)
                # tf.summary.scalar("Lr", opt._decayed_lr('float32').numpy(), step=step)

                print("Official: step{}-{} L_inertia={},L_strain={},L_bend={},L_gravity={},L_collision={},L_bend2={}".format(\
                    step, tr_steps,\
                    L_inertia,L_strain,L_bend,\
                    L_gravity,L_collision,L_bend_New))
                
                tf.summary.scalar("L_inertia", L_inertia, step=epoch)# （3）将scalar("loss", loss, step)写入 summary 
                tf.summary.scalar("L_strain", L_strain, step=epoch)
                tf.summary.scalar("L_bend", L_bend, step=epoch)
                tf.summary.scalar("L_gravity", L_gravity, step=epoch)
                tf.summary.scalar("L_collision", L_collision, step=epoch)

                writer_1.flush()  # （4）强制 SummaryWriter 将缓存中的数据写入到日志文件中（可选）
                step += 1
                
                    
            batch_iter+=1

            model_save_path = os.path.join(save_model_dir,'{}'.format(epoch))
            tf.saved_model.save(model, model_save_path)
            #print("One epoch passed, model saved in", model_save_path)

train()
        
