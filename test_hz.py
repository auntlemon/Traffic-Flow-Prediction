import time
import scipy.sparse
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import networkx as nx
import utils
import layer
import metrics
import csv
import numpy as np
import tensorflow as tf
import os
import Time
import datetime


object = []
with open("data/hz/graph.pkl", 'rb') as f:
    if sys.version_info > (3, 0):
        object.append(pkl.load(f, encoding='latin1'))
        # print('object', inf)
#     else:
#         objects.append(pkl.load(f))
graph = object[0]
#adj为0/1矩阵
adj = (nx.adjacency_matrix(nx.from_dict_of_lists(graph))).todense()
adj = np.array(adj.astype(dtype=np.float32))

#邻接矩阵预处理
adj = utils.preprocess_adj(adj)

#similar_adj 相似度矩阵
similar_adj = np.loadtxt("data/hz/similar_matrix_hz.txt")
#相似度矩阵预处理
similar_adj = utils.preprocess_adj(similar_adj)

cosine_adj = np.loadtxt("data/hz/cosine_matrix_hz.txt")
cosine_adj = utils.preprocess_adj(cosine_adj)

distance_adj = np.loadtxt('data/adj_distance.txt')
distance_adj = utils.preprocess_adj(distance_adj)

width = 8
length =96//width
road_num = 202
norm_matrix = np.loadtxt("data/hz/norm_matrix_hz.txt")

train_norm = np.array(norm_matrix[:, (length*width)*0:(length*width)*140])
val_norm = np.array(norm_matrix[:, (length*width)*140:(length*width)*180])
test_norm = np.array(norm_matrix[:, (length * width ) * 140:(length * width ) * 180])




# TensorFlow placeholders
with tf.name_scope('input'):
    ph = {
        'distance_adj':tf.placeholder(tf.float32, name="distance_adj_mat"),
        'similar_adj': tf.placeholder(tf.float32, name="similar_adj_mat"),
        'cosine_adj': tf.placeholder(tf.float32, name="cosine_adj_mat"),
        'adj': tf.placeholder(tf.float32, name="adj_mat"),
        'data': tf.placeholder(tf.float32, name="data"),
        'time':tf.placeholder(tf.float32,name='time'),
        'period':tf.placeholder(tf.float32,name='period'),
        'labels': tf.placeholder(tf.float32, shape=(None, 1))
    }


l_sizes = [16, 32, 1, 1]
li = []
li_1 = []
li_2 = []
li_3 = []


with tf.name_scope('layer_1_4'):
    for i in range(length):
        o_fc1_4 = layer.GraphConvLayer(input_dim=width,
                                     output_dim=l_sizes[0],
                                     name='fc1_4',
                                     act=tf.nn.relu)(adj_norm=ph['distance_adj'], x=tf.reshape((ph['data'])[i],[road_num,width]))
        o_fc_14= tf.reshape(o_fc1_4,(1,-1))
        li_3.append(o_fc_14)

o_fc1_v1_4 = tf.reshape(li_3,[length,road_num,l_sizes[0]])#这里的output_din=4,故填入width也正确。否则，应该填入l_sizes[0]。
o_fc1_v2_4 = tf.transpose(o_fc1_v1_4,perm=[1,0,2])
o_fc1_v3_4 =tf.layers.batch_normalization(o_fc1_v2_4,center=False, scale=True,training=True,fused = True,name='bn3')#name error

#
#lstm层
lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
# init_state = lstm_cell_1.zero_state(batch_size=202, dtype=tf.float32)
output,state=tf.nn.dynamic_rnn(cell=lstm_cell_1,inputs=o_fc1_v3_4 ,dtype=tf.float32)
output_v1 = output[:,-1,:]


# # 加入时间周期矩阵
with tf.name_scope('concat'):
    output_v2 = tf.concat([output_v1, ph['period']], 1)
    output_v3 = tf.concat([output_v2, ph['time']], 1)
#
with tf.name_scope('layer_3'):
    o_fc3 = layer.Dense(
        input_dim=output_v1.get_shape().as_list()[-1] +15,
        output_dim=1,
        name='fc3',
        act=tf.nn.sigmoid)(inputs=output_v3)




#优化
with tf.name_scope('optimizer'):
    with tf.name_scope('loss'):
        max = 130.0
        min = 1.17
    with tf.name_scope('train'):
        loss = tf.sqrt(tf.reduce_mean(tf.square((ph['labels']*(max-min)+min) -(o_fc3*(max-min)+min))))
        # tf.summary.scalar('loss',loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            opt_op = optimizer.minimize(loss)



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.Session()
sess.run(tf.global_variables_initializer())

#保存节点
checkpoint_dir = "checkpoints_distance_adj_n5h/"
epoch_n = 0

saver = tf.train.Saver(max_to_keep=1)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    print(("Restored Epoch ", epoch_n))
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    init = tf.global_variables_initializer()
    sess.run(init)



epochs = 1000
days =3330
outputs = {}



min_loss = 20
sum_val_loss = 0
jg = width//2

max = 130.0
min = 1.17
label = []
pre = []
sum_val_loss = 0
test_days = 930
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
saver.restore(sess, ckpt.model_checkpoint_path)
val_date_str = "2014-03-22 04:45:00"
for i in range(0, 930):
    # 时刻
    # print(i)
    time_2 = Time.classify_onehot(val_date_str)
    val_time = np.tile(time_2, (202, 1))
    delta = datetime.timedelta(hours=1)
    val_date_str2dt = datetime.datetime.strptime(val_date_str, '%Y-%m-%d %H:%M:%S')
    val_date_dt = val_date_str2dt + delta
    val_date_str = val_date_dt.strftime('%Y-%m-%d %H:%M:%S')

    val = np.array(val_norm[:, i * jg:length * width + i * jg +20])
    # print(i)
    val_data = np.array(val[:, :96])
    val_label = np.array(val[:, -1])
    val_label = val_label.reshape((202, 1))
    #label


    val_data_v1 = val_data.reshape((road_num, length, width))
    val_data_v2 = np.transpose(val_data_v1, (1, 0, 2))
    val_data_v3 = val_data_v2.reshape((length, val_data_v2.shape[1] * val_data_v2.shape[2]))
    # 周期
    val_period = np.array(val_data[:, 17:22])




    val_loss = sess.run(loss,feed_dict={
                                         ph['period']: val_period, ph['time']: val_time,ph['distance_adj']: distance_adj,
                                         ph['similar_adj']: similar_adj,ph['cosine_adj']: cosine_adj,ph['adj']: adj, ph['data']: val_data_v3, ph['labels']: val_label})
    out = sess.run(o_fc3,feed_dict={
                                    ph['period']: val_period, ph['time']: val_time,ph['distance_adj']: distance_adj,
                                    ph['similar_adj']: similar_adj,ph['cosine_adj']: cosine_adj,ph['adj']: adj, ph['data']: val_data_v3, ph['labels']: val_label})
    sum_val_loss = sum_val_loss + val_loss
    if i==0:
        label = val_label
        pre = out
    else:
        label = np.hstack((label,val_label))
        pre = np.hstack((pre,out))
# print(label.shape)
# print(pre.shape)
label = label*(max-min)+min
pre = pre*(max-min)+min
np.savetxt('data/n5h_hz_dis_adj_label.txt',label,fmt='%0.5f')
np.savetxt('data/n5h_hz_dis_adj_pre.txt', pre,fmt='%0.5f')
print('mean_loss:%f' % (sum_val_loss / 930))
