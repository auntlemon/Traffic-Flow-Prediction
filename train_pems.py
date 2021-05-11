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
# np.set_printoptions(threshold=np.inf)
#基于距离，经纬度计算得到
adj = np.loadtxt('data/pems/similar_distance_pems.txt') #Gr
#邻接矩阵预处理
adj = utils.preprocess_adj(adj)

#similar_adj 相似度矩阵
similar_adj = np.loadtxt("data/pems/similar_matrix_pems.txt") #Gp
#相似度矩阵预处理
similar_adj = utils.preprocess_adj(similar_adj)

# print(adj)
# print(similar_adj)


is_train=True
# is_train=False
width = 8
length =96//width
road_num = 608
norm_matrix = np.loadtxt("data/pems/norm_matrix_pems.txt")

train_norm = np.array(norm_matrix[:, (length*width)*0:(length*width)*72])
val_norm = np.array(norm_matrix[:, (length*width)*72:(length*width)*90])
test_norm = np.array(norm_matrix[:, (length * width ) * 72:(length * width ) * 90])
np.savetxt('data/test_pems.txt', test_norm,fmt='%0.5f')




# TensorFlow placeholders
with tf.name_scope('input'):
    ph = {
        'similar_adj': tf.placeholder(tf.float32, name="similar_adj_mat"),
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


with tf.name_scope('layer_1_1'):
    for i in range(length):
        o_fc1_1 = layer.GraphConvLayer(input_dim=width,
                                       output_dim=l_sizes[0],
                                       name='fc1_1',
                                     act=tf.nn.relu)(adj_norm=ph['adj'], x=tf.reshape((ph['data'])[i],[road_num,width]))


        o_fc_111= tf.reshape(o_fc1_1,(1,-1))
        li.append(o_fc_111)

o_fc1_v1 = tf.reshape(li,[length,road_num,l_sizes[0]])#这里的output_din=4,故填入width也正确。否则，应该填入l_sizes[0]。
o_fc1_v2 = tf.transpose(o_fc1_v1,perm=[1,0,2])
o_fc1_v3 = tf.layers.batch_normalization(o_fc1_v2,center=False, scale=True,training=True,fused = True,name='bn1')


with tf.name_scope('layer_1_2'):
    for i in range(length):
        o_fc1_2 = layer.GraphConvLayer(input_dim=width,
                                     output_dim=l_sizes[0],
                                     name='fc1_2',
                                     act=tf.nn.relu)(adj_norm=ph['similar_adj'], x=tf.reshape((ph['data'])[i],[road_num,width]))

        o_fc_12= tf.reshape(o_fc1_2,(1,-1))
        li_1.append(o_fc_12)

o_fc1_v1_2 = tf.reshape(li_1,[length,road_num,l_sizes[0]])#这里的output_din=4,故填入width也正确。否则，应该填入l_sizes[0]。
o_fc1_v2_2 = tf.transpose(o_fc1_v1_2,perm=[1,0,2])
o_fc1_v3_2 = tf.layers.batch_normalization(o_fc1_v2_2,center=False, scale=True,training=True,fused = True,name='bn2')


o_fc1_v4_1 = tf.concat([o_fc1_v3,o_fc1_v3_2],2)#融合后的输出，即lstm层的输入


#
#lstm层
lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
# init_state = lstm_cell_1.zero_state(batch_size=202, dtype=tf.float32)
output,state=tf.nn.dynamic_rnn(cell=lstm_cell_1,inputs=o_fc1_v4_1 ,dtype=tf.float32)
output_v1 = output[:,-1,:]
# output_v1 = tf.reshape(output,(road_num,length*64))


# # 加入时间周期矩阵
with tf.name_scope('concat'):
    output_v2 = tf.concat([output_v1, ph['period']], 1)
    output_v3 = tf.concat([output_v2, ph['time']], 1)
#
with tf.name_scope('layer_3'):
    # dim = tf.Session().run(tf.shape(output_v3))
    # dense1 = tf.layers.dense(inputs=output_v3, units=1, activation=tf.nn.sigmoid,use_bias=False)
    o_fc3 = layer.Dense(
        input_dim=output_v1.get_shape().as_list()[-1]+15,
        # input_dim=d,
        # input_dim=133,
        output_dim=1,
        name='fc3',
        act=tf.nn.sigmoid)(inputs=output_v3)



#优化
with tf.name_scope('optimizer'):
    with tf.name_scope('loss'):
        max = 82.6
        min = 3.0
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
checkpoint_dir = "checkpoints_pems_adj_n1h/"
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
days = 1704
outputs = {}



min_loss = 5
sum_val_loss = 0
jg = width//2
# jg = 4
# Train model
if is_train:
    for epoch in range(epoch_n,epochs):
        # sum_loss = 0
        train_date_str = "2018-01-02 04:45:00"##时刻要对应好,如00：45：00为n1h,01：45：00为n2h,以此类推。
        for day in range(days):
            t = time.time()
            #时刻
            time_1 = Time.classify_onehot(train_date_str)
            train_time = np.tile(time_1,(608,1))
            # print(train_time.shape)
            delta = datetime.timedelta(hours=1)
            train_date_str2dt = datetime.datetime.strptime(train_date_str, '%Y-%m-%d %H:%M:%S')
            train_date_dt = train_date_str2dt + delta
            train_date_str = train_date_dt.strftime('%Y-%m-%d %H:%M:%S')


            train = np.array(train_norm[:, day * jg:length * width + day * jg + 20])
            train_data = np.array(train[:, :96])
            # print(train_data.shape,day)
            train_label = np.array(train[:, -1])
            train_label = train_label.reshape((road_num, 1))
            # print(day)
            # print(train_data.shape)
            # 大小窗口
            train_data_v1 = train_data.reshape((road_num, length, width))
            train_data_v2 = np.transpose(train_data_v1, (1, 0, 2))
            train_data_v3 = train_data_v2.reshape((length, train_data_v2.shape[1] * train_data_v2.shape[2]))
            # 周期
            train_period = np.array(train_data[:, 17:22])

            _, train_loss = sess.run((opt_op, loss),feed_dict={ph['period']:train_period,ph['time']:train_time,ph['similar_adj']: similar_adj,ph['adj']: adj,ph['data']: train_data_v3,ph['labels']: train_label})

            # 验证集合
        val_date_str = "2018-03-15 04:45:00"
        for i in range(0,407):
            # 时刻
            # print(i)
            time_2 = Time.classify_onehot(val_date_str)
            val_time = np.tile(time_2, (608, 1))
            delta = datetime.timedelta(hours=1)
            val_date_str2dt = datetime.datetime.strptime(val_date_str, '%Y-%m-%d %H:%M:%S')
            val_date_dt = val_date_str2dt + delta
            val_date_str = val_date_dt.strftime('%Y-%m-%d %H:%M:%S')

            val = np.array(val_norm[:, i * jg:length * width + i * jg + 20])
            # print(i)
            val_data = np.array(val[:, :96])
            val_label = np.array(val[:, -1])
            # print(val_data.shape,day)
            val_label = val_label.reshape((608, 1))
            val_data_v1 = val_data.reshape((road_num, length, width))
            val_data_v2 = np.transpose(val_data_v1, (1, 0, 2))
            val_data_v3 = val_data_v2.reshape((length, val_data_v2.shape[1] * val_data_v2.shape[2]))
            # 周期
            # 周期
            val_period = np.array(val_data[:, 17:22])



            val_loss = sess.run(loss, feed_dict={ph['period']:val_period,ph['time']:val_time,ph['similar_adj']: similar_adj,
                                                 ph['adj']: adj, ph['data']: val_data_v3, ph['labels']: val_label})
            sum_val_loss = sum_val_loss + val_loss
        if sum_val_loss /407 < min_loss:
            min_loss = sum_val_loss / 407
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch + 1)


        # Print results
        print("Epoch:", '%04d' % (epoch + 1),

              "train_loss=", train_loss,

              "val_loss", sum_val_loss / 407,

              "time=", "{:.5f}".format(time.time() - t))
        sum_val_loss = 0



#test
else:
    max = 82.6
    min = 3.0
    label = []
    pre = []
    sum_val_loss = 0
    test_days = 407
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, ckpt.model_checkpoint_path)
    val_date_str = "2018-03-15 04:45:00"
    for i in range(0, 407):
        # 时刻
        # print(i)
        time_2 = Time.classify_onehot(val_date_str)
        val_time = np.tile(time_2, (608, 1))
        delta = datetime.timedelta(hours=1)
        val_date_str2dt = datetime.datetime.strptime(val_date_str, '%Y-%m-%d %H:%M:%S')
        val_date_dt = val_date_str2dt + delta
        val_date_str = val_date_dt.strftime('%Y-%m-%d %H:%M:%S')

        val = np.array(val_norm[:, i * jg:length * width + i * jg + 20])
        # print(i)
        val_data = np.array(val[:, :96])
        val_label = np.array(val[:, -1])
        val_label = val_label.reshape((608, 1))
        #label


        val_data_v1 = val_data.reshape((road_num, length, width))
        val_data_v2 = np.transpose(val_data_v1, (1, 0, 2))
        val_data_v3 = val_data_v2.reshape((length, val_data_v2.shape[1] * val_data_v2.shape[2]))
        # 周期
        val_period = np.array(val_data[:, 17:22])





        val_loss = sess.run(loss,feed_dict={ph['period']: val_period, ph['time']: val_time, ph['similar_adj']: similar_adj,ph['adj']: adj, ph['data']: val_data_v3, ph['labels']: val_label})
        out = sess.run(o_fc3,feed_dict={ph['period']: val_period, ph['time']: val_time, ph['similar_adj']: similar_adj,ph['adj']: adj, ph['data']: val_data_v3, ph['labels']: val_label})
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
    np.savetxt('data/pems_n5h_label.txt',label,fmt='%0.5f')
    np.savetxt('data/pems_n5h_pre.txt', pre,fmt='%0.5f')
    print('mean_loss:%f' % (sum_val_loss / 407))

