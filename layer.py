#实现gcn层

import tensorflow as tf

def _dot(x, y):

    return tf.matmul(x, y)





class GraphConvLayer:

    def __init__(self, input_dim, output_dim,name, act=tf.nn.relu, bias=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.bias = bias

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.name_scope('weights'):
                self.w = tf.get_variable(name='w',shape=(self.input_dim, self.output_dim), initializer=tf.contrib.layers.xavier_initializer())


            if self.bias:
                with tf.name_scope('biases'):
                    self.b = tf.get_variable(
                        name='b',
                        initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x):
        hw = _dot(x=x, y=self.w)
        ahw = _dot(x=adj_norm, y=hw)
        if not self.bias:
            return self.act(ahw)
        return self.act(tf.add(ahw, self.bias))



    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class Dense:
    """Dense layer."""
    def __init__(self, input_dim, output_dim, name,act=None, bias=False):

        # if dropout:
        #     self.dropout =dropout
        #     # self.dropout = placeholders['dropout']
        #
        # else:
        #     self.dropout = 0.

        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias


        with tf.variable_scope(name):
            with tf.name_scope('weights'):
                self.w = tf.get_variable(
                    name='w',
                    shape=(self.input_dim, self.output_dim),
                    # initializer = tf.random_normal_initializer(mean=0, stddev=1))
                    initializer=tf.contrib.layers.xavier_initializer())
            if self.bias:
                with tf.name_scope('biases'):
                    self.b = tf.get_variable(
                        name='b',
                        initializer=tf.constant(0.1, shape=(self.output_dim,)))


    def call(self, inputs, sparse=False):
        x = inputs

        # dropout
        # x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = _dot(x, self.w)

        # bias
        if self.bias:
            output += self.b

        if self.act==None:

            return output
        else:
            return self.act(output)




    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
