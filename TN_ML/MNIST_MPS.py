import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Import tensornetwork
import tensornetwork as tn
tn.set_default_backend("tensorflow")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_vec = []
for i in range(x_train.shape[0]):
    x_vec_ = np.concatenate([x_train[i][j, ::-2*(j%2)+1] for j in range(x_train.shape[1])])
    x_vec.append(x_vec_)
x_train_1d = np.vstack(x_vec)

x_vec = []
for i in range(x_test.shape[0]):
    x_vec_ = np.concatenate([x_test[i][j, ::-2*(j%2)+1] for j in range(x_test.shape[1])])
    x_vec.append(x_vec_)
x_test_1d = np.vstack(x_vec)

def feature_map(p):
    phi = [1-p, p]
    return phi

def data_tensorize(vec):
    data_tensor = [tn.Node(feature_map(p)) for p in vec]
    return data_tensor

def block(*dimensions):
    size = tuple([x for x in dimensions])
    return tf.Variable(
        tf.random.normal(shape=size, dtype=tf.dtypes.float64, mean= 1/ np.max(size), stddev = 0.5 / np.max(size)),
        trainable=True)

def create_blocks(rank, dim, bond_dim, label_dim):
    half = np.int((rank - 2) / 2)
    blocks = [
        block(dim, bond_dim) ] + \
        [ block(bond_dim, dim, bond_dim) for _ in range(half)] + \
        [ block(bond_dim, label_dim, bond_dim) ] + \
        [ block(bond_dim, dim, bond_dim) for _ in range(half, rank-2)] + \
        [ block(bond_dim, dim) 
        ]
    return blocks

def create_MPS_labeled(blocks, rank, dim, bond_dim):
    '''Build the MPS tensor'''
    half = np.int((rank - 2) / 2)
    mps = []
    for b in blocks:
        mps.append(tn.Node(b))

    #connect edges to build mps
    connected_edges=[]
    conn=mps[0][1]^mps[1][0]
    connected_edges.append(conn)
    for k in range(1,rank):
        conn=mps[k][2]^mps[k+1][0]
        connected_edges.append(conn)

    return mps, connected_edges

class TNLayer(tf.keras.layers.Layer):
    def __init__(self, input_len, label_num, bond_dim):
        self.label_len = 1
        self.label_dim = label_num
        self.rank = input_len
        self.dim = 2
        self.bond_dim = bond_dim
        #super(TNLayer, self).__init__()
        super().__init__()
        # Create the variables for the layer.
        self.blocks = create_blocks(self.rank, self.dim, self.bond_dim, self.label_dim)

    def call(self, inputs):
        def f(input_vec, blocks, rank, dim, bond_dim, label_len):
            mps, edges = create_MPS_labeled(blocks, rank, dim, bond_dim)
            data_tensor = []
            for p in tf.unstack(input_vec):
                data_tensor.append(tn.Node([1-p, p]))
            edges.append(data_tensor[0][0] ^ mps[0][0])
            half_len = np.int(rank / 2)
            [edges.append(data_tensor[i][0] ^ mps[i][1]) for i in range(1, half_len)]
            [edges.append(data_tensor[i-label_len][0] ^ mps[i][1]) \
                 for i in range(half_len + label_len, rank + label_len)]
            for k in reversed(range(len(edges))):
                A = tn.contract(edges[k])
            #result = tf.math.log(A.tensor)
            result = A.tensor - tf.math.reduce_max(A.tensor)
            return result

        result = tf.vectorized_map(
        lambda vec: f(vec, self.blocks, self.rank, self.dim, self.bond_dim, self.label_len), inputs)
        return result

N = x_train_1d.shape[1]
label_len = 1
label_num = 10
data_len = x_train_1d.shape[1]
rank = data_len
dim = 2
bond_dim = 10

tf.keras.backend.set_floatx('float64')

tn_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(N,)),
        TNLayer(N, label_num, bond_dim),
        tf.keras.layers.Softmax()
    ])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
tn_model.compile(optimizer=optimizer, 
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                 metrics=['accuracy'])
tn_model.fit(x_train_1d, y_train, batch_size=100, epochs=100, verbose=1)