
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensornetwork as tn
tn.set_default_backend("tensorflow")
from tensorflow.keras import Model

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

# Add a channels dimension
#x_train_1d = x_train_1d[..., tf.newaxis].astype("float32")
#x_test_1d = x_test_1d[..., tf.newaxis].astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train_1d, y_train)).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test_1d, y_test)).batch(32)

def block(*dimensions):
    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = tuple([x for x in dimensions])
    return tf.Variable(tf.random.normal(shape=size, stddev=1.0/3.0),trainable=True)

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

class MyModel(Model):
    def __init__(self, input_len, label_num, bond_dim):
        super(MyModel, self).__init__()
        self.label_len = 1
        self.label_dim = label_num
        self.rank = input_len
        self.dim = 2
        self.bond_dim = bond_dim
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
            result = A.tensor
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
bond_dim = 5

model = MyModel(N, label_num, bond_dim)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    print(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))