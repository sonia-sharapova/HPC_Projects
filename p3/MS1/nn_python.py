# Reference: https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
import struct
import numpy as np

train_image_path = '../Data/train-images-idx3-ubyte'
train_label_path = '../Data/train-labels-idx1-ubyte'
test_image_path = '../Data/t10k-images-idx3-ubyte'
test_label_path = '../Data/t10k-labels-idx1-ubyte'
with open(train_label_path, 'rb') as file:
    magic, size = struct.unpack(">II", file.read(8)) # 使わないけど読み込んどかないとlabalsに余計なものが読み込まれちゃう。
    labels = file.read()
with open(train_image_path, 'rb') as file:
    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    image_data = file.read()
print("size:", size, '  rows:', rows, '  cols:', cols)
images = []
for i in range(size):
    images.append([0] * rows * cols)
    images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]



with open(test_label_path, 'rb') as file:
    magic, size_test = struct.unpack(">II", file.read(8))
    labels_test = file.read()
with open(test_image_path, 'rb') as file:
    magic, size_test, rows, cols = struct.unpack(">IIII", file.read(16))
    image_data_test = file.read()
print("size:", size_test, '  rows:', rows, '  cols:', cols)
images_test = []
print(size_test)
for i in range(size_test):
    images_test.append([0] * rows * cols)
    images_test[i][:] = image_data_test[i * rows * cols:(i + 1) * rows * cols]


train = []
for i in range(size):
    label_onehot = [0] * 10
    label_onehot[labels[i]] = 1
    train.append([np.reshape(images[i], (-1,1)), np.reshape(label_onehot, (-1,1))])
test = []
for i in range(size_test):
    label_onehot = [0] * 10
    label_onehot[labels_test[i]] = 1
    test.append([np.reshape(images_test[i], (-1,1)), np.reshape(label_onehot, (-1,1))])



#Inputs
#nl: number of layers in the neural network (excluding the first and last layers)
#nh: number of units per hidden layers
#ne: number of training epochs
#nb: number of training samples per batch
#alpha: learning rate



nl = 2
nh = 10
ne = 10
nb = 160
alpha = 0.3


import numpy as np

def sigmoid(x):
    #print(x, type(x))
    x = np.clip(x, -20, 20)
    sig = 1 / (1 + np.exp(-x))
    return sig

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    
    def forward_propagation(self, x):
        a = x
        a_list = [x] 
        z_list = [] 
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_list.append(z)
            a = sigmoid(z)
            a_list.append(a)
        return a_list, z_list
    
    def back_propagation(self, x, y):
        b_grad = [np.zeros(b.shape) for b in self.biases]
        w_grad = [np.zeros(w.shape) for w in self.weights]
        a_list, z_list = self.forward_propagation(x)
        delta = self.cost_derivative(a_list[-1], y) * sigmoid_derivative(z_list[-1])
        b_grad[-1] = delta
        w_grad[-1] = np.dot(delta, a_list[-2].transpose())
        
        for l in range(2, len(self.layer_sizes)):
            z = z_list[-l]
            s_p = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * s_p
            b_grad[-l] = delta
            w_grad[-l] = np.dot(delta, a_list[-l-1].transpose())
        return b_grad, w_grad
    
    def cost_derivative(self, y_pred, y):
        return (y_pred - y)
    
    def update_mini_batch(self, mini_batch, alpha):
        b_grad = [np.zeros(b.shape) for b in self.biases]
        w_grad = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:  # y coming from how training data is formatted
            delta_b_grad, delta_w_grad = self.back_propagation(x, y)
            b_grad = [nb+dnb for nb, dnb in zip(b_grad, delta_b_grad)]
            w_grad = [nw+dnw for nw, dnw in zip(w_grad, delta_w_grad)]
        self.biases = [b-(alpha/len(mini_batch))*nb for b, nb in zip(self.biases, b_grad)]
        self.weights = [w-(alpha/len(mini_batch))*nw for w, nw in zip(self.weights, w_grad)]

    def train(self, training_data, ne, nb, alpha, test_data):
        n = len(training_data)
        for j in range(ne):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+nb] for k in range(0, n, nb)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
                
            if j % 100 == 0:
                accuracy = self.accuracy(test_data)
                print(f"Epoch {j+1}: {accuracy:.2f}% accuracy on test data")


    def predict(self, x):
        activations, _ = self.forward_propagation(x)
        return activations[-1]

    def evaluate(self, test_data):
        test_results = [(self.predict(x).argmax(), y.argmax()) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def accuracy(self, test_data):
        return self.evaluate(test_data) * 100
    
        
        
layer_sizes = [784, nh, nh, 10] # Example layer sizes for a neural network
nn = NeuralNetwork(layer_sizes)
nn.train(train, ne, nb, alpha, test_data=test)

print("Done Training")
