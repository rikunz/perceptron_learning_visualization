import numpy as np

class Perceptron:
    def __init__(self, training_data, target_data, threshold=0.2, learning_rate=1, max_epoch=100):
        """
        Initialize the Perceptron model
            :param training_data: numpy array of training data
            :param target_data: numpy array of target data
            :param threshold: threshold value for the activation function
            :param learning_rate: learning rate for the model
        """
        self.model_name = 'Perceptron'
        # Model Section
        self.training_data = training_data if type(training_data) == np.ndarray else np.array(training_data)
        self.target_data = np.array(target_data)
        self.weights = np.zeros(self.training_data.shape[1])
        self.bias = 0
        self.threshold = threshold
        self.learning_rate = learning_rate
        # Training Section
        self.epochs = 1
        self.current_data = 0
        self.num_weight_not_change = 0
        self.max_epoch = max_epoch
    def get_weights(self):
        return self.weights
    def get_bias(self):
        return self.bias
    def get_model_parameter(self):
        return self.weights, self.bias
    def train_step(self):
        if self.num_weight_not_change == len(self.training_data):
            return f"Model sudah konvergen pada epoch ke-{self.epochs}"
        if self.epochs >= self.max_epoch:
            return f"Epoch sudah mencapai batas maksimal"
        data = self.training_data[self.current_data]
        target = self.target_data[self.current_data]
        activation = self.activation(data)
        if (activation == target):
            self.num_weight_not_change += 1
            self.current_data = (self.current_data + 1) % len(self.training_data)
            if (self.current_data == 0):
                self.epochs += 1
            return
        deltaData = self.learning_rate * data * target
        deltaBias = self.learning_rate * target
        self.weights += deltaData
        self.bias += deltaBias
        self.current_data = (self.current_data + 1) % len(self.training_data)
        if (self.current_data == 0):
            self.epochs += 1
        self.num_weight_not_change = 0
        return
    def train(self):
        while self.num_weight_not_change < len(self.training_data) and self.epochs < self.max_epoch:
            self.train_step()
        return f"Model sudah konvergen pada epoch ke-{self.epochs}"
    def get_epoch(self):
        return self.epochs
    def activation(self, input_data):
        activation = np.dot(input_data, self.weights) + self.bias
        return 1 if activation > self.threshold else -1 if activation < self.threshold else 0
    def predict(self, input_data):
        return self.activation(input_data)
    
class PerceptronSoftmax:
    def __init__(self, training_data, target_data, threshold=0.2, learning_rate=1, max_epoch=100):
        """
        Initialize the Perceptron model
            :param training_data: numpy array of training data
            :param target_data: numpy array of target data
            :param threshold: threshold value for the activation function
            :param learning_rate: learning rate for the model
        """
        self.model_name = 'Perceptron'
        # Model Section
        self.training_data = training_data if type(training_data) == np.ndarray else np.array(training_data)
        self.target_data = np.array(target_data)
        self.weights = np.zeros(self.training_data.shape[1], self.target_data.shape[0])
        self.bias = 0
        self.threshold = threshold
        self.learning_rate = learning_rate
        # Training Section
        self.epochs = 1
        self.current_data = 0
        self.num_weight_not_change = 0
        self.max_epoch = max_epoch

    def get_weights(self):
        return self.weights
    def get_bias(self):
        return self.bias
    def get_model_parameter(self):
        return self.weights, self.bias
    def train(self):
        while self.num_weight_not_change < len(self.training_data) and self.epochs < self.max_epoch:
            self.train_step()
        return f"Model sudah konvergen pada epoch ke-{self.epochs}"
    def get_epoch(self):
        return self.epochs
    def activation(self, input_data):
        activation = np.dot(input_data, self.weights) + self.bias
        return 1 if activation > self.threshold else -1 if activation < self.threshold else 0
    def activation_all(self, input_data):
        activation = np.sum(self.weights * input_data, axis=1) + self.bias
        return np.where(activation > self.threshold, 1, np.where(activation < -self.threshold, -1, 0))
    def predict(self, input_data):
        activation = self.activation(input_data)
        predicted = np.argmax(activation)
        return predicted
    def train_step(self):
        if self.num_weight_not_change == len(self.training_data):
            return f"Model sudah konvergen pada epoch ke-{self.epochs}"
        if self.epochs >= self.max_epoch:
            return f"Epoch sudah mencapai batas maksimal"
        data = self.training_data[self.current_data]
        target = self.target_data[self.current_data]
        activation = self.activation_all(data)
        if (np.allclose(activation, target)):
            self.num_weight_not_change += 1
            self.current_data = (self.current_data + 1) % len(self.training_data)
            if (self.current_data == 0):
                self.epochs += 1
            return
        for idx, idxdata, idxtarget in enumerate(zip(data, target)):
            if (self.activation(idxdata) == idxtarget):
                continue
            deltaData = self.learning_rate * idxdata * idxtarget
            deltaBias = self.learning_rate * idxtarget
            self.weights[idx] += deltaData
            self.bias[idx] += deltaBias
        self.current_data = (self.current_data + 1) % len(self.training_data)
        if (self.current_data == 0):
            self.epochs += 1
        self.num_weight_not_change = 0
        return
    
if __name__ =="__main__":
    perceptron_model = Perceptron(
    [
        np.random.choice([1,-1], 63),
        np.random.choice([1,-1], 63),
        np.random.choice([1,-1], 63)
    ], 
    [
        np.random.choice([-1, 1]),
        np.random.choice([-1, 1]),
        np.random.choice([-1, 1])
    ]
    )   
    print("before training")
    print(perceptron_model.get_epoch())
    print(perceptron_model.get_weights())
    print(perceptron_model.get_bias())
    print(perceptron_model.training_data[perceptron_model.current_data])
    print(perceptron_model.target_data[perceptron_model.current_data])

    print("training")
    perceptron_model.train_step()
    print(perceptron_model.get_epoch())
    print(perceptron_model.get_weights())
    print(perceptron_model.get_bias())
    print(perceptron_model.get_model_parameter())