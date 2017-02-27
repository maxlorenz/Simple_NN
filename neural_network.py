from random import random, choice


class Neuron(object):

    def __init__(self, num_inputs):
        self.inputs = []
        self.learning_rate = 0.01
        self.weights = [random() for _ in range(num_inputs)]
        self.bias = random()

    @staticmethod
    def activation(x):
        """ReLU function"""
        return (x > 0) * x

    @staticmethod
    def sensitivity(x):
        """Derivative of ReLU"""
        return (x > 0) * 1

    def output(self):
        c = sum([w * i for w, i in zip(self.weights, self.inputs)])
        return self.activation(self.bias + c)

    def adjust(self, error):
        correction = self.sensitivity(self.output())
        correction *= self.learning_rate * error
        self.weights = [w + correction * i
                        for w, i in zip(self.weights, self.inputs)]
        self.bias += correction


class NeuralNetwork(object):

    def __init__(self, inputs=2, hidden_neurons=2):
        self.hidden = [Neuron(inputs) for _ in range(hidden_neurons)]
        self.y = Neuron(hidden_neurons)

    def predict(self, input):
        for h in self.hidden:
            h.inputs = input

        self.y.inputs = [h.output() for h in self.hidden]

        return self.y.output()

    def learn(self, input, target):
        error = target - self.predict(input)
        self.y.adjust(error)

        for h, w in zip(self.hidden, self.y.weights):
            h.adjust(error * w)


if __name__ == "__main__":

    nn = NeuralNetwork(inputs=2, hidden_neurons=4)
    training = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    for epoch in range(100000):
        input, target = choice(training)
        nn.learn(input, target)

    for input, target in training:
        print('IN: {}, EXPECTED: {}, RESULT: {:.2f}'
              .format(input, target, nn.predict(input)))
