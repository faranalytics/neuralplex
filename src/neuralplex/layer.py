class Layer:

    def __init__(self, perceptrons, deg):
        self.perceptrons = perceptrons
        self.deg = deg

        if self.deg:
            for perceptron in self.perceptrons:
                perceptron.deg = self.deg

    def connect(self, layer):
        for p1 in self.perceptrons:
            for p2 in layer.perceptrons:
                p1.connect(p2)