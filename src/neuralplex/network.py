class Network:

    def __init__(self, layers):
        self.layers = layers;
        self.input = layers[0];
        self.output = layers[len(layers) - 1];

        for i in range(0, len(layers) - 1):
            l1 = layers[i]
            l2 = layers[i + 1]
            l1.connect(l2)

    def epoch(self, inputs, outputs):
        if len(self.input.perceptrons) != len(inputs):
            raise Exception(f'The length of the training values, {len(inputs)}, is not equal to the number of input perceptrons: {len(self.input.perceptrons)}')

        if len(self.output.perceptrons) != len(outputs):
            raise Exception(f'The length of the training values, {len(outputs)}, is not equal to the number of output perceptrons: {len(self.output.perceptrons)}')

        for i in range(0, len(inputs)):
            self.input.perceptrons[i].activate(inputs[i])

        for i in range(0, len(outputs)):
            o = outputs[i]
            p = self.output.perceptrons[i]
            p.propagate(p.q-o)

    def predict(self, inputs):

        if len(self.input.perceptrons) != len(inputs):
            raise Exception(f'The length of the training values, ${len(inputs)}, is not equal to the number of input perceptrons: {len(self.input.perceptrons)}')

        for i in range(0, len(inputs)):
            self.input.perceptrons[i].activate(inputs[i]);

        return [perceptron.q for perceptron in self.output.perceptrons]