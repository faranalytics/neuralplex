class Perceptron:

    def __init__(self, coef, deg=1):
        self.coef = coef
        self.deg = deg
        self.perceptronsRHS = []
        self.perceptronsLHS = []
        self.activations = []
        self.propagations = []
        self.q = 0

    def connect(self, perceptron):
        self.perceptronsRHS.append(perceptron);
        perceptron.perceptronsLHS.append(self);

    def activate(self, t):
        self.activations.append(self.coef * t**self.deg)
        if len(self.perceptronsLHS) == 0 or len(self.activations) == len(self.perceptronsLHS):
            self.q = sum(self.activations)
            for perceptron in self.perceptronsRHS:
                perceptron.activate(self.q)
            self.activations = []

    def propagate(self, t):
        self.propagations.append(t)
        if len(self.perceptronsRHS) == 0 or len(self.propagations) == len(self.perceptronsRHS):
            d = sum(self.propagations)
            if d > 0:
                self.coef = self.coef - 1e-5
            else: 
                self.coef = self.coef + 1e-5
            
            for perceptron in self.perceptronsLHS:
                perceptron.propagate(d)
            self.propagations = []
