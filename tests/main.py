import random
import json
from neuralplex import Network, Layer, Perceptron

l1 = Layer(
    perceptrons=[
        Perceptron(coef=random.randint(0, 10)),
        Perceptron(coef=random.randint(0, 10)),
        Perceptron(coef=random.randint(0, 10)),
        Perceptron(coef=random.randint(0, 10)),
    ],
    deg=1,
)

l2 = Layer(
    perceptrons=[Perceptron(coef=random.randint(0, 1)) for i in range(0, 1000)], deg=2
)

l3 = Layer(perceptrons=[Perceptron(coef=random.randint(0, 1))], deg=1)

n1 = Network([l1, l2, l3])

for i in range(0, int(1e5 * 3)):

    n = random.randint(1, 15)

    binary = [int(i) for i in list("{0:0b}".format(n))]

    while len(binary) < 4:
        binary = [0] + binary

    print(binary, n)

    n1.epoch(binary, [n])

    prediction = n1.predict(binary)

    print(
        f"{i} input: {json.dumps(binary)}, truth: {n} prediction: {json.dumps(prediction)}"
    )
