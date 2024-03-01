import random
import json
import pickle
from sklearn.metrics import r2_score
from neuralplex import Network, Layer, Perceptron

EPOCHS = int(1e5*3)

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
    perceptrons=[Perceptron(coef=random.randint(0, 1)) for i in range(0, 100)], deg=2
)

l3 = Layer(perceptrons=[Perceptron(coef=random.randint(0, 1))], deg=1)

n1 = Network([l1, l2, l3])

for i in range(0, EPOCHS):

    print(i)
    
    n = random.randint(1, 15)

    binary = [int(n) for n in bin(10)[2:]]

    while len(binary) < 4:
        binary = [0] + binary

    n1.epoch(binary, [n])

y_true = []
y_pred = []
for i in range(0, 1000):

    rn = random.randint(1, 15)

    binary = [int(n) for n in bin(rn)[2:]]

    while len(binary) < 4:
        binary = [0] + binary

    n1.epoch(binary, [rn])

    prediction = n1.predict(binary)

    y_true.append(rn)
    y_pred.append(prediction[0])

    print(f"{i} input: {json.dumps(binary)}, truth: {rn} prediction: {json.dumps(prediction)}")

R2 = r2_score(y_true, y_pred)

print(R2)

if R2 >= .7:
    with open("model.pkl", "wb") as f:
        pickle.dump(n1, f)