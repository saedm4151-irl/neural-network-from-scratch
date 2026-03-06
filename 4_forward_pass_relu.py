# Defining Activation Function ReLU
def relu(x):
    return max(x, 0)

# Inputs
x1 = 2
x2 = 3

# Hidden layer weights and bias
w11, w12, b1 = 1,2,1 # Neuron 1
w21, w22, b2 = 2,1,0 # Neuron 2

# Output weights and bias
w1, w2, b = 1,2,0

# Hidden layer (Pre-activation)
z1 = (w11 * x1) + (w12 * x2) + b1
z2 = (w21 * x1) + (w22 * x2) + b2

# Hidden layer (Post-activation)
a1 = relu(z1)
a2 = relu(z2)

# Output layer (Final Prediction)
y_pred = (w1 * a1) + (w2 * a2) + b

print(y_pred)