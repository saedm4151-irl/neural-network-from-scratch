# Training Data
x = 2.0
y_real = 5.0

# Initialize weights
w1 = 0.5
b1 = 0.0
w2 = 1.0
b2 = 0.0

# Hyper Parameters
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):

    # ---- Forward Pass ----
    z1 = w1 * x + b1
    a1 = max(0, z1) # ReLU
    y_pred = w2 * a1 + b2

    loss = (y_real - y_pred) ** 2

    # ---- Backward Pass ----
    dloss_dypred = 2 * (y_pred - y_real) # dL/dy_pred
    dypred_da1 = w2 # dy_pred/da1
    da1_dz1 = 1 if z1 > 0 else 0
    dz1_dw1 = x

    dloss_dw2 = dloss_dypred * a1
    dloss_db2 = dloss_dypred

    dloss_dw1 = dloss_dypred * dypred_da1 * da1_dz1 * dz1_dw1
    dloss_db1 = dloss_dypred * dypred_da1 * da1_dz1 * 1

    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2

    if epoch % 10 == 0: print(f"Epoch: {epoch}, Loss: {loss}, Prediction: {y_pred}")