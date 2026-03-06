import numpy as np

# Inputs (features) and outputs (labels)
X = np.array([1, 2, 3, 4])  # shape (4,)
Y = np.array([2, 4, 6, 8])  # shape (4,)

# Initialize weights
w1 = 0.5
b1 = 0.0
w2 = 1.0
b2 = 0.0

# Hyper Parameters
learning_rate = 0.02
epochs = 200

for epoch in range(epochs):

    # ---- Forward ----
    Z1 = w1 * X + b1
    A1 = np.maximum(0, Z1)
    Y_pred = w2 * A1 + b2

    # ---- Loss ----
    loss = np.mean((Y_pred - Y) ** 2)

    # ---- Backward Pass ----
    dloss_dypred = 2 * (Y_pred - Y) / len(Y)
    dypred_da1 = w2
    da1_dz1 = (Z1 > 0).astype(float)
    dz1_dw1 = X

    dloss_dw2 = np.sum(dloss_dypred * A1)
    dloss_db2 = np.sum(dloss_dypred)
    dloss_dw1 = np.sum(dloss_dypred * dypred_da1 * da1_dz1 * dz1_dw1)
    dloss_db1 = np.sum(dloss_dypred * dypred_da1 * da1_dz1)


    # ---- Update ----
    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2

    if epoch % 10 == 0: print(f"Epoch: {epoch}, Loss: {loss}, Prediction: {Y_pred}")