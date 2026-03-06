# Training Data
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11] # y = 2x + 1

# Setting random values for weight and bias
w = 0.5
b = 0.0

# Setting values for hyper-parameter
learning_rate = 0.015
epochs = 500

for epoch in range(epochs):

    # Resetting Gradient of w, b and total_loss
    gradient_w = 0
    gradient_b = 0
    total_loss = 0

    for i in range(len(x)):

        y_pred = w * x[i] + b

        error = y_pred - y[i]

        total_loss += error ** 2

        gradient_w += 2 * x[i] * error
        gradient_b += 2 * error

    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if epoch % 10 == 0: print(f"epoch: {epoch}, total loss: {total_loss}, weight: {w}, bias : {b}")