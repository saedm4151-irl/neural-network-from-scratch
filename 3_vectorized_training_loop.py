# Training Data
x = [1,2,3,4]
y = [3,5,7,9]

w = 0.5
b = 0.0

learning_rate = 0.016
epochs = 200

for epoch in range(epochs):
    gradient_w = 0
    gradient_b = 0
    total_loss = 0

    for xi, yi in zip(x,y):
        y_pred = w * xi + b

        error = y_pred - yi

        total_loss += error ** 2

        gradient_w += 2 * xi * error
        gradient_b += 2 * error

    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if epoch % 10 == 0: print(f"epoch: {epoch}, total loss: {total_loss}, weight: {w}, bias : {b}")