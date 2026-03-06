# Training Data
x = [1, 2, 3, 4 ,5]
y = [2, 4, 6, 8, 10] # y = 2x

# Setting initial weight randomly
w = 0.5

# Setting hyper-parameters
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):

    # Resetting Gradient and total loss
    total_loss = 0
    gradient = 0

    for i in range(len(x)):

        y_pred = w * x[i]

        error = y_pred - y[i]

        total_loss += error ** 2

        gradient += 2* w * error

    w -= learning_rate * gradient

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, total_loss: {total_loss}, weight: {w}")