import matplotlib.pyplot as plt
import numpy as np

prediction = np.genfromtxt('pred_sin_data.csv', delimiter=",")
groundtruth = np.genfromtxt('sin_data.csv', delimiter=",")

fig, ax = plt.subplots()
ax.plot(groundtruth[:, 0], groundtruth[:, 1], label="GT", color='g')
ax.scatter(prediction[:, 0], prediction[:, 1], label="NN pred", color='y')
ax.legend()
ax.set(xlabel="X Value", ylabel="sin(x)")
plt.show()
