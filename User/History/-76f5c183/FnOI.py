import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Define the parameters
means = np.array([1.80, 1.13, 1.05, 1.12]).reshape(-1, 1)
stds = np.array([46.81, 6.91, 3.31, 7.62]).reshape(-1, 1)
pi = np.array([0.57, 0.11, 0.22, 0.10])

# Generate random samples
n_samples = 1000
np.random.seed(42)
random_state = np.random.choice(len(means), size=n_samples, p=pi)
samples = np.random.randn(n_samples) * stds[random_state] + means[random_state]

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=len(means))
gmm.fit(samples.reshape(-1, 1))

# Generate points to evaluate the PDF
x = np.linspace(np.min(samples), np.max(samples), 1000).reshape(-1, 1)

# Calculate the PDF values for each component
pdf_values = np.exp(gmm.score_samples(x))

# Plotting the Gaussian mixture components
plt.figure(figsize=(8, 6))
for i in range(len(means)):
    plt.plot(x, gmm.weights_[i] * gmm.means_[i], 'r--', label='Component {}'.format(i+1))
    plt.plot(x, gmm.weights_[i] * pdf_values[:, i], 'b-', label='PDF Component {}'.format(i+1))

plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Gaussian Mixture Model')
plt.show()