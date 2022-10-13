import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0, 2, 4], [2, 4, 2], [3, 3, 1]])
b = np.array([[-2], [-2], [-4]])
c = np.array([[1], [1], [1]])

A_inv = np.linalg.inv(A)
A_invb = np.dot(A_inv, b)
Ac = np.dot(A, c)

print(f'A inverse:\n{A_inv}\n')
print(f'A^(-1)b:\n{A_invb}\n')
print(f'Ac:\n{Ac}')


n = int(np.ceil(1.0 / (0.0025 * 2))) ** 2
ks = [1, 8, 64, 512]
for k in ks:
    Y_k = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1.0 / k), axis=1)
    plt.step(sorted(Y_k), np.arange(1, n + 1) / float(n), label=str(k))

Z = np.random.randn(n)
plt.step(sorted(Z), np.arange(1, n + 1) / float(n), label="Gaussian")

plt.legend(fancybox=True, shadow=True)
plt.xlim(-3,3)
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.show()