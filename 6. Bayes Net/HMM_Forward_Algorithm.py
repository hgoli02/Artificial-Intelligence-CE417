import numpy as np

k = int(input())
n = int(input())
m = int(input())

e = []
e.extend(map(int, input().split()))
X_CPT = []
X_0_CPT = []
E_CPT = []
X_0_CPT.extend(list(map(float, input().split())))
for i in range(n):
    X_CPT.append(list(map(float, input().split())))
for i in range(n):
    E_CPT.append(list(map(float, input().split())))
X_CPT = np.array(X_CPT)
E_CPT = np.array(E_CPT)
B = np.array(X_0_CPT).reshape(n, 1)

# B = np.array([
#     [0.5, 0.5],
# ]).T
#
# X_CPT = np.array([
#     [0.7, 0.3],
#     [0.3, 0.7],
# ])
#
# E_CPT = np.array([
#     [0.9, 0.1],
#     [0.2, 0.8]
# ])
#e = [1, 1]
B_prime = None
for i in range(k):
    P_e_x = E_CPT[:, e[i] - 1: e[i]]
    sigma = np.zeros((n, 1))
    for j in range(n):
        sigma += np.reshape(X_CPT[j: j+1, :], (n, 1)) * B[j:j+1, :]
    B = P_e_x * sigma / np.sum(P_e_x * sigma)
    B_prime = sigma

predictions = np.zeros((1, m))
for i in range(n):
    term_1 = E_CPT[i: i+1, :]
    term_2 = B_prime
    qui = term_2[i, 0]
    predictions += term_1 * term_2[i:i+1, :]

print(np.argmax(predictions) + 1, np.round(np.max(predictions), 2), sep=" ")
