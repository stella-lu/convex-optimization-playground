import numpy as np

lambdaVal = 69
n = 3
x = [1, 2, 3] # Should be of length n


B = np.random.randn(n, n)
A = (B @ np.matrix.transpose(B)) + np.eye(n) + lambdaVal
# A is positive definite n by n matrix

A_mul_x = np.dot(A, x)
f = np.dot(np.matrix.transpose(np.array(x)), A_mul_x)

print(A)
print("matrix A_mul_x: ")
print(A_mul_x)
print("x:")
print(np.array(x))
print("f(x) = ", f)

print(type(f))