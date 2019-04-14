import numpy as np

lambdaVal = 69
n = 3
x = [1, 2, 3] # Should be of length n

rand = np.random.randn(n, n)
# A is positive definite 
A = rand @ rand.T + np.eye(n) * lambdaVal

def f(x):
	# f(x) = x^T*A*x
	return 0.5 * x.T @ A @ x

# Initial x and B values
x = np.random.randn(n) * 1000.0
B = np.eye(n)

def bfgs():
	# derivative of f = 2Ax
	f_derivative = np.dot(A, x)
	p_k = -B*f_derivative



	# Line search in direction p_k to find x_k+1 (golden section search, fixed step size)


	# Find s_k, y_k, p_k, 





# Test that BFGS works
# 1. Collect all xs throughout the descent in a list
# 2. plot f(x) over time gets smaller
# 3. Make sure f(x) goes to 0