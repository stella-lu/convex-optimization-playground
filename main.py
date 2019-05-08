import numpy as np
import matplotlib.pyplot as plt

lambdaVal = 69
sizeOfMatrix = 6
# Initial x and B values
x = np.random.randn(sizeOfMatrix, 1)
B = np.eye(sizeOfMatrix)
# Construct a random positive definite matrix, A, for f(v)
rand = np.random.randn(sizeOfMatrix, sizeOfMatrix)
A = rand @ rand.T + np.eye(sizeOfMatrix) * lambdaVal
def f(v):
	# f(v) = 0.5 * v^T * A * v
	# multiply by 0.5 now so that when taking the derivative
	# we only have to multiply A by x.
	return 0.5 * v.T @ A @ v

def bfgs(numIterations=50):
	# Returns a tuple containing an array of x-values and an array of y-values
	# as a result of performing bfgs on B.
	global x, B
	xAxis, yAxis = [], []
	count = 0
	while count < numIterations:
		# Returns B_{k+1}
		# Derivative of x.T @ A @ x = 2Ax
		xAxis.append(count)
		yAxis.append(f(x)[0][0])
		
		f_derivative = A @ x
		p = -B @ f_derivative
		x_next = x + 0.01 * p # Line search in direction p to find x_k+1, fixed step size
		s = x_next - x
		y = A @ x_next - f_derivative
		x = x_next 
		B = s @ np.linalg.pinv(y)
		count += 1
	return xAxis, yAxis

# Test that BFGS works
# 1. Collect all xs throughout the descent in a list
# 2. plot f(x) over time gets smaller
# 3. Make sure f(x) goes to 0
def main():
	bfgsX, bfgsY = bfgs(4)
	plt.plot(bfgsX, bfgsY)
	plt.show()

main()



