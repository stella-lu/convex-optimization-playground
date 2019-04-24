import numpy as np
import matplotlib.pyplot as plt

lambdaVal = 69
sizeOfMatrix = 2

# Initial x and B values
x = np.random.randn(sizeOfMatrix, 1)
B = np.eye(sizeOfMatrix) * 2000

rand = np.random.randn(sizeOfMatrix, sizeOfMatrix)
A = rand @ rand.T + np.eye(sizeOfMatrix) * lambdaVal # A is positive definite 

testVals = []
xAxis = []

def f(v):
	# f(v) = 0.5 * v^T * A * v
	# multiply by 0.5 now so that when taking the derivative
	# we only have to multiply A by x.
	return 0.5 * v.T @ A @ v

def bfgs():
	# Returns B_{k+1}
	# Derivative of x.T @ A @ x = 2Ax
	global x, B
	f_derivative = A @ x

	p = -B @ f_derivative
	x_next = x + 0.1 * p # Line search in direction p to find x_k+1 (golden section search, fixed step size)
	s = x_next - x
	y = A @ x_next - f_derivative
	x = x_next 
	B = s @ np.linalg.pinv(y)
	return B

# Test that BFGS works
# 1. Collect all xs throughout the descent in a list
# 2. plot f(x) over time gets smaller
# 3. Make sure f(x) goes to 0
def test():
	for i in range(100):
		testVals.append(f(x)[0][0])
		xAxis.append(i)
		bfgs()

	#np.set_printoptions(suppress=True)
	print(testVals)

def main():
	test()

main()

plt.plot(xAxis, testVals)
plt.show()

