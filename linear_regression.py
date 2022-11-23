import numpy as np
import matplotlib.pyplot as plt

def estimateCoeff(x, y):
	# number of observations/points
	n = np.size(x)

	# mean of x and y vector
	mx = np.mean(x)
	my = np.mean(y)

	# calculating cross-deviation and deviation about x
	xy = np.sum(y*x) - n*my*mx
	xx = np.sum(x*x) - n*mx*mx

	# calculating regression coefficients
	b1 = xy / xx
	b0 = my - b1*mx

	return (b0, b1)

def plotRegressionLine(x, y, b):
	# plotting the actual points as scatter plot
	plt.scatter(x, y, color="m", marker="o", s=30)

	# predicted response vector
	y_pred = b[0] + b[1]*x

	# plotting the regression line
	plt.plot(x, y_pred, color="g")

	# putting labels
	plt.xlabel('x')
	plt.ylabel('y')

	# function to show plot
	plt.show()

def main():
	# data
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

	# estimating coefficients
	b = estimateCoeff(x, y)
	print("Estimated coefficients:\nb0 = {} \
		\nb1 = {}".format(b[0], b[1]))

	# plotting regression line
	plotRegressionLine(x, y, b)

if __name__ == "__main__":
	main()
