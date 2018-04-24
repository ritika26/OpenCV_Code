from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))
def predict(X, W):
    	# take the dot product between our features and weight matrix
	preds = sigmoid_activation(X.dot(W))

	# apply a step function to threshold the outputs to binary
	# class labels
	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1

	# return the predictions
	return preds

  
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
args = vars(ap.parse_args())
print(args)

# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
	cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
#print(y)
X = np.c_[X, np.ones((X.shape[0]))]
print(X)

# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
print(X.shape[0])
print(X.shape[1])
print(X.shape)
print(trainX.shape,trainY.shape,testX.shape,testY.shape)
W = np.random.randn(X.shape[1], 1)
print(W)
losses=[]
print(trainX.shape)
print(trainX.T)
for epoch in range(0,args['epochs']):
    pred = sigmoid_activation(trainX.dot(W))
    error = pred- trainY
    loss=np.sum(error**2)
    losses.append(loss)
    gradient = trainX.T.dot(error)
    W += -args["alpha"] * gradient
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),loss))
# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(preds)
print(classification_report(testY, preds))
# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure that plots the loss over time
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
