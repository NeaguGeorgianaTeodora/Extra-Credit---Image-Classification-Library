# Image-Classification-Library

In this project, we implement an image classification library in an OOP idiomatic way.
Image classification is a computer vision task that analyzes images and “tells” the user the class of the main object depicted in the image. Usually, an image classifier learns from a set of images that are labeled with names, like "dog" or "cat.", and then, when a new image is presented to the algorithm, it tries to determine its class based on the features it previously learned. 

An image classification framework typically involves the following steps:
  -Data collection and preparation: A (large) dataset of labeled images is gathered for training the model. The images are manually annotated or labeled with their corresponding class or category. Several image    classification benchmarks already exist, so you don’t have to worry about this part.
  -Feature extraction: Image features are extracted from the training images to represent the unique characteristics of each class. These features can include color histograms, texture descriptors, edge            information, or more complex features extracted by learning algorithms. For this simple image classification library the features that you will use are the plain pixel intensities.

## Training the model: 
A machine learning algorithm is trained using the labeled images and their corresponding labels. During training, the model learns to associate the extracted features with the correct class labels.

## Testing and evaluation: 
Once the model is trained, it is evaluated using a separate set of labeled images called the test set. The model predicts the class labels for the test images, and the predictions are compared against the ground truth labels (what the images in the test set actually represent). Performance metrics such as accuracy, precision, recall, and F1 score are calculated to assess the model's performance.

## Inference or prediction: 
After the model has been trained and evaluated, it can be used for classifying new, unseen images. The model takes an input image, extracts the relevant features, and assigns it to one of the predefined classes based on the learned associations.

### We will implement two simple classifiers: K nearest neighbor classifier and Naive Bayes classifier.

## K Nearest Neighbours Classifier
K-Nearest Neighbors (KNN) is a simple and intuitive machine-learning algorithm used for classification tasks. It is a type of instance-based learning, meaning it doesn't explicitly learn a model from the training data but instead uses the training examples themselves as the knowledge for making predictions. The algorithm is a non-parametric algorithm, meaning it doesn't make any assumptions about the underlying data distribution. It can handle complex decision boundaries and works well with small to medium-sized datasets. However, it can be computationally expensive, especially when dealing with large datasets, as it requires calculating distances between the new data point and all the training data.
To use KNN effectively, it is important to choose an appropriate value for K (a hyperparameter of the algorithm that needs to be tuned), select an appropriate distance metric (Euclidian, Manhattan etc.), and preprocess the data to ensure meaningful distance calculations.

### The KNN classifier works as follows:

#### Training phase
During training, the algorithm stores all the labeled data points in memory. Each data point consists of features (attributes) and the corresponding class label.

#### Prediction phase
When a new, unlabeled data point is given for classification, the KNN algorithm finds the K closest labeled data points (nearest neighbors) in the feature space based on a distance metric, typically Euclidean distance. 
Once the K nearest neighbors are identified, the algorithm looks at their class labels. In the case of classification, the class labels of the K neighbors are used to determine the class of the new data point. This can be done by majority voting, where the class with the highest count among the neighbors is assigned to the new data point. Alternatively, for regression tasks, the algorithm can average the values of the K nearest neighbors to predict a continuous value.
Again, K is a hyperparameter and needs to be set and tuned by the user.



## Naive Bayes Classifier
Bayes' rule is a fundamental principle in probability theory and statistics. It provides a way to update our beliefs or estimates about the probability of an event occurring based on new evidence or information.
Formally, Bayes' rule is expressed as:

P(A|B) = (P(B|A) * P(A)) / P(B)

where:

P(A|B) represents the conditional probability of event A given event B has occurred. It represents the probability of event A occurring after considering the new evidence or information provided by event B.
P(B|A) is the conditional probability of event B given event A has occurred. It represents the probability of observing event B, assuming that event A is true.
P(A) is the prior probability of event A, representing our initial belief or estimate about the probability of A before considering any new evidence.
P(B) is the prior probability of event B, representing the overall probability of observing event B.
In words, Bayes' rule states that the probability of A given B is proportional to the probability of B given A, multiplied by the prior probability of A, and divided by the prior probability of B.

Bayes' rule allows us to incorporate new evidence or data to update our beliefs or estimates. We start with an initial belief (prior probability) about the probability of an event, and as new evidence becomes available, we adjust our beliefs using Bayes' rule to obtain the updated probability (posterior probability) of the event.

The Naive Bayes classifier applies the Bayes rule and an independence assumption to calculate the posterior probability of each class. The class with the highest probability is chosen as the output. Due to the independence assumption it can handle an arbitrary number of independent variables whether continuous or categorical. Given a set of random variables, {x1, x2, …, xd}, we want to construct the posterior probability for the random variable C having the set of possible outcomes { c1, c2, …, cj}. The elements x are the predictors or features and C is the set of categorical levels or classes present in the dependent variable. Using Bayes' rule we can write:

P(c|x1, x2,..., xd) = P(c)P(x1, x2,..., xd | c)

where P(c|x1, x2,..., xd) is the posterior probability of class membership, i.e., the probability that x belongs to C given the features; P(x1, x2,..., xd | c) is is the likelihood and is the prior.

Using Bayes' rule from above, we can implement a classifier that predicts the class based on the input features x. This is achieved by selecting the class c that achieves the highest posterior probability.

Although the assumption that the predictor variables (features) are independent is not always accurate, it does simplify the classification task dramatically, since it allows the class conditional densities to be calculated separately for each variable. In effect, Naive Bayes reduces a high-dimensional density estimation task to multiple one-dimensional kernel density estimations.

#### Training phase
Let X denote the feature matrix for the training set, as usual. In this case X contains on every row the binarized values of each training image to either 0 or 255 based on a selected threshold. X has the dimension n x d, where n is the number of training instances and d=28 x 28 is the number of features which is equal to the size of an image.
The class labels are stored in the vector y of dimension n.
The prior for class i is calculated as the fraction of instances from class i from the total number of instances:
P(C=i ) = ni/n
The likelihood of having feature j equal to 255 given class i is given by the fraction of the training instances which have feature j equal to 255 and are from class i:.
The likelihood of having feature j equal to 0 is the complementary event.
To avoid multiplication by zero in the posterior probability, likelihoods having the value of 0 need to be treated carefully. A simple solution is to change all values smaller than 10-5 to 10-5.

#### Classification phase
Once the likelihood values and priors are calculated classification is possible. The values for the likelihood are in the interval [0,1] and the posterior is a product of 784 numbers each less than 1. To avoid precision problems, it is recommended to work with the logarithm of the posterior. Denote the test vector as T and its elements as Tj . These are the binarized values from the test image in the form of a vector. The log posterior of each class can be evaluated as:

Since the ordering of the posteriors does not change when the log function is applied, the
predicted class will be the one with the highest log posterior probability value.

The prior will be a Cx1 vector, where C is is the number of classes.
The likelihood is a Cxd vector, where d is the number of dimensions (i.e. the number of pixels in the image).
You will need to store these in the prior and the likelihoold as attributes in the BayesClassifier class.

## Implementation details
We have an abstract base class Classifier, with the following abstract methods:
  - void fit(T trainImages, vector<int> trainLabels ) -> this will be implemented in the subclasses and will actually train the classifier.
  - vector<int> predict(T) -> this will be implemented in the subclasses and will return the predicted labels for all the images in the matrix T.
  - bool save(string filepath); -> stores all the information related to the classifier in a file. You can choose the format.
  - bool load(string filepath); -> reads all the information related to the classifier from the file passed as parameter.
  - double eval(T) -> returns the accuracy of the classifier (the number of correct predictions for the images in T divided by the total number of samples in T).
  - we have two subclasses KNNClassifier and BayesClassifier that implement these classification algorithms.


## Dataset

We use the MNIST dataset for classification. The MNIST database of handwritten digits (so we have 10 classes, the digits from 0 to 9). The dataset comprises a training set of 60,000 grayscale images, and a test set of 10,000 grayscale images. Each image has a dimension of 28x28 pixels.

We work with this version, in which the images are stored in a csv file: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download .
