from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train):

    #defining a Gaussian Naive classifier
    classifier = GaussianNB()
    return classifier.fit(features_train, labels_train)


def NBAccuracy(features_train, labels_train, features_test, labels_test, classifier):
    """ compute the accuracy of your Naive Bayes classifier """
    ### use the trained classifier to predict labels for the test features
    pred = classifier.predict(features_test)

    ### calculate and return the accuracy on the test data
    accuracy = accuracy_score(pred, labels_test)
    return accuracy