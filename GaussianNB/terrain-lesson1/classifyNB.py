from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):

    #defining a Gaussian Naive classifier
    classifier = GaussianNB()
    return classifier.fit(features_train, labels_train)
