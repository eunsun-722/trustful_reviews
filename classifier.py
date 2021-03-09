from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Naive bayes function that trains the data using the counter dictionary
# Input: train data and label set
# Output: counter object and naive bayes model used for training
def naive_bayes(train_data, train_labels):
    # vectorizer to map counts of the occurences of each word
    counter = CountVectorizer()
    counter.fit(train_data)
    training_counts = counter.transform(train_data)
    #Naive bayes model
    classifier = MultinomialNB()
    classifier.fit(training_counts, train_labels)
    #classifier.predict gives you a label, predict_proba gives you the percentage for each lable
    return counter, classifier

# Validation function that returns the accuracy of naive bayes model
# Input: counter object, naive bayes model, validation data and label set
# Output: Accuracy of the validation data set based on the validation label
def validate_naive_bayes(counter, model, validation_data, validation_labels):
    validation_data = counter.transform(validation_data)
    return model.score(validation_data, validation_labels)

# Function that returns the prediction of test data using the naive bayes model
# Input: counter object, naive bayes model, test dataset
# Output: list of predictions for each test data
def predict_naive_bayes(counter, model, test_data):
    test_input = counter.transform(test_data)
    prediction = model.predict(test_input)
    return prediction

if __name__ == '__main__':
    print("classify data")
