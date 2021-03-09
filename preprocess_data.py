import numpy as np
from sklearn.model_selection import train_test_split
import gzip
import json

# Function that loads gz file and save that file as json document in txt format
# Input: file path to load, number of data to load: default=100000
# Output: Parsed data in json format
def load_gz_file(filename, file_size=100000):
    data = []
    for line in gzip.open(filename, "r"):
        my_json = json.loads(line)
        data.append(my_json)
        if (len(data) == file_size):
            break
    #save as json file
    with open('amazon_reviews_short.txt', 'w') as outfile:
        json.dump(data, outfile)
    return data

# Function that loads json document in txt format
# Input: path to the input file
# Output: json data
def load_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


# Load file, separate features and label columns, and set reviews threshold and above to be positive, else negative
# threshold default value is 3
# Returns train_data/labels and test_data/labels
def preprocess(data_file, threshold=3):
    reviews = []
    labels = []
    for entry in data_file:
        reviews.append(entry["reviewText"])
        label = entry["overall"]
        if (label >= threshold):
            labels.append(1)
        else:
            labels.append(0)
    train_data, test_data, train_labels, test_labels = train_test_split(reviews, labels, test_size = 0.2)
    return train_data, test_data, train_labels, test_labels
