# Trustful Reviews
This is a simple machine learning project that correlates whether good reviews result in good rating or vice versa.
Using the amazon product reviews from Electronics category, the application trains the segmentation of review texts using naive bayes model with 0.9+ accuracy and use that model for prediction of reviews for each product.

For the customer review dataset, it was sorted based on the product name and for each review in product, segmentation was performed using trained naive bayes model, and average rating and review segmentation were calculated. Two data were plotted using scatter plot to visualize the correlation on jupyter notebook.

## Datasets
Amazon reviews in Electronics category.
Source: (http://deepyeti.ucsd.edu/jianmo/amazon/index.html)
Amazon customer reviews in Datafiniti Product Database updated between September 2017 and October 2018.
Source: Kaggle (https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products)



## Necessary components for compilation
Numpy, Pandas, Sklearn, gzip, json

## Future Approach
The current approach was only able to train naive bayes model using Electronics department with 100000 data due to memory limitations. Future work includes further training using equally distributed data across different departments for increased accuracy. In addition, the correlation was only computed for about 50 products with about 35000 reviews. Therefore the scatter plot looks rather sparse due to lack of product counts. I was unable to discover larger dataset, but the further work includes finding larger and more diverse dataset for correlation, not only with the ratings, but other factors such as prices, release data, etc.
