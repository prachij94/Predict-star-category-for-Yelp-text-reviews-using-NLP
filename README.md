# Predict-star-category-for-Yelp-text-reviews-using-NLP

We have used **Natural Language Processing (NLP)** to classify *Yelp* Reviews into 1 star or 5 star categories based off the text content in the reviews.
Natural Language Processing basically consists of combining machine learning techniques with text, and using math and statistics to get that text in a format that the machine learning algorithms can understand.

We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).

Each observation in this dataset is a review of a particular business by a particular user.
The features include:
- 'business_id': encrypted business id
- 'user_id': encrypted user id
- 'review_id': encrypted review id
- 'stars': star rating
- 'type': 'review'
- 'text': review text
- 'date': date of review
- 'useful': count of votes for useful
- 'funny':  count of votes for funny
- 'cool': count of votes for cool

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users. 

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.

The "useful" and "funny" columns are similar to the "cool" column.

The data is explored using seaborn by finding general relationships like histograms and boxplots between **text length and star ratings**,countplot for **star ratings**,heatmap for the correlation of mean values of the columns **cool,useful,funny with	text length**.

For training the model, a pipeline is created with a ```Count Vectorizer```, a ```Tfidf Transformer``` and a ```Multinomial Naive Baiyes``` object and fit to the train data. But the predictions' accuracy fell remarkably down as compared to when the Tfidf Transformer was not included.

As Tfidf is crucial for better text classification, the previous classifier in the pipeline i.e. *Multinomial Naive Baiyes* was replaced with a linear support vector machine (SVM) classifier i.e. ```SGDClassifier```.
The final accuracy of prediction on the dataset was observed to be *approx. 95%* using these objects.
