These files are a transformation of the Yelp challenge data set.

https://www.kaggle.com/c/yelp-recsys-2013

The data has been reduced to a 5-core, meaning that each user has a minimum of 5 ratings and each business has a minimum of 5 raters. 

-----------------
5-core JSON files
-----------------

review.json
business.json
checkin.json
user.json

These files contain all of the data in the 5-core of the dataset. Note that these are not actually JSON files, but rather text files where each line is a JSON expression. So, they have to be read in one line at a time.

---------------
Data files
---------------

These files represent a transformation of the Yelp into a form that Librec can use using integer feature ids. The features for businesses are binary features defined in feature-lookup.csv. Mostly these are business types such as restaurant cuisines. Note that the "total review count" variable for users differs from the number of reviews for that user in this data set because the k-core operation has removed many businesses.


user ids in the range 10000 - 18042
business ids in the range 0 - 5198
feature ids in the range 0 - 436
ratings in the range 1 - 5
business token, user token: these string-encoded ids correspond to the identifier values in the JSON files.

reviews.csv: user id, business id, rating

user.csv: user id, # of funny votes, # of useful votes, # of cool votes, avg rating, total review count

business.csv: business id, feature list

business-lookup.csv: business id, business token

user-lookup.csv: user id, user token

feature-lookup.csv: feature id, feature name
