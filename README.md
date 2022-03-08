# About the Project : Netflix-Movie-Recommendation

## To run this project you need

- python3
- pip

## Steps for running this project

- git clone the project
- Create a virtual environment by running the following command
  - `python3 -m venv netflix-venv` (netflix-venv is the name of the virtual environment)
- Activate the virtual environment by running the following command
  - `source netflix-venv/bin/activate`
- Now run the following command
  - `pip install -r requirements.txt`
- Now to install nlk packages
  - `python -m nltk.downloader popular`

## Abstract

Our lives have been drastically altered as a result of the coronavirus pandemic. The general public is advised to keep a social distance and stay at home. So, besides online school and work, what do they do at home? They require some form of entertainment. We have no other options for entertainment besides watching TV, going to the movies, or engaging in indoor activities. Statistics show that the number of people watching movies on OTT services like Netflix has increased since the lockdown. So we gathered a dataset containing information such as movie title, cast, type, and ratings and used various algorithms to recommend films.

## Introduction

Netflix is unquestionably the undisputed king of streaming media. The company, which started out as a mail-order DVD rental service more than 20 years ago, has since completely changed its business model to keep up with the ever-changing tech landscape.As a result, the company now has over 200 million members worldwide and has established itself as one of the world's top media publishers thanks to its Netflix Originals program.

As a result of the coronavirus outbreak, our lives have been profoundly impacted. The general population is advised to maintain a safe distance from others and to remain at home. So, aside from online school and employment, what do they do at home? They require some form of entertainment. Other than watching TV, going to the movies, or engaging in indoor activities, we don't have any other options for entertainment. According to statistics, the number of individuals watching movies on OTT services like Netflix has increased since the lockout. As a result, we compiled a dataset containing information such as the film's title, cast, genre, and ratings, and used multiple algorithms to generate movie recommendations.

## Recommendation System and it's types

A recommendation engine is a filtering system that analyzes data from many sources belonging to various users and generates solutions to predict their interests and offer suitable products to the appropriate consumers. The Recommendation system, on the other hand, is a machine learning algorithm that suggests things users might like based on their past choices. Recommendation engines use two different sorts of techniques:

## Collaborative filtering

Collaborative recommendation systems combine object ratings or suggestions, identify user relationships based on ratings, and produce new recommendations based on user-to-user comparisons. It's usually focused on gathering and analyzing data about a user's behavior, activities, or interests, and then predicting what they'll like based on their similarity to other users. Similarities between users and/or items are analyzed by collaborative systems. Identifying the users who are the most similar to the one being recommended We examine the user-to-user relationship (similarity), which we determined using correlation.

## Content-Based Filtering

Additional information on users and/or things is used in the content-based approach. This form of filtering uses item features to suggest other items that are comparable to what the user likes, as well as previous actions or explicit input. If we take the case of a movie recommender system, the additional information may include the user's age, gender, occupation, or any other personal information, as well as the category, principal actors, duration, or other features of the movies (i.e. the items). The basic goal of content-based techniques is to create a model that explains observed user-item interactions using the given "features." Such a model allows us to quickly make fresh predictions for a user by simply looking at their profile and determining appropriate movies to recommend based on their information.

## Datasets Used

### Netflix Prize data

This is the Netflix Prize competition's official data collection. The Netflix Prize is made up of about 100,000,000 ratings for 17,770 films submitted by 480,189 individuals. Each rating in the training dataset has four components: user, movie, grade date, and grade. Integer IDs are used to identify users and movies, and ratings range from 1 to 5.

Link - [https://www.kaggle.com/netflix-inc/netflix-prize-data]

The attributes of the ratings dataset are given below -

- Cust_Id
- Rating
- Movie_Id

The attributes of the movies_titles are dataset are given below -

- Movie_Id
- Year
- Name

### The Netflix Movies and TV Shows dataset

There are so many datasets under the topic netflix analysis. From that, we have selected a dataset "Netflix Movies and TV shows" based on the fact that more the rows the more data we can infer. As of 2020, this dataset contains TV series and movies available on Netflix. The information was gathered through Flixable, a third-party Netflix search engine.

Link - [https://www.kaggle.com/shivamb/netflix-shows],
The attributes of the selected dataset are - show_id, type,title, director, cast, country, date_added, release_year, rating, duration, listed_in, description.

## Algorithm Used

### Word2vec

#### Introduction

This tool uses the continuous bag-of-words and skip-gram designs to efficiently compute word vector representations. These representations can then be employed in a number of natural language processing applications and for additional study.

#### Word Embedding

It is a language modeling technique for mapping words to real-number vectors. It is a vector space representation of words or phrases with multiple dimensions. Word embeddings can be generated using a variety of techniques such as neural networks, co-occurrence matrices, probabilistic models, and so on.

#### How does it work?

The word2vec tool takes a text corpus as input and outputs word vectors. It creates a vocabulary from the training text data and then learns word vector representation. The word vector file that results can be used as a feature in a variety of natural language processing and machine learning applications. Finding the closest words for a user-specified word is a simple way to investigate the learned representations. That is what the distance tool is for. For example, if you enter 'france,' distance will display the most similar words and their distances from 'france,

### Node2vec

#### Introduction

A node embedding algorithm that computes a vector representation of a node based on random graph walks. Deep random walks are also used to sample the graph's neighbor nodes. In order to efficiently explore diverse neighborhoods, this algorithm employs a biased random walk procedure. It operates in the same way as Word2Vec, except instead of word embeddings, node embeddings are created.

#### How it works?

The Node2Vec system works on the premise of learning continuous feature representations for network nodes while retaining information from the previous 100. This allows us to understand how the algorithm works.Assume we have a graph with a few interconnected nodes that form a network. As a result, the Node2Vec algorithm learns a dense representation of each node in the network (say, 100 dimensions/features). According to the algorithm, if we plot these 100 dimensions of each node in a two-dimensional graph using PCA, the distance between the two nodes in that low-dimensional graph will be the same as their actual distance in the given network. In this way, the framework increases the likelihood of neighboring nodes being preserved even if they are represented in a low-dimensional space.

### Sentence Transformer

This framework makes it simple to generate dense vector representations of sentences, paragraphs, and images. This framework can be used to compute sentence / text embeddings for over 100 languages. These embeddings can then be compared, for example, using cosine-similarity to find sentences with similar meanings. This can help with semantic textual similarity, semantic search, and paraphrase mining. The models are based on transformer networks such as BERT / RoBERTa / XLM-RoBERTa and achieve cutting-edge performance in a variety of tasks. Text is embedded in vector space in such a way that similar text is close and can be found quickly using cosine similarity.

### MiniBatchKMeans

#### Introduction

When clustering large datasets, the Mini-batch K-means clustering algorithm is a variation of the K-means algorithm that can be used instead of the K-means algorithm. Because it does not cycle over the complete dataset, it sometimes outperforms the usual K-means algorithm when working on large datasets. It generates random batches of data to be stored in memory, then collects a random batch of data to update the clusters on each cycle.

#### How it works?

The Mini Batch K-means algorithm's main idea is to use tiny random batches of data with a preset size that may be stored in memory. Each iteration utilizes a new random sample from the dataset to update the clusters, and the procedure is repeated until convergence. With a diminishing learning rate as the number of iterations grows, each micro batch updates the clusters using a convex mix of the prototype values and the data. This learning rate is the inverse of the number of data assigned to a cluster during the operation. Convergence may be detected when no changes in the clusters occur for numerous iterations in a row, because the influence of incoming data lessens as the number of iterations grows.

For each iteration, the algorithm selects tiny groups of the dataset at random. Each piece of data in the batch is assigned to one of the clusters based on the cluster centroids' previous placements. The cluster centroids' coordinates are then updated based on the new points from the batch. The update is gradient descent, which is much faster than a batch K-Means update.

### Cosine Similarity and TF-IDF

In Natural Language Processing, cosine similarity is one of the metrics used to compare the text similarity of two documents, regardless of their size. A vector representation of a word is created. In n-dimensional vector space, the text documents are represented. The cosine of the angle between two n-dimensional vectors projected in a multi-dimensional space is measured by the Cosine similarity metric. A document's Cosine similarity will vary from 0 to 1. When the Cosine similarity score is 1, it signifies that two vectors are oriented in the same way. The closer the value is to 0, the less similar the two documents are.

The TF-IDF statistic investigates the relation of a word to a document in a collection of documents. This is done by multiplying two metrics: the number of times a word appears in a document and the word's inverse document frequency across a group of documents. It has a wide range of applications, including automatic text analysis and word scoring in Natural Language Processing approaches using machine learning techniques (NLP). The TF-IDF format was created for document search and retrieval. It works by increasing in proportion to the number of times a word appears in a document, while decreasing in proportion to the number of sheets that include the term. As a result, words like this, what, and if, which appear frequently in all documents, rank low since they don't mean much to that document in particular. However, if the word Bug appears frequently in one document but not in others, it is likely to be very relevant. If we're trying to figure out which themes some NPS replies belong to, the term Bug, for example, will almost certainly be associated with the topic Reliability, because most responses including that word will be about that topic.

### Pearson's Correlation

The Pearson correlation coefficient is a measure of linear correlation between two sets of data in statistics. It is also known as Pearson's r, the Pearson product-moment correlation coefficient (PPMCC), the bivariate correlation, or simply the correlation coefficient. It is the ratio of two variables' covariance to the product of their standard deviations; consequently, it is effectively a normalized measurement of covariance, with the result always falling between 1 and 1. The measure, like covariance, can only show a linear correlation of variables and eliminates many other forms of interaction or linkage. The age and height of a group of high school students, for example, should have a Pearson correlation value that is considerably more than 0, but less than 1. (Because 1 would be an implausibly perfect connection).

The Pearson correlation technique, which assigns a value between 0 and 1, with 0 indicating no correlation, 1 representing entire positive correlation, and 1 representing total negative correlation, is the most generally used methodology for numerical variables. This means that a correlation value of 0.7 between two variables indicates that there is a significant and positive association between them. A positive correlation indicates that if variable A rises, so will variable B, whereas a negative correlation indicates that if A rises, so will B.

### SVD

Singular value decomposition (SVD) is a collaborative movie suggestion filtering method. The goal of the code implementation is to provide consumers with movie recommendations based on item-user matrices' latent properties. The Singular Value Decomposition (SVD) is a linear algebra method that has been widely employed in machine learning as a dimensionality reduction tool. The SVD is a collaborative filtering approach that is utilized in the recommender system. It is organized as a matrix, with each row representing a user and each column representing an object. The ratings that users give to items are the constituents of this matrix. The singular value decomposition is used to factorize this matrix. From the factorization of a high-level (user-item-rating) matrix, it determines matrices' factors.

## Conclusion

By employing several algorithms to select movies, I was able to meet the project's objectives. I've discussed the several algorithms that makeup Netflix's recommender system, as well as the steps I take to improve it and some of our current difficulties. Humans are faced with an increasing number of choices in every aspect of their lives, including media such as videos, music, and books, as well as other taste-based questions such as vacation rentals, restaurants, and so on, but also in areas such as health insurance plans, treatments, and tests, job searches, education and learning, dating and finding life partners, and many other areas where choice is important. I am convinced that recommender systems will continue to play an important part in using the massive amounts of data now accessible to make these decisions more manageable, successfully guiding people to the truly best few options for them to assess, resulting in more informed judgments. I also believe that recommender systems can democratize access to long-tail products, services, and information since machines can learn from far bigger data sets than experts, allowing them to make effective predictions in areas where human capacity is simply insufficient to generalize usefully.
