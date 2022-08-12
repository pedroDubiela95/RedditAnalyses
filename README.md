# RedditAnalyses
<p>RedditAnalyses is a Python language package for processing and analyzing posts from the Reddit website https://www.reddit.com/.
</p>
<p>The package works with text classification from supervised learning. With this package it is possible to extract (webscrapping), analyze and classify the posts. From the trained model, it is possible to classify new posts, as long as they refer to the same subjects that were used to train the model.</p>
<p>
The package was built in such a way to allow the construction of 3 models simultaneously, which are KNN, Random Forest and Logistic Regression with Cross-Validation
</p>

## Using <RedditAnalyses>
Module Import
```
from RedditAnalyses.redditpipeline import *
```

Using the API provided by Reddit itself to perform web scraping
```
client_id     = "**************"
client_secret = "**************"
password      = "**************"
username      = "**************"
user_agent    = "**************" 
topic         = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']
num_chars_min = 100
limit_posts   = 1000
reddit = RedditAnalyses(client_id, client_secret, password, username, user_agent, num_chars_min, topic)
```
Run the pipeline
```
# Load
classification = reddit.load_from_reddit()

# View
reddit.plot_distribution()

# Split the data into training and testing
TEST_SIZE                        = .2 
X_train, X_test, y_train, y_test = reddit.split_data(TEST_SIZE)

# Preprocessing NLP
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER       = 30
reddit.pipeline_NLP(MIN_DOC_FREQ, N_COMPONENTS, N_ITER)

# Create Models
N_NEIGHBORS = 4
CV          = 3
reddit.create_models(N_NEIGHBORS, CV)

# Train and evaluate
reddit.train_evaluate(X_train, X_test, y_train, y_test)

# Plot
reddit.plot_confusion(y_test)

# Predict
X = ["One of the most common uses for Python is in its ability to create and manage data structures quickly ", "Naive Bayes and Gradient Descent", "Scorpio and Pisces"]
yp = reddit.predict(X)

# Only Logistic Regression (the best model)
df = pd.DataFrame(yp[2][1])

# Fix result
lab_num = [(int(np.unique(classification[item][1])), item) for item in classification.keys()]

for item in lab_num:
    df.loc[df["predictions"] == item[0], "predictions"] = item[1] 

print(df)
```

## Contact
Pedro Gasparine Dubiela <br>
pedrodubielabio@mgmail.com 