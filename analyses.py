from RedditAnalyses.redditpipeline import *

if __name__ == "__main__":

    client_id     = "**************"
    client_secret = "**************"
    password      = "**************"
    username      = "**************"
    user_agent    = "**************" 
    topic         = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']
    num_chars_min = 100
    limit_posts   = 1000
    reddit = RedditAnalyses(client_id, client_secret, password, username, user_agent, num_chars_min, topic)
    
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
    X = ["Python is very used", "Naive Bayes", "Cancer"]
    yp = reddit.predict(X)
   
    # Only Logistic Regression (the best model)
    df = pd.DataFrame(yp[2][1])
    
    # Fix result
    lab_num = [(int(np.unique(classification[item][1])), item) for item in classification.keys()]

    for item in lab_num:
        df.loc[df["predictions"] == item[0], "predictions"] = item[1] 
    
    print(df)
 











