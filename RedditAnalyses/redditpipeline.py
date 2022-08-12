"""
    Text Classifier with Supervised Learning
    Extracting, Analyzing and Classifying topics on Reddit forum

    @author Pedro G. Dubiela
"""

import re     
import praw   
import config 
import pandas                          as     pd
import numpy                           as     np 
from   sklearn.model_selection         import train_test_split
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.decomposition           import TruncatedSVD
from   sklearn.neighbors               import KNeighborsClassifier
from   sklearn.ensemble                import RandomForestClassifier
from   sklearn.linear_model            import LogisticRegressionCV
from   sklearn.metrics                 import classification_report
from   sklearn.pipeline                import Pipeline
from   sklearn.metrics                 import confusion_matrix
import matplotlib.pyplot               as     plt
import seaborn                         as     sns


class RedditAnalyses:
    """
    A class to represent a RedditAnalyses.
    ...
    
    Attributes
    ----------
    client_id : str
        Client identifier.
    client_secret : str
        Client secret.
    password : str
        Client password.
    username : str
        Client username.
    user_agent : str
        Nickname.
    num_chars_min : int
        Minimun number of chars by posts.
    subject : list
        Subjects.
    limit_posts : int, default = 1000
        Maximum number of posts by subject.
    random_state : int, default = 0.
        This is necessary for performs the splitting with the same pattern.
    data : list
        A list with all the data obtained from the webscrapping.
    labels : list
        Labels for the data.
    
   """
  
    # Constructor
    def __init__(self, client_id, client_secret, password, username, 
                 user_agent, num_chars_min, topic, limit_posts = 1000, random_state = 0):
        """
        This method constructs all the necessary attributes for the RedditAnalyses object.

        Parameters
        ----------
        client_id : str
            Client identifier.
        client_secret : str
            Client secret.
        password : str
            Client password.
        username : str
            Client username.
        user_agent : str
            Nickname.
        num_chars_min : int
            Minimun number of chars by posts.
        subject : list
            Subjects.
        limit_posts : int, default = 1000
            Maximum number of posts by subject.
        random_state : int, default = 0.
            This is necessary for performs the splitting with the same pattern.

        """
        
        self.__client_id     = client_id
        self.__client_secret = client_secret
        self.__password      = password
        self.__username      = username
        self.__user_agent    = user_agent
        self.__num_chars_min = num_chars_min
        self.__topic         = topic
        self.__limit_posts   = limit_posts
        self.__random_state  = random_state 
        self.__data          = None
        self.__label         = None
        
    # Webscrapping
    def load_from_reddit(self):
        """
        This method performs the webscrapping from Reddit. It loads and creates the labels for each
        topic passed by user.
    
        Returns
        -------
        None : tuple, length = 2
            Tuple with topics (data) and their respective labels
        """
        
        # Open connection
        api_reddit = praw.Reddit(
            client_id     = self.__client_id,
            client_secret = self.__client_secret,
            password      = self.__password,
            user_agent    = self.__user_agent,
            username      = self.__username
        )
     
     
        # Where the results will be storage
        data   = []
        labels = []
        classification = {}
     
        for i, subject in enumerate(self.__topic):
            
            n_chars = lambda post : len(re.sub(pattern = '\W|\d', repl = '', string = post.selftext))
            mask    = lambda post : n_chars(post) >= self.__num_chars_min
            
            # Posts extraction
            submissions = api_reddit.subreddit(subject).new(limit = self.__limit_posts)
            
            # Posts filter
            posts = [post.selftext for post in filter(mask, submissions)]
            
            classification[subject] = (posts, [i] * len(posts))
            data.extend(classification[subject][0])
            labels.extend(classification[subject][1])
        
        self.__data, self.__labels = data, labels
        print("Operation perfomed successfully")
        return classification
        
     
    # Split data into training and testing
    def split_data(self, TEST_SIZE):
        """
        This method performs the data splitting into train and test.
        
        Parameters
        ----------
        TEST_SIZE : float
            Size of test data.
     
        Returns
        -------
        X_train : list
            Train data.
        X_test : list
            Test data.
        y_train : list
            Train label.
        y_test : list
            Test label.
     
        """
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.__data, 
            self.__labels, 
            test_size = TEST_SIZE, 
            random_state = self.__random_state
        )
        
        return X_train, X_test, y_train, y_test 
     
     
    # Create Natural Language Processing Pipeline
    def pipeline_NLP(self, MIN_DOC_FREQ, N_COMPONENTS, N_ITER):
        """
        This method creates the Natural Language Processing Pipeline.
     
        Parameters
        ----------
        MIN_DOC_FREQ : int
            Mininum number of words.
        N_COMPONENTS : int
            Desired dimensionality of output data.
        N_ITER : int
            Number of iterations for ramdomized SVD solver.
     
        """
        
        # Remove http
        pattern       = r'\W|\d|http.*\s+|www.*\s+'
        preprocessor  = lambda text : re.sub(pattern, ' ', text)
        
        # Vectorizer: TF - IDF
        vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words = 'english', min_df = MIN_DOC_FREQ)
        
        # Dimensionality reduction
        decomposition = TruncatedSVD(n_components = N_COMPONENTS, n_iter = N_ITER)
        
        # Pipeline
        self.__pipeline_NLP = [('tfidf', vectorizer), ('svd', decomposition)]
        
        print("Operation perfomed successfully")
        
     
    # Create models
    def create_models(self, N_NEIGHBORS = 4, CV = 4):
        """
        This method creates the models for classification.
        
        Parameters
        ----------
        N_NEIGHBORS : int, default = 4
            Number of neighbors (knn)
        CV : int, default = 3
            Cross-validation generator.
        """
        
        
        model1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
        model2 = RandomForestClassifier(random_state = self.__random_state)
        model3 = LogisticRegressionCV(cv = CV, random_state = self.__random_state)
        
        self.__models = [("KNN", model1), ("RF", model2), ("LRcv", model3)]
        print("Operation perfomed successfully")
         
        
    # Traine evaluate
    def train_evaluate(self, X_train, X_test, y_train, y_test):
        """
        This method train and evaluate the classification models.
        
        Parameters
        ----------
        X_train : list
            Train data.
        X_test : list
            Test data.
        y_train : list
            Train label.
        y_test : list
            Test label.

        Returns
        -------
        results : list
            A list filled by models and their respective dictionaries.
            The dictionaries are filled whithin information about models, 
            for example, test predictions and performance report. 
            
        """
        
        results = []
        for name, model in self.__models:
                
            pipe = Pipeline(steps = self.__pipeline_NLP + [(name, model)])
            
            # Train
            pipe.fit(X_train, y_train)
            
            # Predict
            y_predict = pipe.predict(X_test)
            
            # Evaluate
            report = classification_report(y_true = y_test, y_pred = y_predict)
            
            # Storage
            results.append([model, {'model':name, 'predictions':y_predict, 'report':report}])  
            
        self.__results = results
        
        
    # Distribution of topics    
    def plot_distribution(self):
        """
        This method creates a bar plot showing the frequency distribution of topics
        
        """
        _, counts = np.unique(self.__labels, return_counts = True)
        sns.set_theme(style = "whitegrid")
        plt.figure(figsize = (15, 6), dpi = 120)
        plt.title("Number of Posts by Topics")
        sns.barplot(x = self.__topic, y = counts)
        plt.legend([' '.join([f.title(),f"- {c} posts"]) for f,c in zip(self.__topic, counts)])
        plt.xticks(rotation = 45)
        plt.show()
        
    
    # Models performance
    def plot_confusion(self, y_test):
        """
        This method create a cofusion matrix by each model

        Parameters
        ----------
        y_test : list
            The test data.

        """
        for result in self.__results:        
        
            print("Classiication Report\n", result[-1]['report'])
            
            y_pred              = result[-1]['predictions']
            conf_matrix         = confusion_matrix(y_test, y_pred)
            _, test_counts      = np.unique(y_test, return_counts = True)
            conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
            
            plt.figure(figsize = (9,8), dpi = 120)
            plt.title(result[-1]['model'].upper() + " Results")
            plt.xlabel("Truth Value")
            plt.ylabel("Modelo Prediction")
            ticklabels = [f"r/{sub}" for sub in self.__topic]
            sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f')
            plt.show()
    
    
    # Predict
    def predict(self, X):
        
        results = []
        for name, model in self.__models:
                
            pipe = Pipeline(steps = self.__pipeline_NLP + [(name, model)])
            
            # Predict
            y_predict = pipe.predict(X)
            
            # Storage
            results.append([model, {'model':name, 'input':X, 'predictions':y_predict}])  
            
        return results
        

