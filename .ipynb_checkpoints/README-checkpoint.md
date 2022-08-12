# RedditAnalyses
<p>RedditAnalyses é um pacote em linguagem Python para processamento e análise de posts provenientes do site Reddit https://www.reddit.com/ .
</p>
<p>O pacote trabalha com classificação de texto a partir de aprendizado supervisionado. Com esse pacote é possível extrair (webscrapping), analisar e classificar os posts. A partir do modelo treinado, é possível classificar novos posts, desde que sejam referentes ao mesmos assuntos que foram utilizados para treinar o modelo.</p>
<p>
O pacote foi construido de tal forma a permitir a construção de 3 modelos simultânemante, os quais são KNN, Random Forest e Logistic Regression com Cross-Validation
</p>

## Using <RedditAnalyses>
Importando o módulo
```
from RedditAnalyses.redditpipeline import *
```

Utilizando a API disponibilizada pelo próprio Reddit para realizar webscrapping
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
Executando o pipeline
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