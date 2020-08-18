# import libraries
import sqlite3
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


# In[5]:


# load data from database
engine = create_engine('sqlite:///DisasterPipeline.db')
df = pd.read_sql('SELECT * FROM DisasterPipeline', engine)
X = df.message.values
Y_df = df.drop(columns=['id', 'message', 'original', 'genre'])

# transform 'genre' data from categorical type to integer
le = LabelEncoder()
y1=list(le.fit_transform(df.genre))
y1=pd.Series(y1, name='genre')


Y_df = pd.concat([y1, Y_df], axis=1) #create a Dataframe of categories
Y=Y_df[['genre', 'related', 'request']].values # take only first 3 categories from Dataframe and convert them to array


# ### 2. Write a tokenization function to process your text data
def tokenize(text):
    '''
    Function to tokenize the text messages
    Input: text
    output: cleaned tokenized text as a list object
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset.
# You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful
# for predicting multiple target variables.

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
])


# ### 4. Train pipeline

# - Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# - Train pipeline and get predicted values
pipeline.fit(X_train, y_train)


y_pred=pipeline.predict(X_test)

#Score pipeline for the first createria

score=pipeline.score(X_test, y_test)
from sklearn.metrics import confusion_matrix
cm_y1 = confusion_matrix(y_test[:,0],y_pred[:,0])


print(score)
print(cm_y1)



# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

from sklearn.metrics import multilabel_confusion_matrix
mmcm=multilabel_confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['genre', 'related', 'request']))


# ### 6. Improve your model
# Use grid search to find better parameters. 

parameters = 
cv = 


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass.
# However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF


# ### 9. Export your model as a pickle file


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

