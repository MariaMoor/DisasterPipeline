import sys
import sqlite3
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Loads data from SQL Database
    INPUT:
    database_filepath: SQL database file
    OUTPUT:
    X : array of features
    Y : array of target values
    category_names list: labels for target values
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    db_name=re.sub(r'(.+/|.db)', '', database_filepath)
    df = pd.read_sql('SELECT * FROM %s' % (db_name), engine)
    X = df.message.values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns
    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize the text messages
    INPUT: text
    OUTPUT: cleaned tokenized text as a list object
    """
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


def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    INPUT None
    OUTPUT: Model
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4],
        #'clf__estimator__max_features':['sqrt', 0.25, 0.50, 0.75, 1.0],
        'clf__estimator__criterion':['gini', 'entropy'],
        'clf__estimator__bootstrap' : [True, False]
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model's performance
    INPUT:
    model:  model to train
    X_test: test features
    Y_test: test targets
    category_names: labels of target values
    OUTPUT: None
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test == y_pred)))


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    INPUT:
    model: trained model
    model_filepath: path where to save the model
    OUTPUT: None
    """

    # save the model to a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
