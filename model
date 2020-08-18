import pandas as pd
import numpy as np
import re
import sqlite3
import nltk


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge messanges and categories datasets
    df = messages.merge(categories, on=['id'], how='outer')
    return df

def clean_data(df):
    '''
    Function to clean the data
    Input: Dataframe
    output: cleaned Dataframe
    '''

    #split values in categories column in separate columns 
    categories=df.categories.str.split(';',expand=True)
   
    #extract names of the columns from the values and save a categories DataFrame
    cl=[re.sub('-.', '', cat) for cat in df['categories'][0].split(';')]
    categories.columns=cl
    
    for column in categories:
    # set each value to be the last character of the string
    categories[column]=categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column]=pd.to_numeric(categories[column], downcast='integer')
    
    # drop the original categories column from Dataframe
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop rows with duplicate IDs 
    df.drop_duplicates(subset='id',keep=False)
    
    return df
    

def save_data(df, database_filename):
    from sqlalchemy import create_engine
    engine = create_engine(database_filename)
    df.to_sql(re.sub(r'(.+///|.db)', '', database_filename), engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
