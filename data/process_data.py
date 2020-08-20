import pandas as pd
import numpy as np
import re
import sqlite3
import nltk
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Takes messages and categories as two CSV files, converts them to Dataframes and merge them together
    INPUT:
    messages_file_path: path of messages CSV file
    categories_file_path: path of categories CSV file
    OUTPUT:
    merged_df pandas_dataframe: resulting Dataframe 
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'], how='outer')
    return df

def clean_data(df):
    """
    Cleans the data from Dataframe for futher processing in Maching Learning:
    Creates separate columns for each categorie and converts categories values to integer.
    Drops raws with Nan values if there are some, drops duplicates, drops raws if values of categories are not 0 or 1.
    INPUT:
    df : merged dataframe returned from load_data() function
    OUTPUT:
    df : the resulting Dataframe after cleaning
    """

    # create a dataframe of the 36 individual category columns
    categories=df.categories.str.split(';',expand=True)
    cl=[re.sub('-.', '', cat) for cat in df['categories'][0].split(';')]
    categories.columns=cl
    
    # converts categories values to single integers
    for column in categories:
        # set each value to be the last character of the string
        categories[column]=categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column]=pd.to_numeric(categories[column], downcast='integer')
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop rows with NaN in messages column and print the result
    n=df.message.isna().sum()
    p=n/len(df.message)
    df=df.dropna(subset=['message'])
    print(n, ' rows were deleted because they didnt have any messages.', '\n', 'It is ', p, ' % from the dataset.');
    
    # drop rows with duplicate IDs and print the result
    n=df.id.duplicated().sum()
    p=n/len(df.id)
    df=df.drop_duplicates(subset='id',keep=False)
    print(n, ' rows were deleted because they contained similar messages.', '\n', 'It is ', p, ' % from the dataset.');
    
    # delete rows if they contain categories values not equal 0 or 1 and print the result
    n=df.iloc[:,4:].shape[0]
    for col in df.iloc[:,4:]:
        df=df[df[col].isin([0,1])]
    n-=df.shape[0]
    p=n/df.shape[0]
    print(n, ' rows were deleted because they contained categories values not equal 0 or 1.', '\n', 'It is ', "{:.2%}".format(p), 'from the dataset.');
    return df
        
    
def save_data(df, database_filename):
    """
    Saves cleaned data to an SQL database
    INPUT:
    df : cleaned data returned from clean_data() function
    database_file_name : file path of SQL Database where cleaned data is going to be saved
    OUTPUT:
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    db_name=re.sub(r'(.+/|.db)', '', database_filename)
    print('db_name', db_name, 'database_filename', database_filename)
    df.to_sql(db_name, engine, index=False, if_exists = 'replace')  


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
