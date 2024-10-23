import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Load messages dataset
    
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, how='outer', on='id')

    return df

def clean_data(df):
    
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove irregular values in label column "Related"
    df = df[df['related'] != 2]

    
    return df

def save_data(df, database_filename):
    # Save the clean dataset into an sqlite database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('categorized_messages', engine, index=False, if_exists='replace')  


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