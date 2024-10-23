import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import nbformat
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')



def load_data(database_filepath):
    
    """
    Load data from a SQLite database and prepare the feature and target datasets.

    This function connects to the specified SQLite database, retrieves the data 
    from the 'categorized_messages' table, and separates it into features and 
    labels for further processing.

    Parameters:
    ----------
    database_filepath : str
        The file path of the SQLite database containing the data.

    Returns:
    -------
    X : pandas.DataFrame
        A DataFrame containing the input features (messages).
    
    Y : pandas.DataFrame
        A DataFrame containing the output labels (multi-label binary indicators).
    
    category_names : pandas.Index
        An index object containing the names of the output categories (labels).
    """
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('categorized_messages', con=engine)
    
    X = df[['message']]
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    
    category_names = Y.columns
    
    X = X.values.ravel()
    Y = Y.values
    
    return X, Y, category_names



def tokenize(text):
    """
    Input: Array of Text
    
    Operations:
    1. Remove non-alphanumeric characters
    2. make all text lower case
    3. remove stop words
    4. lemmatize text
    """


    # Normalize text (convert to lowercase and remove punctuation)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return clean_tokens


def build_model(X_train, X_test, Y_train, Y_test):
    
    """
    Build and initialize a machine learning model pipeline for multi-label 
    classification.

    This function prepares the training and testing datasets, and constructs a 
    machine learning pipeline that includes a count vectorizer with custom tokenization,
    TF-IDF transformation and a multi-output classifier using RandomForest.

    Returns:
    -------
    model : sklearn.pipeline.Pipeline
      
    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # Vectorizer using the custom tokenizer)
    ('tfidf', TfidfTransformer()),  # Tf-Idf transformer
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])  # Multi-label classifier

    parameters = {
    'clf__estimator__n_estimators': [100, 200],
    'clf__estimator__max_depth': [None, 10, 20],
    'clf__estimator__min_samples_split': [2, 5],
}


    cv = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=5, n_jobs=-1)

    cv.fit(X_train. Y_train)

    model = cv.best_estimator_
    
    return model
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a trained machine learning model using a test dataset.

    This function makes predictions using the provided model on the test 
    dataset and prints the classification report for each output category.

    Parameters:
    ----------
    model : object
    The trained machine learning model to be evaluated. It should have 
    a `predict` method that can accept the test feature set.

    X_test : array-like or DataFrame
    The feature set used for testing. It should have the same number 
    of features as the training dataset used to fit the model.

    Y_test : array-like or DataFrame
    The true labels for the test dataset. This should be a one-hot encoded 
    DataFrame or array with the same number of samples as `X_test`.

    category_names : list of str A list containing the names of the output 
    categories. The order of the names should match the columns of `Y_test`.

    Returns:
    -------
    None
    This function does not return any value. It prints the classification 
    reports directly to the console for each category.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f"Classification report for label: {category}")
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
    
    
    
def save_model(model, model_filepath):
    """
    Save the given model to a pickle file.

    Parameters:
    model: The model object to be saved.
    filepath: The name of the file where the model will be saved.

    Returns:
    str: A message indicating the model has been saved.
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    return f"Model saved as {model_filepath}"


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, X_test, Y_train, Y_test)
        
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