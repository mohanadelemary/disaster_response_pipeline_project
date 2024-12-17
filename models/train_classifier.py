import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning
import pickle
import re



def load_data(database_filepath):
    
    """
    Load data from an SQLite database and prepare the feature and target datasets.

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
    Y = df.drop(columns='message')
    
    category_names = Y.columns
    
    X = X.values.ravel()
    Y = Y.values
    
    return X, Y, category_names


def simple_tokenizer(text):
    # Normalize and split on whitespace
    return re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()


def build_model(X_train, X_test, Y_train, Y_test):
    
    """
    Build and initialize a machine learning model pipeline for multi-label 
    classification.

    This function prepares the training and testing datasets, and constructs a 
    machine learning pipeline that includes a Tokenization + TF-IDF vectorization
    and a multi-output classifier using RandomForest.

    Returns:
    -------
    model : sklearn.pipeline.Pipeline
      
    """


    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=simple_tokenizer)),  # Tokenization + TF-IDF vectorization
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])  # Multi-label classifier

    parameters = {
    'clf__estimator__n_estimators': [100, 200],
    'clf__estimator__max_depth': [None, 10],
    'clf__estimator__min_samples_split': [2, 5]}


    cv = GridSearchCV(pipeline, parameters, scoring='f1_weighted', cv=3, verbose=2,n_jobs=1)

    cv.fit(X_train, Y_train)

    model = cv.best_estimator_
    
    return model
    
    
def evaluate_model(model, X_test, Y_test):
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
    This function does not return any value. It prints the average weighted scores
    directly to the console (F1, precision, recall).
    """
   
   # Initialize lists to store metrics for each label
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Ensure Y_test and Y_pred are NumPy arrays
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)

    # Iterate over each label
    for i in range(Y_test.shape[1]):
        f1 = f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        precision = precision_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        recall = recall_score(Y_test[:, i], Y_pred[:, i], average='weighted')

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Aggregate metrics across all labels
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)

    print(f"Average weighted F1 Score: {average_f1:.2f}")
    print(f"Average weighted Precision: {average_precision:.2f}")
    print(f"Average weighted Recall: {average_recall:.2f}")

   return None
    
    
    
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
        
        print('Building & Training model...')
        model = build_model(X_train, X_test, Y_train, Y_test)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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