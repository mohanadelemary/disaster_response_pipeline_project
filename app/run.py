import json
import re
import sys

from flask import Flask, jsonify, render_template, request
import plotly
from plotly.graph_objs import Bar
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

import joblib
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)


def tokenize(text):
    # Normalize and split on whitespace
    return re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    graph = {
        'data': [Bar(x=category_names, y=category_counts)],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {'title': "Count"},
            'xaxis': {'title': "Category", 'tickangle': -45},
        },
    }

    graphJSON = json.dumps([graph], cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html', graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()