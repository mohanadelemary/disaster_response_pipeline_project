# Disaster Response Pipeline Project

This project develops a web app with an ETL and ML pipeline in the backend. The app trains on pre-labeled text (messages and tweets) to determine the correct naturak disaster responses required based on citizens text data. The app itself serves as an input interface to analyze and classify new text.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Repository Structure

disaster_response_pipeline_project/  
│
├── app/                               # Flask web app files
│   ├── run.py                         # Script to run the web application
│   └── templates/                     # HTML templates for the web app
│       ├── go.html                    # Result page for classification query
│       └── master.html                # Main page of the web app with visuals
│
├── data/                              # Data processing files and raw datasets
│   ├── disaster_categories.csv        # Dataset containing message categories
│   ├── disaster_messages.csv          # Dataset containing disaster messages
│   └── process_data.py                # Script to clean and process the data
│
├── models/                            # Model training and saving files
│   └── train_classifier.py            # Script to train the machine learning model
│
├── .git/                              # Git-related files and folders
│
├── README.md                          # ReadMe file with project documentation
│
