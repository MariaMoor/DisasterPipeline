# DisasterPipeline
<!-- blank line -->
----
<!-- blank line -->


## Table of Contents:
1. [ Project Overview. ](#prov)
2. [ Installations. ](#instal)
3. [ Files Descriptions. ](#fd)
4. [ Model explanation. ](#me)
5. [ Instructions. ](#instr)
6. [Licensing, Authors, Acknowledgements](#li)


<a name="prov"></a>
## Project Overview:

In this project I am working with a dataset of real messages that were sent during disaster events, provided by [Figure Eight](https://www.figure-eight.com).
I am creating a machine learning pipeline to categorize events so that the messages can be sent to an appropriate disaster relief agency.
My project also includes a web app where an emergency worker can input a new message and get classification results in several categories.
The web app also displays visualizations of the data. 


<a name="instal"></a>
## Installations:

I did my project on the 3.6.3  version of Python and used the following libraries:

re<br />
sys<br />
sqlchemy<br />
sqlite3<br />
pandas<br />
numpy<br />
nltk<br />
sklearn<br />
pickle<br />
flask SQLAlchemy<br />
plotly<br />
Matplotlib<br />


<a name="fd"></a>
## Files Descriptions:

messages.csv - a csv file with the messages (one column should be in English)<br />
categories.csv - a csv file containing a training sample of categorized messages


<a name="me"></a>
## Model explanation:

**ETL**

The ETL script, process_data.py, takes the file paths of the two datasets (messanges and categories) and SQL database.
It cleans the datasets, merges them, splits the categories column into separate columns, converts values to binary, and drops duplicates.
Then it stores the clean data into a SQLite database in the specified database file path.

**Machine Learning**

The machine learning script, train_classifier.py takes the data from the SQL database, creates and trains a multi-output classifier on the 36 categories in the dataset.
This is done using the machine learning pipeline to first vectorize and apply TF-IDF to the text and then train the model.
A custom tokenize function is used to case normalize, lemmatize, and tokenize textin the estimator.
GridSearchCV is used to find the best parameters for the model. 
The f1 score, precision and recall for the test set is outputted for each category.

<a name="instr"></a>
## Instructions:

There are 3 steps to be executed to run the project:

1. Cleaning Data
to run the data you need to go to the project directory and run the folliwing command:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db


<a name="li"></a>
## Licensing, Authors, Acknowledgements

I would like to give a credit to Figure Eight Machine intelligence company for providing original dataset of Disaster Response Messages.
