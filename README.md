# DisasterPipeline
<!-- blank line -->
----
<!-- blank line -->


## Table of Contents:
1. [ Installations. ](#instal)
2. [ Project Overview. ](#prov)
3. [ Files Descriptions. ](#fd)
4. [ Instructions. ](#instr)
5. [ Model explanation. ](#me)


<a name="prov"></a>
## Project Overview:

In this project I am working with a dataset of real messages that were sent during disaster events, provided by [Figure Eight](https://www.figure-eight.com)
A machine learning pipeline to categorize events so that the messages can be sent to an appropriate disaster relief agency.

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
**Files Descriptions:**

messages.csv - a csv file with the messages (one column should be in English)<br />
categories.csv - a csv file containing a training sample of categorized messages


<a name="me"></a>
**Model explanation:**

ETL

The ETL script, process_data.py, takes the file paths of the two datasets (messanges and categories) and SQL database.
It cleans the datasets, merges them, splits the categories column into separate columns, converts values to binary, and drops duplicates.
Then it stores the clean data into a SQLite database in the specified database file path.

Machine Learning

The machine learning script, train_classifier.py takes the data from the SQL database, creates and trains a multi-output classifier on the 36 categories in the dataset.
This is done using the machine learning pipeline to first vectorize and apply TF-IDF to the text and then train the model.
A custom tokenize function is used to case normalize, lemmatize, and tokenize textin the estimator.
GridSearchCV is used to find the best parameters for the model. 
The f1 score, precision and recall for the test set is outputted for each category.

<a name="instr"></a>
**Instructions:**


**Results:**


**Licensing, Authors, Acknowledgements**

I would like to give a credit to Figure Eight Machine intelligence company for providing original dataset of Disaster Response Messages.
