# Neural_Network_Charity_Analysis
A binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup is created.

## Overview of the Analysis





## Purpose of the Analysis
Beks has come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively


## Resources Used
*DataSources*:  [charity_data.csv](https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Resources/charity_data.csv)<br>
*Software used*: Jupyter Notebook <br>
*Language*: Python<br>
*Libraries*: tensor-flow,scikit-learn. <br>


## Results
## Deliverable 1: Preprocessing Data for a Neural Network Model
Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.
Follow the instructions below and use the AlphabetSoupCharity_starter_code.ipynb file to complete Deliverable 1.

Open the AlphabetSoupCharity_starter_code.ipynb file, rename it AlphabetSoupCharity.ipynb, and save it to your Neural_Network_Charity_Analysis GitHub folder.

Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are considered the target(s) for your model?
What variable(s) are considered the feature(s) for your model?
Drop the EIN and NAME columns.
Determine the number of unique values for each column.
For those columns that have more than 10 unique values, determine the number of data points for each unique value.
Create a density plot to determine the distribution of the column values.
Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
Generate a list of categorical variables.
Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
At this point, your merged DataFrame should look like this:

Split the preprocessed data into features and target arrays.
Split the preprocessed data into training and testing datasets.
Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.
Save your AlphabetSoupCharity.ipynb file to your Neural_Network_Charity_Analysis folder.



## Deliverable 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
Follow the instructions below and use the information file to complete Deliverable 2.

Continue using the AlphabetSoupCharity.ipynb file where you’ve already performed the preprocessing steps from Deliverable 1.
Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
Create the first hidden layer and choose an appropriate activation function.
If necessary, add a second hidden layer with an appropriate activation function.
Create an output layer with an appropriate activation function.
Check the structure of the model.
Compile and train the model.
Create a callback that saves the model's weights every 5 epochs.
Evaluate the model using the test data to determine the loss and accuracy.
Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.


## Deliverable 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Adding more neurons to a hidden layer.
Adding more hidden layers.
Using different activation functions for the hidden layers.
Adding or reducing the number of epochs to the training regimen.

### Attempt 1:Dropping EIN Column alone
### Attempt 2: Reducing the number of nodes
### Attempt 3: Adding hidden layer
### Attempt 4: Changing from Relu to Tanh activation

## Summary
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
