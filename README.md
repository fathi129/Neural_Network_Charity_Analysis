# Neural_Network_Charity_Analysis
A deep learning neural network model capable of predicting whether applicants will be successful if funded by Alphabet Soup is created.

## Overview of the Analysis
A neural network is a powerful machine learning technique modeled after neurons in the brain. It can rival the performance of the most robust statistical algorithms without having a statistical theory. Neural networks are used for analyzing images and Natural Language Processing datasets. It gives robust deep learning models for complex and irregular data. We will explore and implement neural network models using Python's tensor flow library.

## Purpose of the Analysis
This analysis aims to analyze the impact of each donation and potential recipients of the Alphabet Soup company. This help's to ensure that the foundation money is being used effectively. Unfortunately, not every donation the company makes is impactful. In some cases, the organization will take the money and disappear. We create a mathematical data-driven solution to predict which organizations are worth donating to and which are at high risk. We will design and train a deep learning neural network model that evaluates all input data types and produces a clear decision-making result. We will use the features in the provided dataset to create a binary classifier capable of predicting whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are some columns that capture metadata about each organization, such as the following:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Resources Used
*DataSources*:  [charity_data.csv](https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Resources/charity_data.csv)<br>
*Software used*: Jupyter Notebook <br>
*Language*: Python<br>
*Libraries*: tensor-flow, scikit-learn. <br>

## Results
## Deliverable 1: Preprocessing Data for a Neural Network Model
Using the knowledge of Pandas and the Scikit-Learn’s StandardScaler(), we will preprocess the dataset to compile, train, and evaluate the neural network model.Read in the charity_data.csv to a Pandas DataFrame, and define taget and feature variables for the model. Drop the EIN and NAME columns. Determine the number of unique values for each column. For those columns that have more than ten unique values, determine the number of data points for each unique value. Create a density plot to determine the distribution of the column values. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.<br>
The density plot for application count is calculated, and count less than 500 is binned as others. 
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/application_count_plot.png"  width = 600><br>
The density plot for classification count is calculated, and count < 1000 is binned as others.<br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/classification_count.png"  width = 600><br>
Generate a list of categorical variables. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame. 
Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals. At this point, the merged DataFrame should look like this:<br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/Merged_df.png"  width = 900><br>
Split the preprocessed data into features and target arrays. Split the preprocessed data into training and testing datasets. Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.

## Deliverable 2: Compile, Train, and Evaluate the Model
Using the knowledge of TensorFlow, we will design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. We will need to consider how many inputs there are before determining the number of neurons and layers in our model. Once we have completed that step, we will compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy. After preprocessing the data, Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras. Create the first hidden layer and choose an appropriate activation function as relu. Create the second hidden layer and choose an appropriate activation function as relu. Create an output layer with an appropriate activation function as sigmoid. Check the structure of the model.<br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/Del2_Model.png"  width = 900><br>
Compile and train the model. Create a callback that saves the model's weights every 5 epochs. Evaluate the model using the test data to determine the loss and accuracy. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.<br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/Accuracy_del1.png"  width = 900><br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del2_Accuracy_plot.png"  width = 500><br>
We can see that the model produced an accuracy of 73% and a loss of 56%

## Deliverable 3: Optimize the Model
To optimize our model to achieve a target predictive accuracy higher than 75% by using any or all of the following:
Adjusting the input data to ensure that there are no variables or outliers that are confusing the model, such as:<br>
- Dropping more or fewer columns.
- Creating more bins for rare occurrences in columns.
- Increasing or decreasing the number of values for each bin.
- Adding more neurons to a hidden layer.
- Adding more hidden layers.
- Using different activation functions for the hidden layers.
- Adding or reducing the number of epochs to the training regimen.<br>

### Attempt 1: Dropping EIN Column alone
By dropping EIN column alone. We will compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy. We have calculated the name count using the density plot.<br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_namecount_density.png"  width = 600><br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt1_accuracy.png"  width = 900><br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt1.png"  width = 600><br>
By adding the NAME column for the training dataset, we can see the model has accuracy of 79% and loss of 46%.<br>

### Attempt 2: Reducing the number of nodes
By reducing the number of neurons i.e. hidden nodes in layer one as 5 and hidden nodes in layer two as 4. we will compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy. 
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt2_accuracy.png"  width = 900><br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt2.png"  width = 600><br>
We can see that the model has produced an accuracy of 78% and loss of 45% by decreasing the number of neurons in both the layers.

### Attempt 3: Adding hidden layer
By adding the third hidden layer. The first hidden layer has input nodes of 4, The second hidden layer has input nodes of 3, and the third hidden layer has input nodes of 3. We will compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy.<br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt3_accuracy.png"  width = 900><br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt3.png"  width = 600><br>
We can see that the model has produced an accuracy of 76% and a loss of 48% by adding the third hidden layer.

### Attempt 4: Changing from Relu to Tanh activation
By changing the activation function from relu to tanh.we will compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy. 
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attemp4_accuracy.png"  width = 900><br>
<img src = "https://github.com/fathi129/Neural_Network_Charity_Analysis/blob/master/Screenshots%20of%20Neural%20n:w/del3_attempt4.png"  width = 600><br>
We can see that the model has produced an accuracy of 79% and a loss of 44% by changing the activation function from relu to tanh.

## Summary
By adding NAME column,reducing neurons,adding hidden layers and changing activation functions, the accuracy of the optimized model for predicting whether a successful donation ended up being 79% and its loss metric was 46%.<br> 
A random forest model is recommended, which could solve this classification problem by randomly sampling the preprocessed data. Some benefits of using a random forest model include its robustness, and it avoids overfitting of the data. Random forest models have been a staple in machine learning algorithms for many years due to their robustness and scalability. Both output and feature selection of random forest models are easy to interpret and can easily handle outliers and nonlinear data. The random forest model achieves comparable predictive accuracy on large tabular data with less code and faster performance.

