#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Importing Necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[32]:


#Importing training data
training_data = pd.read_csv('C:\\Users\\AishwaryaM\\Downloads\\BTC-USD Training Data - 1st Jan 2016 to 1st Jan 2022.csv')
training_set = training_data.iloc[:, 1:2].values


# In[33]:


#Assesing the headers
training_data.head()


# In[34]:


#Assesing basic information
training_data.info()


# In[35]:


training_set = training_data['Open']
training_set = pd.DataFrame(training_set)


# In[12]:


#Feature Scaling to transform data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
scaled_data = sc.fit_transform(training_set)


# In[16]:


#Creating a dataframe with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range (60,1258):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[1,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshaping
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# In[20]:


#Importing Keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[21]:


#Initialising RNN
regressor = Sequential()


# In[ ]:


#Adding four LSTM layers and Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape =(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True, input_shape =(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True, input_shape =(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[ ]:


#Adddding the output layer
regressor.add(Dense(units=1))


# In[ ]:


#Compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[ ]:


#Fitting RNN to the training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)


# In[28]:


#Importing test data
test_data = pd.read_csv('C:\\Users\\AishwaryaM\\Downloads\\BTC-USD Out of Time Testing 1st Jan 2022 to 4th Feb 2022.csv')
testing_set = test_data.iloc[:, 1:2]


# In[25]:


testing_set.head()


# In[29]:


testing_set.info()


# In[36]:


#Predicting Stock prices
complete_data = pd.concat((training_data['Open'], test_data['Open']), axis = 0)
inputs = complete_data[len(complete_data) - len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[ ]:


#Visualizing the predicted values and original values
plt.plot(predicted_stock_price, color ='red', label = 'predicted_stock_price')
plt.plot(testing_set, color ='blue', label = 'original_stock_price')
plt.title('BTC-USD Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()

