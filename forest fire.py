#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd


# In[25]:


df = pd.read_csv('C:\\Users\\shubhanshikat\\Downloads\\forestfires.csv')


# In[26]:


#the number of rows and columns in the data frame.


# In[27]:


df.shape


# In[28]:


#the first 5 rows or a random sample with 5 rows


# In[29]:


df.head()


# In[73]:


fires = pd.read_csv('C:\\Users\\shubhanshikat\\Downloads\\forestfires.csv')
fires['areaclass'] = [0 if val==0.0 else 1 for val in fires['area']]
fires.head()


# In[72]:


pd.DataFrame({'Column Type':df.dtypes, 'Memory Usage': df.memory_usage(deep=True)}) 


# In[32]:


# counts how many null values in each column.


# In[33]:


df.describe()


# In[ ]:


#linear regression


# In[78]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# considering only relevant columns
linearcols = ['month','temp','RH','wind','rain','FFMC','DMC','DC','ISI']
datafires = fires[linearcols]

# label encoding for 'month' column
le = LabelEncoder()
xdata = datafires.apply(le.fit_transform)
xreg = xdata


# In[79]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score

yreg = fires['area']

xtrain, xtest, ytrain, ytest = train_test_split(xreg,yreg)

#fittiing the model
regressor = LinearRegression(fit_intercept=False)
regressor.fit(xtrain,ytrain)
yregpred = regressor.predict(xtest)

#results
print('Coefficient of determination r^2: %.2f' % r2_score(ytest,yregpred))
print('RMSE: %.2f' % mean_squared_error(ytest,yregpred,squared=False))


# In[80]:


lrresults = pd.DataFrame(yregpred,ytest)
lrresults.reset_index()


# In[ ]:





# In[70]:


#During which months are forest fires most common?


# In[ ]:





# In[35]:


import matplotlib.pyplot as plt
# Group the data by month and count the number of fires in each month
fire_in_months = df.groupby('month').size().reset_index(name='count')

# Create a bar plot of the number of fires in each month
plt.bar(fire_in_months['month'], fire_in_months['count'])
plt.title('Number of Fires Occurred in Each Month')
plt.xlabel('Month')
plt.ylabel('Number of Fires')
plt.show()


# In[39]:


#Number of forest fires by a week-day


# In[40]:


fire_in_days = df.groupby('day').size().reset_index(name='count_weekday')

# Create a bar plot of the number of fires on each day of the week
plt.bar(fire_in_days['day'], fire_in_days['count_weekday'])
plt.title('Number of Fires Occurred on Each Day')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Fires')
plt.show()


# In[41]:


#What are the causes?


# In[42]:


#Box plots of independent var. by month


# In[43]:



import seaborn as sns
def create_box_by_month(x, y):
    sns.boxplot(x=x, y=y, data=df)
    plt.title(f"{y} Distribution by {x}")
    plt.show()

# Define the x and y variables to create box plots for
x_var = "month"
y_var = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]

# Create a box plot for each combination of x and y variable
for y in y_var:
    create_box_by_month(x_var, y)


# In[44]:


#Box Plots of independent var. by week day


# In[45]:


def create_box_by_day(x, y):
    sns.boxplot(x=x, y=y, data=df)
    plt.title(f"{y} Distribution by {x}")
    plt.show()

# Define the x and y variables to create box plots for
x_var = "day"
y_var = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]

# Create a box plot for each combination of x and y variable
for y in y_var:
    create_box_by_day(x_var, y)


# In[46]:


#Which variables are related to forest fire severity?


# In[47]:


def create_scatter(x, y):
    sns.scatterplot(x=x, y=y, data=df)
    plt.title(f"{y} vs {x}")
    plt.show()

# Define the x and y variables to create scatter plots for
x_var = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
y_var = ["area"]

# Create a scatter plot for each combination of x and y variable
for x in x_var:
    for y in y_var:
        create_scatter(x, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




