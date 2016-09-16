# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 01:45:49 2016

@author: Raleigh Littles, sourced from: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

A script for 'solving' the Titanic Kaggle challenge (see README for link).
"""
#-----------------------------------------#
# Part 1 -- Data Exploration/Data Analysis
#-----------------------------------------#

import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
import time

initial_time = time.time()

# Show 100 columns in pandas
pandas.options.display.max_columns = 100
matplotlib.style.use('ggplot')

training_data_path = "C:/users/ralei/Kaggle/Titanic/data/train_titanic.csv"
testing_data_path = training_data_path.replace('train', 'test')

data = pandas.read_csv(training_data_path, encoding = 'utf-8-sig')

print data.describe()

# Replace missing values in the 'Age' column with the median age.

data['Age'].fillna(data['Age'].median(), inplace=True)

# Plot survival based on gender as a stacked bar chart. Start by organizing data for those who
# survived and those who died.
survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
dead_sex = data[data['Survived'] == 0]['Sex'].value_counts()

survival_gender_dataframe = pandas.DataFrame([survived_sex, dead_sex])
survival_gender_dataframe.index = ["Survived", "Dead"]

survival_gender_dataframe.plot(kind = 'bar', stacked=True, figsize=(15,8))
print survival_gender_dataframe

# Now plot a stacked bar chart of survival based on age only.

figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, 
         color = ['g','r'], bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

# Now we'll analyze the fare price and survival.

figure = plt.figure(figsize=(18,5))
plt.hist([data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], stacked=True,
         color = ['g', 'r'], bins=30, label=['Survived', 'Died'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()

# Create a scatterplot comparing Age, Fare price, and Survival.
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'], data[data['Survived']==1]['Fare'], c = 'green', s =40)
ax.scatter(data[data['Survived']==0]['Age'], data[data['Survived']==0]['Fare'], c='r', s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')

ax.legend(('Survived', 'Dead'), scatterpoints=1, loc='upper right',fontsize=15)

plt.gcf().clear()  # Clear figure.

# Check the coorrelation of Fare with Class.

ax_2 = plt.subplot()
ax_2.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(15,8), ax =ax_2)


# Check how the embarkation site affects survival rate.

embarkment_survived = data[data['Survived']==1]['Embarked'].value_counts()
embarkment_dead = data[data['Survived']==0]['Embarked'].value_counts()

embarkment_survival_dataframe = pandas.DataFrame([embarkment_survived, embarkment_dead])
embarkment_survival_dataframe.index = ['Survived', 'Dead']
embarkment_survival_dataframe.plot(kind='bar', stacked=True, figsize=(15,8))
embarkment_survival_dataframe.index = ['Survived', 'Dead']
embarkment_survival_dataframe.plot(kind='bar', stacked=True, figsize=(15,8))

#-----------------------------------------#
# Part 2 -- Feature Engineering
#-----------------------------------------#

def status(feature):
  """
  Helper function that prints whether or not a feature has been processsed.
  """
  print "Processing", feature,": ok"
  
def get_combined_data():
  """
  Function to combine both the training and testing data sets.
  """
  # Reading training set data
  train = pandas.read_csv(training_data_path, encoding="utf-8-sig")
  
  # Reading testing set data
  test = pandas.read_csv(testing_data_path, encoding='utf-8-sig')
  
  # Extract and remove targets from the training data
  targets = train.Survived
  train.drop(labels='Survived', axis=1, inplace=True)
  
  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(labels='index', axis=1, inplace=True)
  
  return combined
  
combined = get_combined_data()
print "combined shape:", combined.shape
  
# Extract the title/salutation from each name

def get_titles():
  global combined
  # Extract the title from each name.
  combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split(".")[0].strip())
  
  # Collection of possible titles
  Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
                        
  combined['Title'] = combined.Title.map(Title_Dictionary)
  
get_titles()

print combined.head()
  
# Since we're missing a large number (177, or 13%) of the Age data, we can't replace them with the mean or median.
  
grouped = combined.groupby(['Sex', 'Pclass', 'Title'])
print grouped.median()

# We can see that the mediian age ranges depending on the sex of the passernger or their class.

# We'll create a function that fills in the missing age in 'combined' dataframe based on the different attributes and the median values as above.

def process_age():
  """
  """
  
  global combined
  
  def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
  
  # If the age is NaN, then use the logic above to fill it in, else use default values
  combined.Age = combined.apply(lambda r: fillAges(r) if numpy.isnan(r['Age']) else r['Age'],
                                axis=1)
                                
  status('age')
  
process_age()

# Check that NaN values in Age have been replaced
print combined.info()

def process_names():
  
  global combined
  
  combined.drop('Name', axis=1, inplace=True)
  # Encoding in dummy variable
  titles_fake = pandas.get_dummies(combined['Title'], prefix='Title')
  combined = pandas.concat([combined, titles_fake],axis=1)
  
  # Remove the 'title' variable
  combined.drop('Title', axis=1, inplace=True)
  status('Names')
  
  
process_names()
  

# One value in fare is missing, so create function to replace it with median
def process_fares():
  global combined
  
  combined.Fare.fillna(combined.Fare.mean(), inplace=True)
  
  status('Fare')
  
process_fares()

# Now replace missing values in 'Embarkment'

def process_embarked():
  global combined
  
  # replace missing values with the most freqeunt value
  
  combined.Embarked.fillna('S', inplace=True)
  embarked_fake = pandas.get_dummies(combined['Embarked'], prefix='Embarked')
  combined = pandas.concat([combined, embarked_fake], axis=1)
  combined.drop('Embarked', axis=1, inplace=True)
  
  status('Embarked')
  
process_embarked()
  
def process_cabin():
  global combined
  # Replace missing cabin information with U, for 'Unknown'
  combined.Cabin.fillna('U', inplace=True)
  
  # Map each cabin value with a cabin letter
  combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
  
  # dummy encoding
  cabin_fake = pandas.get_dummies(combined['Cabin'], prefix='Cabin')
  
  combined = pandas.concat([combined, cabin_fake], axis=1)
  
  combined.drop('Cabin', axis=1, inplace=True)
  status('Cabin')
  
process_cabin()

print combined.info()  # All missing values should now be removed.

def process_sex():
  global combined
  
  # Map string values of Sex to a number
  combined['Sex'] = combined['Sex'].map({'male':1, 'female':2})
  
  status('Sex')
  
process_sex()

def process_pclass():
  global combined
  # encoding into 3 categories:
  pclass_dummies = pandas.get_dummies(combined['Pclass'], prefix='Pclass')
    
  # adding dummy variables
  combined = pandas.concat([combined, pclass_dummies],axis=1)
    
  # removing "Pclass"
  combined.drop('Pclass',axis=1,inplace=True)   
  status('pclass')
  
process_pclass()
  
def process_ticket():
  
  global combined
  
  def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = filter(lambda t: not t.isdigit(), ticket)
    if len(ticket) > 0:
      return ticket[0]
      
    else:
      return 'XXX'
      
    # Extract 'dummy variables' from tickets
    
  combined['Ticket'] = combined['Ticket'].map(cleanTicket)
  tickets_fake = pandas.get_dummies(combined['Ticket'],prefix='Ticket')
  combined = pandas.concat([combined, tickets_fake],axis=1)
  combined.drop('Ticket', inplace=True, axis=1)
    
  status('ticket')

process_ticket()

# Now we'll create a new variable based on the size. The reasoning is that if large families
# are grouped together, they are more likely to get rescued than people travelling alone.

def process_family():
  
  global combined
  # Introduce feature for the size of the families
  combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
  
  # Introduce other features based on the family size.
  
  # Single if the person is travelling alone
  combined['Single'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
  
  # Small family if between 2-4 people
  combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2 <= s <=4 else 0)
  
  combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if s >= 5 else 0)
  
  status('Family')
  
process_family()

print "There are now", combined.shape[1], "features."

# Once all the features are completed, we'll scale/normalize all of them (except for the Passenger ID)

def scale_all_features():
  global combined
  
  features = list(combined.columns)
  features.remove('PassengerId')
  combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
  
  print "Features scaled succesfully."
  
scale_all_features()


#-----------------------------------------#
# Part 3 - Generating the machine learning model
#-----------------------------------------#

# We're going to use Random Forrests as our machine learning model.
# The process is basically:
# 1. Break the combined dataset into the training set and test set.
# 2. Use the training set to build the predictive model.
# 3. Evaluate the model using the training set.
# 4. Test the model using the testing set and geenrate an output file for the submisson.

# Start by importing packages.
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

# Use a 5-fold cross validaton with the 'Accuracy' metric. Start by defining a sjmall scoring
# function.

def compute_score(clf, x, y, scoring='accuracy'):
  xval = cross_val_score(estimator=clf, x=x, y=y, cv=5, scoring=scoring)
  return numpy.mean(xval)
  
def recover_training_testing_target():
  global combined
  
  train_0 = pandas.read_csv(training_data_path)
  
  targets = train_0.Survived
  train = combined.ix[0:890]
  test = combined.ix[891:]
  
  return train, test, targets
  
train, test, targets = recover_training_testing_target()

# Since we have created a large number of features, we'll try to reduce the number by only 
# selecting the most important features to use. To do this, we'll use tree-based estimators

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)

features = pandas.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

#  Sort features from most important to least important
print features.sort(['importance'], ascending=False)

# Compress features from 68 to 15
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
print train_new.shape

test_new = model.transform(test)
print test_new.shape

# Tune hyperparameters in random forest model
forest = RandomForestClassifier(max_features = 'sqrt')
parameter_grid = {
                  'max_depth' : [4,5,6,7,8],
                  'n_estimators': [200,210,240,250],
                  'criterion' : ['gini', 'entropy']
                  }

cross_validation = StratifiedKFold(y=targets, n_folds=5) 

grid_search = GridSearchCV(forest, param_grid = parameter_grid, cv = cross_validation)

grid_search.fit(train_new, targets)

print 'Best score: {}'.format(grid_search.best_score_)
print 'Best parameters: {}'.format(grid_search.best_params_)


output = grid_search.predict(test_new).astype(int)
df_output = pandas.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)


final_time = time.time()

print "------------------"
print "Operation complete. Time elapsed:", round(final_time - initial_time, 4), "seconds."

# END

    
  
