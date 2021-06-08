#Kütüphane okutma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_excel('C:\\Users\\cihat\\OneDrive\\Masaüstü\\Programlar\\Sertifika Dersleri\\Churn Analizi\\Telco-Customer-Churn.xlsx')

df.head()
df.info()
df.describe()

df.loc[df.Churn == 'No','Churn'] = 0
df.loc[df.Churn == 'Yes','Churn'] = 1
df.head()

dataset = df['Churn'].value_counts()
dataset

df1 = pd.read_excel('C:\\Users\\cihat\\OneDrive\\Masaüstü\\Programlar\\Sertifika Dersleri\\Churn Analizi\\Telco-Customer-Churn.xlsx')
df1.head()

char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
for c in char_cols:
    df[c] = pd.factorize(df[c])[0]
    
df.head()

sizes=[5174,1869]
labels='NO','YES'
explode=(0,0.1) #yes olan dışa doğru
fig1 , ax1 = plt.subplots()
ax1.pie(sizes,explode=explode,autopct='%1.1f%%',shadow= True,startangle=75)
ax1.axis('equal')
ax1.set_title('Client Churn Distribution')
ax1.legend(labels)
plt.show()
    
df.groupby('gender').Churn.mean()

# create a dataset
Churn_Mean = [0.269209, 0.261603]
Gender = ('Female', 'Male')
x_pos = np.arange(len(Churn_Mean))

# Create bars with different colors
plt.bar(x_pos, Churn_Mean, color=['orange','blue'])

# Create names on the x-axis
plt.xticks(x_pos, Gender)

# Add title and axis names
#plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
#plt.legend()

# Show graph
plt.show()

catvars = df1.columns.tolist()
catvars = [e for e in catvars if e not in ('TotalCharges', 'MonthlyCharges', 
                                           'tenure', 'customerID', 'Churn')]

y = 'Churn'
for x in catvars:
    plot = df1.groupby(x)[y]\
        .value_counts(normalize=True).mul(100)\
        .rename('percent').reset_index()\
        .pipe((sns.catplot,'data'), x=x, y='percent', hue=y, kind='bar')
    plot.fig.suptitle("Churn by " + x)
    plot

#Tenure
sns.displot(df1.tenure)

#Churn by tenure 
bins = 30
plt.hist(df1[df1.Churn == 'Yes'].tenure, 
         bins, alpha=0.5, density=True, label='Churned')
plt.hist(df1[df1.Churn == 'No'].tenure, 
         bins, alpha=0.5, density=True, label="Didn't Churn")
plt.legend(loc='upper right')
plt.show()

churners_number = len(df[df['Churn'] == 1])
print("Number of churners", churners_number)

churners = (df[df['Churn'] == 1])

non_churners = df[df['Churn'] == 0].sample(n=churners_number)
print("Number of non-churners", len(non_churners))
df3 = churners.append(non_churners)

def show_correlations(df, show_chart = True):
    fig = plt.figure(figsize = (20,10))
    corr = df.corr()
    if show_chart == True:
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True)
    return corr

correlation_df = show_correlations(df3,show_chart=True)

# Define the target variable (dependent variable) 
y = df.Churn 
df = df.drop(['Churn'], axis= 1)

df

# Splitting training and testing data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.20)

# Applying Support Vector Machine algorithm
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', degree=8)  
svclassifier.fit(X_train, y_train)

SVC(degree=8, kernel='linear')
# Predicting part, applying the model to predict
y_pred = svclassifier.predict(X_test)

# Evaluating model performance
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))





















