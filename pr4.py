import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#  Dataset with missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, np.nan, 35, 28],  
    'Salary': [50000, 60000, 45000, np.nan, 55000],  
    'Color': ['Red', 'Blue', 'Green', 'Blue', np.nan], 
    'Date': ['2020-05-20', '2018-07-15', '2021-09-10', '2017-03-25', '2019-12-01']
}

df = pd.DataFrame(data)

#  Handle Missing Values
#  Age and Salary with the median of their respective columns
imputer = SimpleImputer(strategy='median')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

#  Color with the most frequent value (mode)
df['Color'].fillna(df['Color'].mode()[0], inplace=True)

#  Convert 'Date' to datetime format and extract useful features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date'], inplace=True)  # Drop original Date column

#  Converts Color into numbers using Label Encoding 
df['Color'] = LabelEncoder().fit_transform(df['Color'])

#  Scale 'Age' and 'Salary'
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])


print(df)
