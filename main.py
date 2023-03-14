import pandas as pd
from IPython.display import display

base_data = pd.read_csv("DSP_4.csv")
df = pd.read_csv("DSP_4.csv", sep=";")

"""
print(df)
print(df.columns)

print(df.isnull().any())

print(df["wiek"].mean())
print(df["wzrost"].mean())

df_2 = df.fillna(df.mean())
print(df_2)

print(df_2.isnull().any())

print(df["wiek"].mean())
print(round(df["wiek"].mean(), 2))
print(df["wiek"].median())
print(df['wiek'].max())
print(df['wiek'].min())
print(df['wiek'].var())
"""

# TASK 1 - to use "display" I needed to import IPython.display
display(df)

# TASK 2 Try two other ways to fill in missing values. Fill missing values with the median and save as the dataframe
# "df_3". Display the median for the age and height variables below the table. Fill missing values as 0 and write as
# the dataframe "df_4".

df_3 = df.fillna(df.median())
print()
print("df_3:")
display(df_3)
print("Age median: ", df_3['wiek'].median())
print("Age median: ", df_3['wzrost'].median())

df_4 = df.fillna(0)
print()
print(df_4)

# TASK 3 Load DSP_5.csv. Check if there are missing data - if necessary fill them with an average.
# Show the mean, variance, range for "hp" variable.
# Show a correlation table for the variables. Is there anything interesting?
dataframe5 = pd.read_csv("DSP_5.csv", sep=";")
filledDF5 = dataframe5.fillna(dataframe5.mean(numeric_only=True))

pd.set_option('display.width', None)  # at this point this is needed to display the whole table in console

print("DSP_5 filled data:")
display(filledDF5)
print("Mean for \"hp\": ", filledDF5['hp'].mean())
print("Variance for \"hp\": ", filledDF5['hp'].var())
print("Range for \"hp\": ", filledDF5['hp'].min(), " - ", filledDF5['hp'].max())

DF5_correlation = filledDF5.corr(numeric_only=True)
print()
print("Correlation table:")
print(DF5_correlation)
# How to read correlation table(note to self rather than the teacher obviously)
# 1. It doesn't matter which side of the table we look at first, the matrix is symmetric
# 2. The results range from -1 to 1:
# - -1 means very strong negative correlation: as one characteristic increases/decreases the other will
# decrease/increase (e.g. mpg(miles per gallon) and hp(horsepower)
# - 0 no correlation
# - 1 strong positive correlation: as one characteristic increases so does the other one

# My very interesting observations:
# There is a significant negative correlation between miles per gallon and horsepower, suggesting that cars with more
# power tend to have a decreased range, possibly related to this might be the weight(I am guessing that's what "wt" is)
# which also has a significant negative correlation. Another aspect at play here might be the relation to
# (what I assume is) the number of cylinders, denoted as cyl, which also has a very significant correlation
# to miles per gallon. All of this suggests that cars which are heavier, and(or?) have more powerful engines
# suffer from lower fuel efficiency



# TASK 4 1. Load the "DSP_8.csv" data. This is the Heart Attack Analysis & Prediction Dataset (aut. Rashik Rahman)
# available from Kaggle. Make sure that decimal numbers are properly formatted.
# 2. Write a code that will return information:
# a. The number of columns (with their names),
# b. The number of rows (observations),
# c. Any missing data,
# d. The mean age and standard deviation in two separate groups (men&women),
# e. the percentage of men in the dataset,
# f. the number of women aged 45 to 50 years,
# g. correlation between the variables, but only for people with their ECG at rest is normal
# (when in the RestingECG column, "Normal" appears).


dataframe8 = pd.read_csv("DSP_8.csv", sep=",", decimal=".")  # appears to work without "decimal" regardless, but doesn't hurt
print()
# display(dataframe8)

# a. & b.
num_rows, num_cols = dataframe8.shape
col_names = dataframe8.columns.tolist()

print(f"The number of rows is: {num_rows}")
print(f"The number of columns is: {num_cols}")
print(f"The names of the columns are: {col_names}")

# c. - assuming here I am supposed to check if there is any missing data and return a boolean
missing_data = dataframe8.isnull().sum()
print()
print(f"The number of missing data: \n{missing_data}")

# d.
age_stats = dataframe8.groupby('Sex')['Age'].agg(['mean', 'std'])
# This will group results by Sex and then apply mean() and std() functions to the Age column
# In data science aggregating often refers to the process of summarizing data by applying mathematical functions

print(f"Mean and standard deviation of age for Men and Women \n{age_stats}")

# e.
num_male = len(dataframe8[dataframe8['Sex'] == 'M'])
percentage_male = num_male / len(dataframe8) * 100
print(f"The percentage of men in the dataset: {percentage_male:.2f}%")

# f.
df_women_45_50 = dataframe8[(dataframe8['Age'] >= 45) & (dataframe8['Age'] <= 50) & (dataframe8['Sex'] == 'F')]
num_women_45_50 = df_women_45_50.shape[0]  # getting the first element which is the number of rows
print(f"The number of women in the dataset: {num_women_45_50}")

# g.
df_ECG_normal = dataframe8[dataframe8["RestingECG"] == "Normal"]
df_ECG_normal_corr = df_ECG_normal.corr(numeric_only=True)
display(f"The correlation between variables for patients with normal RestingECG: \n{df_ECG_normal_corr}")