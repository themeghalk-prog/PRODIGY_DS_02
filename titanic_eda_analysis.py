import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Pandas, Matplotlib, and Seaborn libraries imported.")
# Output:
# Pandas library imported as 'pd'.
# Matplotlib and Seaborn libraries imported.

csv_url = 'https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%202/test.csv'
df = pd.read_csv(csv_url)
print("
Dataset loaded successfully. Displaying the first 5 rows:")
print(df.head().to_markdown(index=False))
# Output:
# Dataset loaded successfully. Displaying the first 5 rows:
# | PassengerId | Pclass | Name                                         | Sex    | Age  | SibSp | Parch | Ticket   | Fare    | Cabin | Embarked |
# |:------------|:-------|:---------------------------------------------|:-------|:-----|:------|:------|:---------|:--------|:------|:---------|
# | 892         | 3      | Kelly, Mr. James                             | male   | 34.5 | 0     | 0     | 330911   | 7.8292  | nan   | Q        |
# | 893         | 3      | Wilkes, Mrs. James (Ellen Needs)             | female | 47   | 1     | 0     | 363272   | 7       | nan   | S        |
# | 894         | 2      | Myles, Mr. Thomas Francis                    | male   | 62   | 0     | 0     | 240276   | 9.6875  | nan   | Q        |
# | 895         | 3      | Wirz, Mr. Albert                             | male   | 27   | 0     | 0     | 315154   | 8.6625  | nan   | S        |
# | 896         | 3      | Hirvonen, Mrs. Alexander (Helga E Lindqvist) | female | 22   | 1     | 1     | 3101298  | 12.2875 | nan   | S        |

print("
--- DataFrame Info ---
")
df.info()
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 418 entries, 0 to 417
# Data columns (total 11 columns):
#  #   Column       Non-Null Count  Dtype
# --  ------       --------------  -----
#  0   PassengerId  418 non-null    int64
#  1   Pclass       418 non-null    int64
#  2   Name         418 non-null    object
#  3   Sex          418 non-null    object
#  4   Age          332 non-null    float64
#  5   SibSp        418 non-null    int64
#  6   Parch        418 non-null    int64
#  7   Ticket       418 non-null    object
#  8   Fare         417 non-null    float64
#  9   Cabin        91 non-null     object
# 10  Embarked     418 non-null    object
# dtypes: float64(2), int64(4), object(5)
# memory usage: 36.1+ KB

print("
--- Descriptive Statistics ---
")
print(df.describe().to_markdown())
# Output:
# |          | PassengerId | Pclass | Age      | SibSp    | Parch    | Fare     |
# |:---------|:------------|:-------|:---------|:---------|:---------|:---------|
# | count    | 418         | 418    | 332      | 418      | 418      | 417      |
# | mean     | 1100.5      | 2.26555| 30.2725  | 0.447368 | 0.392344 | 35.627181|
# | std      | 120.81      | 0.841838| 14.1812  | 0.89676  | 0.981429 | 55.907576|
# | min      | 892         | 1      | 0.17     | 0        | 0        | 0        |
# | 25%      | 996.25      | 1      | 21       | 0        | 0        | 7.8958   |
# | 50%      | 1100.5      | 3      | 27.5     | 0        | 0        | 14.4542  |
# | 75%      | 1204.75     | 3      | 39       | 1        | 0        | 31.5     |
# | max      | 1309        | 3      | 76       | 8        | 9        | 512.329  |

print("
--- Missing Values Count ---
")
print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
# Output:
# |            | 0   |
# |:-----------|:----|
# | PassengerId| 0   |
# | Pclass     | 0   |
# | Name       | 0   |
# | Sex        | 0   |
# | Age        | 86  |
# | SibSp      | 0   |
# | Parch      | 0   |
# | Ticket     | 0   |
# | Fare       | 1   |
# | Cabin      | 327 |
# | Embarked   | 0   |

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
# The 'Cabin' column was dropped due to a high number of missing values.
df.drop('Cabin', axis=1, inplace=True)
print("
Missing values in 'Age' and 'Fare' imputed with median.")
print("'Cabin' column dropped.")
print("Updated missing values count:")
print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
# Output:
# Missing values in 'Age' and 'Fare' imputed with median.
# 'Cabin' column dropped.
# Updated missing values count:
# |            | 0   |
# |:-----------|:----|
# | PassengerId| 0   |
# | Pclass     | 0   |
# | Name       | 0   |
# | Sex        | 0   |
# | Age        | 0   |
# | SibSp      | 0   |
# | Parch      | 0   |
# | Ticket     | 0   |
# | Fare       | 0   |
# | Embarked   | 0   |

# Exploratory Data Analysis
print("
--- Descriptive Statistics for Numerical Columns (After Cleaning) ---
")
print(df.describe().to_markdown())
# Output:
# |          | PassengerId | Pclass | Age      | SibSp    | Parch    | Fare     |
# |:---------|:------------|:-------|:---------|:---------|:---------|:---------|
# | count    | 418         | 418    | 418      | 418      | 418      | 418      |
# | mean     | 1100.5      | 2.26555| 29.599282| 0.447368 | 0.392344 | 35.576535|
# | std      | 120.81      | 0.841838| 12.70377 | 0.89676  | 0.981429 | 55.850103|
# | min      | 892         | 1      | 0.17     | 0        | 0        | 0        |
# | 25%      | 996.25      | 1      | 23       | 0        | 0        | 7.8958   |
# | 50%      | 1100.5      | 3      | 27       | 0        | 0        | 14.4542  |
# | 75%      | 1204.75     | 3      | 35.75    | 1        | 0        | 31.471875|
# | max      | 1309        | 3      | 76       | 8        | 9        | 512.329  |

print("
--- Value Counts for Categorical Columns ---
")
print("Sex:")
print(df['Sex'].value_counts().to_markdown(numalign="left", stralign="left"))
# Output:
# Sex:
# | Sex    | count |
# |:-------|:------|
# | male   | 266   |
# | female | 152   |

print("
Embarked:")
print(df['Embarked'].value_counts().to_markdown(numalign="left", stralign="left"))
# Output:
# Embarked:
# | Embarked | count |
# |:---------|:------|
# | S        | 270   |
# | C        | 102   |
# | Q        | 46    |

print("
Pclass:")
print(df['Pclass'].value_counts().to_markdown(numalign="left", stralign="left"))
# Output:
# Pclass:
# | Pclass | count |
# |:-------|:------|
# | 3      | 218   |
# | 1      | 107   |
# | 2      | 93    |

print("
--- Correlation Matrix for Numerical Columns ---
")
print(df.select_dtypes(include=['number']).corr().to_markdown())
# Output:
# |            | PassengerId | Pclass    | Age       | SibSp     | Parch     | Fare      |
# |:-----------|:------------|:----------|:----------|:----------|:----------|:----------|
# | PassengerId| 1           | -0.026751 | -0.031447 | 0.003818  | 0.04308   | 0.008633  |
# | Pclass     | -0.026751   | 1         | -0.467853 | 0.001087  | 0.018721  | -0.577313 |
# | Age        | -0.031447   | -0.467853 | 1         | -0.071197 | -0.043731 | 0.342357  |
# | SibSp      | 0.003818    | 0.001087  | -0.071197 | 1         | 0.306895  | 0.171912  |
# | Parch      | 0.04308     | 0.018721  | -0.043731 | 0.306895  | 1         | 0.230325  |
# | Fare       | 0.008633    | -0.577313 | 0.342357  | 0.171912  | 0.230325  | 1         |

print("
--- Mean Age and Fare grouped by Sex and Pclass ---
")
print(df.groupby(['Sex', 'Pclass'])[['Age', 'Fare']].mean().to_markdown())
# Output:
# | Sex    | Pclass | Age      | Fare      |
# |:-------|:-------|:---------|:----------|
# | female | 1      | 40.76    | 115.591   |
# |        | 2      | 24.464   | 26.4388   |
# |        | 3      | 24.2732  | 13.7351   |
# | male   | 1      | 38.8596  | 75.5866   |
# |        | 2      | 30.6905  | 20.1847   |
# |        | 3      | 25.3727  | 11.8443   |

# Visualize Patterns and Trends
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
# Output:
# (Image of histograms)

categorical_cols = ['Sex', 'Pclass', 'Embarked']

plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 3, i)
    sns.countplot(x=df[col])
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()
# Output:
# (Image of countplots)

plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Columns')
plt.show()
# Output:
# (Image of heatmap)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age Distribution by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Age')

plt.subplot(1, 2, 2)
sns.boxplot(x='Sex', y='Age', data=df)
plt.title('Age Distribution by Sex')
plt.xlabel('Sex')
plt.ylabel('Age')

plt.tight_layout()
plt.show()
# Output:
# (Image of boxplots)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')

plt.subplot(1, 2, 2)
sns.boxplot(x='Sex', y='Fare', data=df)
plt.title('Fare Distribution by Sex')
plt.xlabel('Sex')
plt.ylabel('Fare')

plt.tight_layout()
plt.show()
# Output:
# (Image of boxplots)
