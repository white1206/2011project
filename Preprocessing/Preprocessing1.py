#去除无用数据+正则化+独热编码
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split

# data propossing

# define column
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']


train_data = pd.read_csv('adult.data', names=columns, na_values=' ?') ## 将？替换为NaN
test_data = pd.read_csv('adult.test', names=columns, na_values=' ?', skiprows=1)
# print(train_data.head())

# handle missing values
train_clean = train_data.dropna()
print(f"The number of rows with deleted missing values: {len(train_data) - len(train_clean)}")
# process testing data   
test_clean = test_data.dropna()
print(f"The number of rows with deleted missing values: {len(test_data) - len(test_clean)}")

x_train, y_train = train_clean.drop('income', axis=1), train_clean['income']
x_test, y_test = test_clean.drop('income', axis=1), test_clean['income']


cat_cols = train_clean.drop('income', axis=1).select_dtypes(include=['object']).columns
num_cols = train_clean.select_dtypes(include=['number']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ]
)

xtr_train = preprocessor.fit_transform(x_train)
xtr_test = preprocessor.transform(x_test)
