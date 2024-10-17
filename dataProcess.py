from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE

# Construct the training set and test set
def load_data():
    df = pd.read_csv('./data/ant-1.7.csv').iloc[:, 3:24]
    x_data = df.drop('bug', axis=1).values
    x_data = StandardScaler().fit_transform(x_data)
    df[df['bug'] != 0] = 1
    y_data = df['bug'].values
    # Balance the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    x_train, y_train = SMOTE(random_state=42).fit_resample(X=x_train, y=y_train)  # Oversampling

    return x_train, x_test, y_train, y_test
