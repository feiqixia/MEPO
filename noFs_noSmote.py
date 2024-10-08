from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
#构建训练集及测试集
df = pd.read_csv('./data/jedit-3.2.csv').iloc[:, 3:24]
def load_data(flag):
    x_data = df.drop('bug', axis=1).values
    x_data = StandardScaler().fit_transform(x_data)
    df[df['bug'] != 0] = 1
    y_data = df['bug'].values
    # 平衡数据集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,random_state=42)
    if(flag == 1):
        x_train, y_train = SMOTE(random_state=42).fit_resample(X=x_train, y=y_train)  # 过采样
    return x_train, x_test, y_train, y_test

x_train, x_valid, y_train, y_valid = load_data(0)

# mdl = DecisionTreeClassifier(random_state=32)
mdl = svm.SVC(random_state=32)
mdl.fit(x_train, y_train)
ypred = mdl.predict(x_valid)
auc = metrics.roc_auc_score(y_valid, ypred)
print("无smote时 auc value is:", auc)



