import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
features_dict={}
y_pred={}
crops = pd.read_csv("soil_measures.csv")
crops.isna().sum().sort_values
crops.info()
crops['crop'].unique()
crops.head()
X=crops.drop('crop', axis=1)
print(X)
y=crops['crop']
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y)
feature_performance={}
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred[feature] = log_reg.predict(X_test[[feature]])
    feature_performance[feature] = metrics.f1_score(y_test, y_pred[feature], average='weighted')
    print(f"F1-score for {feature}: {feature_performance[feature]}")
best_predictive_feature={"K":0.24431389311137577}