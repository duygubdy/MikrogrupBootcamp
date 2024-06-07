import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("data/data.csv")
features = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome',
            'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
            'JobInvolvement', 'JobLevel', 'JobRole',
            'MaritalStatus', 'MonthlyIncome',
            'OverTime', 'RelationshipSatisfaction',
            'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

X = data[features].copy()
y_attrition = data['Attrition']
y_performance = data['JobSatisfaction']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    X.loc[:, col] = label_encoders[col].fit_transform(X[col])

X_train_attrition, X_test_attrition, y_train_attrition, y_test_attrition = train_test_split(X, y_attrition,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

model_attrition = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=10000)
model_attrition.fit(X_train_attrition, y_train_attrition)

joblib.dump(model_attrition, 'logistic_model_attrition.pkl')
joblib.dump(label_encoders, 'label_encoders_attrition.pkl')


X_train_performance, X_test_performance, y_train_performance, y_test_performance = train_test_split(X, y_performance,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)

model_performance = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=10000)
model_performance.fit(X_train_performance, y_train_performance)

joblib.dump(model_performance, 'logistic_model_performance.pkl')

class AttritionPerformancePredictor:
    def __init__(self, model_attrition_path, model_performance_path, encoders_path):
        self.model_attrition = joblib.load(model_attrition_path)
        self.model_performance = joblib.load(model_performance_path)
        self.label_encoders = joblib.load(encoders_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        for col in self.label_encoders:
            input_df[col] = self.label_encoders[col].transform(input_df[col])

        attrition_prediction = self.model_attrition.predict(input_df)
        performance_prediction = self.model_performance.predict(input_df)
        return attrition_prediction, performance_prediction