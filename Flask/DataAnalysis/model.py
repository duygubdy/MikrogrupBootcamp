import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/data.csv")

# Categorize columns
data['Age'] = pd.cut(data['Age'], bins=[-1, 25, 37, 47, data['Age'].max()], labels=['0-25', '26-37', '38-47', f'48-{data["Age"].max()}'])
data['DistanceFromHome'] = pd.cut(data['DistanceFromHome'], bins=[-1, 10, data['DistanceFromHome'].max()], labels=['Near', 'Far'])
data['MonthlyIncome'] = pd.cut(data['MonthlyIncome'], bins=[-1, 2000, 3000, 7000, 11000, data['MonthlyIncome'].max()], labels=['0-2000', '2001-3000', '3001-7000', '7001-11000', f'11001-{data["MonthlyIncome"].max()}'])
data['StockOptionLevel'] = pd.cut(data['StockOptionLevel'], bins=[0, 1.5, 3.5], labels=['Group1-2', 'Group3-4'], right=False)
data['TotalWorkingYears'] = pd.cut(data['TotalWorkingYears'], bins=[-1, 3, 10, 25, data['TotalWorkingYears'].max()], labels=['0-3 Years', '4-10 Years', '11-25 Years', f"26-{data['TotalWorkingYears'].max()} Years"])
data['YearsAtCompany'] = pd.cut(data['YearsAtCompany'], bins=[-1, 5, 11, data['YearsAtCompany'].max()], labels=['0-5', '6-11', f'11-{data["YearsAtCompany"].max()}'])
data['YearsInCurrentRole'] = pd.cut(data['YearsInCurrentRole'], bins=[-1, 2, 4, 6, 9, data['YearsInCurrentRole'].max()], labels=['0-2', '3-4', '5-6', '7-9', f'9-{data["YearsInCurrentRole"].max()}'])
data['YearsSinceLastPromotion'] = pd.cut(data['YearsSinceLastPromotion'], bins=[-1, 2, 7, data['YearsSinceLastPromotion'].max()], labels=['0-2', '3-7', f'8-{data["YearsSinceLastPromotion"].max()}'])
data['YearsWithCurrManager'] = pd.cut(data['YearsWithCurrManager'], bins=[-1, 2, 7, data['YearsWithCurrManager'].max()], labels=['0-2', '3-7', f'8-{data["YearsWithCurrManager"].max()}'])

features = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 'OverTime', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Separate dataset into features and target variable
X = data[features]
y_attrition = data['Attrition']
y_performance = data['PerformanceRating']
# Apply One-Hot Encoding for categorical columns after label encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_resampled_attrition, y_resampled_attrition = adasyn.fit_resample(X_encoded, y_attrition)
X_resampled_performance, y_resampled_performance = adasyn.fit_resample(X_encoded, y_performance)

# Split the resampled dataset
X_train_attrition, X_test_attrition, y_train_attrition, y_test_attrition = train_test_split(X_resampled_attrition, y_resampled_attrition, test_size=0.2, random_state=42)
X_train_performance, X_test_performance, y_train_performance, y_test_performance = train_test_split(X_resampled_performance, y_resampled_performance, test_size=0.2, random_state=42)

# Use OneHotEncoder for categorical columns
X_train_attrition = pd.get_dummies(X_train_attrition, drop_first=True)
X_test_attrition = pd.get_dummies(X_test_attrition, drop_first=True)
X_train_performance = pd.get_dummies(X_train_performance, drop_first=True)
X_test_performance = pd.get_dummies(X_test_performance, drop_first=True)

# Ensure the test set has the same columns as the training set
X_test_attrition = X_test_attrition.reindex(columns=X_train_attrition.columns, fill_value=0)
X_test_performance = X_test_performance.reindex(columns=X_train_performance.columns, fill_value=0)


# Logistic Regression model with regularization
model_attrition = LogisticRegression(penalty='l2', C=10.0, solver='lbfgs', max_iter=5000, multi_class='auto')
model_svm_rbf_performance = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
# Train the model
model_attrition.fit(X_train_attrition, y_train_attrition)
model_svm_rbf_performance.fit(X_train_performance, y_train_performance)

# Predict the target attribute
y_train_attrition_pred = model_attrition.predict(X_train_attrition)
y_test_attrition_pred = model_attrition.predict(X_test_attrition)
y_train_svm_rbf_performance_pred = model_svm_rbf_performance.predict(X_train_performance)
y_test_svm_rbf_performance_pred = model_svm_rbf_performance.predict(X_test_performance)


train_attrition_accuracy = accuracy_score(y_train_attrition, y_train_attrition_pred)
test_attrition_accuracy = accuracy_score(y_test_attrition, y_test_attrition_pred)
train_svm_rbf_performance_accuracy = accuracy_score(y_train_performance, y_train_svm_rbf_performance_pred)
test_svm_rbf_performance_accuracy = accuracy_score(y_test_performance, y_test_svm_rbf_performance_pred)


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