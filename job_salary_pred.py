import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

# Read the data into a DataFrame
data = pd.read_csv("salary_data_cleaned.csv")

# Drop irrelevant columns
data = data.loc[:, ['job_title', 'age', 'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'min_salary', 'max_salary', 'avg_salary']]

# Split the data into features and target
features = data[["age", "python_yn", "R_yn", "spark", "aws", "excel"]]
target = data[["min_salary", "max_salary", "avg_salary"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.1, random_state=42
)

# Train a linear regression model to predict the salary
regression_model = LinearRegression()
regression_model.fit(X_train, y_train[["min_salary", "max_salary", "avg_salary"]])

# Make predictions on the testing set for salary
salary_pred = regression_model.predict(X_test)

# Calculate the mean absolute percentage error (MAPE) for salary prediction
mape_salary = (
    np.mean(
        np.abs(
            (y_test[["min_salary", "max_salary", "avg_salary"]] - salary_pred)
            / y_test[["min_salary", "max_salary", "avg_salary"]]
        )
    )
    * 100
)

print(f"\nSalary MAPE: {mape_salary:.2f}%")


# Job title prediction
target_job_title = data["job_title"]

# Split the data into training and testing sets for job title prediction
(
    X_train_job_title,
    X_test_job_title,
    y_train_job_title,
    y_test_job_title,
) = train_test_split(features, target_job_title, test_size=0.1, random_state=42)

# Train a random forest classifier to predict the job title
classifier_model = RandomForestClassifier()
classifier_model.fit(X_train_job_title, y_train_job_title)

# Make predictions on the testing set for job title
job_title_pred = classifier_model.predict(X_test_job_title)

# Calculate the accuracy for job title prediction
accuracy_job_title = classifier_model.score(X_test_job_title, y_test_job_title) * 100
print(f"Job Title Prediction Accuracy: {accuracy_job_title:.2f}%\n")


# Ask the user for trait values
print(
    "Please enter the following information to get a salary and job title prediction:"
)
age = int(input("Enter age: "))
python_yn = int(input("Python skills (1 for Yes, 0 for No): "))
r_yn = int(input("R skills (1 for Yes, 0 for No): "))
spark = int(input("Spark skills (1 for Yes, 0 for No): "))
aws = int(input("AWS skills (1 for Yes, 0 for No): "))
excel = int(input("Excel skills (1 for Yes, 0 for No): "))

# Create a DataFrame with the user input
user_input = pd.DataFrame(
    {
        "age": [age],
        "python_yn": [python_yn],
        "R_yn": [r_yn],
        "spark": [spark],
        "aws": [aws],
        "excel": [excel],
    }
)

# Make predictions using the model
predictions = regression_model.predict(user_input)
job_title_predictions = classifier_model.predict(user_input)

# Print the predicted job title
print(f"\nPredicted Job Title: {job_title_predictions[0]}\n")

# Convert predictions to integers
min_salary = int(predictions[0][0])
max_salary = int(predictions[0][1])
avg_salary = int(predictions[0][2])

# Calculate monthly salaries
min_salary_monthly = int((min_salary / 12) * 1000)
max_salary_monthly = int((max_salary / 12) * 1000)
avg_salary_monthly = int((avg_salary / 12) * 1000)

# Format salaries with $ and K signs
min_salary_str = "$ {}k".format(min_salary)
max_salary_str = "$ {}k".format(max_salary)
avg_salary_str = "$ {}k".format(avg_salary)
min_salary_monthly_str = "$ {}".format(min_salary_monthly)
max_salary_monthly_str = "$ {}".format(max_salary_monthly)
avg_salary_monthly_str = "$ {}".format(avg_salary_monthly)

# Print the predicted salary range
print("Predicted Salary Range:")

# Prepare data for the table
data = [
    ["Salary Type", "Yearly Salary", "Monthly Salary"],
    ["Minimum", min_salary_str, min_salary_monthly_str],
    ["Maximum", max_salary_str, max_salary_monthly_str],
    ["Average", avg_salary_str, avg_salary_monthly_str],
]

# Print the table
print(tabulate(data, headers="firstrow", tablefmt="grid"))
