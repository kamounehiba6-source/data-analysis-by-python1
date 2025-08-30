import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {'Name': ['hiba', 'jamal', 'zaineb', 'jamila', 'ferdaus'],
        'Age': [25, 30, 35, 40, 45],
        'Salary': [50000, 60000, 70000, 80000, 90000]}
df = pd.DataFrame(data)
mean_salary = np.mean(df['Salary'])
max_age = np.max(df['Age'])
plt.bar(df['Name'], df['Salary'])
plt.xlabel('Name')
plt.ylabel('Salary')
plt.title('Salary Information')
plt.show()
print("Mean Salary: $", mean_salary)
print("Maximum Age: ", max_age)
plt.bar(df['Name'], df['Salary'])
plt.xlabel('Name')
plt.ylabel('Salary')
plt.title('Salary Information')
plt.savefig('salary_plot.png')
summary = df.describe()
print("\nStatistical Summary of the Data:")
print(summary)
from sklearn.linear_model import LinearRegression

X = df[['Age']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)
print("\nLinear Regression Results:")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("\nData analysis complete. Presentation ready for your boss.")