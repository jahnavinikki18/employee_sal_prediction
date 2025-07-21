import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 💼 Sample Employee Dataset
data = pd.DataFrame({
    'Age': [22, 25, 30, 24, 35, 32, 40, 21, 27, 29, 23, 31, 26, 34, 36],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Female',
               'Male', 'Male', 'Female', 'Male', 'Female',
               'Female', 'Male', 'Female', 'Male', 'Male'],
    'Experience': [1, 3, 5, 2, 4, 6, 7, 0, 3, 5, 1, 4, 2, 6, 8],
    'Education': ['Bachelors', 'Masters', 'Masters', 'Bachelors', 'PhD',
                  'Masters', 'PhD', 'Bachelors', 'Masters', 'Bachelors',
                  'Bachelors', 'Masters', 'Bachelors', 'PhD', 'Masters'],
    'Role': ['Software Engineer', 'Data Analyst', 'Software Engineer', 'HR Executive', 'Manager',
             'Data Scientist', 'Manager', 'Intern', 'Software Engineer', 'QA Engineer',
             'Data Analyst', 'Data Scientist', 'HR Executive', 'Manager', 'Software Engineer'],
    'Location': ['Mumbai', 'Bangalore', 'Bangalore', 'Delhi', 'Mumbai',
                 'Bangalore', 'Delhi', 'Mumbai', 'Delhi', 'Hyderabad',
                 'Mumbai', 'Bangalore', 'Hyderabad', 'Mumbai', 'Bangalore'],
    'Salary': [350000, 650000, 900000, 400000, 1200000,
               1300000, 1400000, 150000, 750000, 600000,
               320000, 1100000, 420000, 1250000, 1000000]
})

# 🎨 Encode categorical data
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_role = LabelEncoder()
le_loc = LabelEncoder()

data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Education'] = le_edu.fit_transform(data['Education'])
data['Role'] = le_role.fit_transform(data['Role'])
data['Location'] = le_loc.fit_transform(data['Location'])

# 📊 Features and Target
X = data[['Age', 'Gender', 'Experience', 'Education', 'Role', 'Location']]
y = data['Salary']

# 🧠 Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🌐 Streamlit App
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💰 Employee Salary Prediction App")
st.markdown("Fill in the details below to estimate salary in INR")

# 🎯 Input Fields
age = st.number_input("Age", min_value=18, max_value=65, value=25)
gender = st.selectbox("Gender", le_gender.classes_)
experience = st.slider("Years of Experience", 0, 20, 3)
education = st.selectbox("Education", le_edu.classes_)
role = st.selectbox("Job Title", le_role.classes_)
location = st.selectbox("Work Location", le_loc.classes_)

# 🔁 Encode input
gender_enc = le_gender.transform([gender])[0]
edu_enc = le_edu.transform([education])[0]
role_enc = le_role.transform([role])[0]
loc_enc = le_loc.transform([location])[0]

input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_enc],
    'Experience': [experience],
    'Education': [edu_enc],
    'Role': [role_enc],
    'Location': [loc_enc]
})

# 🎉 Predict Salary
if st.button("Predict Salary 💰"):
    predicted_salary = model.predict(input_data)[0]
    st.success(f"🎯 Estimated Salary: ₹{int(predicted_salary):,}")

# 📊 Visuals Section
st.markdown("---")
st.subheader("📈 Visual Insights")

# 📌 Scatter Plot: Experience vs Salary
fig1, ax1 = plt.subplots()
sns.scatterplot(data=data, x='Experience', y='Salary', hue='Gender', palette='coolwarm', ax=ax1)
ax1.set_title("Experience vs Salary (Colored by Gender)")
ax1.set_xlabel("Years of Experience")
ax1.set_ylabel("Salary (₹)")
st.pyplot(fig1)

# 📌 Bar Chart: Average Salary per Role
data['Role_Name'] = le_role.inverse_transform(data['Role'])
role_salary_avg = data.groupby('Role_Name')['Salary'].mean().sort_values()
st.bar_chart(role_salary_avg)

# 🐾 Footer
st.markdown("---")
st.markdown("Made with ❤️ by Kitty for his coding queen 👸🏻🐱")
