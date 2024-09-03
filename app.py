import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import base64

# Function to add background image from a local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        image_bytes = image.read()
        base64_image = base64.b64encode(image_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Path to your background image
image_file = r"C:\Users\siddiqui taj 2024\PycharmProjects\pythonProject\House_Price_Predictor\img.jpg"

# Call the function to set the background
add_bg_from_local(image_file)













# Load and prepare the dataset
house_df = pd.read_csv('zameen-updated.csv')
house_df.dropna(inplace=True)

df = house_df[house_df["city"] == "Karachi"]

# Dropping unnecessary columns
df = df.drop(['property_id', 'location_id', 'page_url', 'province_name', 'property_type', 'latitude', 'longitude',
              'purpose', 'date_added', 'agency', 'agent', 'city', 'Area Type', 'Area Category', 'area'], axis=1)

# Dropping rows with 0 bedrooms or baths and rows with more baths than bedrooms
df = df[(df['bedrooms'] > 0) & (df['baths'] > 0) & (df['baths'] <= df['bedrooms'])]

# Add price_per_size column for feature engineering
df['price_per_size'] = df['price'] / df['Area Size']

# Group locations with fewer than 10 occurrences as 'others'
location_stats = df['location'].value_counts()
locations_less_than_10 = location_stats[location_stats <= 10].index
df['location'] = df['location'].apply(lambda x: 'others' if x in locations_less_than_10 else x)


# Check and handle infinities and large values
def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    for col in non_numeric_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.dropna(inplace=True)

    return df


df = clean_data(df)

# Feature and target preparation
X = df.drop(['price'], axis=1)
y = df['price']

# ColumnTransformer and Pipeline setup
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['location']),
        ('num', SimpleImputer(strategy='median'), ['Area Size', 'bedrooms', 'baths'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=0))
])

# Train-test split and fitting the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


def predict_price(location, AreaSize, bedrooms, baths):
    input_data = pd.DataFrame({
        'location': [location],
        'Area Size': [AreaSize],
        'bedrooms': [bedrooms],
        'baths': [baths],
        'price_per_size': [0]  # Placeholder value
    })
    input_data = clean_data(input_data)  # Clean input data as well
    prediction = pipeline.predict(input_data)[0]
    return prediction


# Streamlit UI
st.title('House Price Prediction')

# Location selection
locations = df['location'].unique().tolist()
location = st.selectbox('Select Location:', locations)

# Input fields
AreaSize = st.number_input('Enter Area Size (sq ft):', min_value=0, value=1000,
                           help="For example, 1000 sq ft is typical for many houses.")
bedrooms = st.number_input('Enter Number of Bedrooms:', min_value=1, value=3)
baths = st.number_input('Enter Number of Bathrooms:', min_value=1, value=2)

# Prediction button
if st.button('Predict'):
    predicted_price = predict_price(location, AreaSize, bedrooms, baths)
    # Display the price in Lakhs or Crores
    if predicted_price > 10000000:
        st.write(f'The predicted price is {predicted_price / 10000000:.4f} Crores')
    else:
        st.write(f'The predicted price is {predicted_price / 100000:.4f} Lakhs')
