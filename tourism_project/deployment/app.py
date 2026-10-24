
import os
import joblib 
import pandas as pd 
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__),"tourism_model.joblib")

st.set_page_config(
    page_title="Tourism Package Prediction",
    layout="centered"
)

st.title("Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase a tourism package."
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at:{MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model:{error}")
    st.stop()
with st.form("customer_details"):
    st.subheader("Customer Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    contact = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    pitch_duration = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, value=15
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Freelancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    visitors = st.number_input(
        "Number of People Visiting", min_value=1, max_value=10, value=2
    )
    followups = st.number_input(
        "Number of Follow-ups", min_value=1, max_value=10, value=3
    )
    product = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    property_star = st.selectbox(
        "Preferred Property Star", [3, 4, 5]
    )
    marital_status = st.selectbox(
        "Marital Status", ["Married", "Divorced", "Unmarried", "Single"]
    )
    trips = st.number_input(
        "Number of Trips", min_value=1, max_value=30, value=3
    )
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    satisfaction = st.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
    )
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    children = st.number_input(
        "Number of Children Visiting", min_value=0, max_value=10, value=0
    )
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    income = st.number_input(
        "Monthly Income", min_value=0, value=25000
    )

    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    customer = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": pitch_duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": visitors,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])

    prediction = int(model.predict(customer)[0])

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The customer is predicted to purchase the package.")
    else:
        st.info("The customer is predicted not to purchase the package.")
