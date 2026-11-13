"""Expected schema of the raw Telco churn dataset."""

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]

CATEGORICAL = {
    "gender": ["Male", "Female"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "Churn": YES_NO,
}

# column -> (min, max); None means unbounded on that side
NUMERIC = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, None),
}

ID_COLUMN = "customerID"
REQUIRED_COLUMNS = [ID_COLUMN, *CATEGORICAL, *NUMERIC]

# add-on services counted by the num_services engineered feature
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
