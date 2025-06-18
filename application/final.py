import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Page Configuration & Title
# ------------------------------------------------
st.set_page_config(page_title="Relay Input & Analytics Dashboard", layout="wide")
st.title("🔌 Relay Input & Analytics Dashboard")

st.write("""
This application provides five sections:
1) Relay1 Prediction  
2) Relay2 Prediction  
3) Relay3 Prediction  
4) Relay4 Prediction  
5) Overall Relay Prediction  

**Relay1** → Marker Distribution (Bar Chart)  
**Relay2** → Histogram of R1:F  
**Relay3** → Bar Chart of Average PM1:V per Relay  
**Relay4** → Joint Plot of R1-PA1:VH vs R1-PM1:V  
**Overall** → Takes all inputs & shows **all 4** graphs.
""")

# ------------------------------------------------
# Sidebar: 5 Sections
# ------------------------------------------------
option = st.sidebar.radio("Choose an option", [
    "Relay1 Prediction", 
    "Relay2 Prediction", 
    "Relay3 Prediction", 
    "Relay4 Prediction", 
    "Overall Relay Prediction"
])

# ------------------------------------------------
# Load Models
# ------------------------------------------------
with open('xgboost.pkl', 'rb') as f:
    model = pickle.load(f)           # For Relay1
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)          # For Relay2
with open('random_forest_model_3.pkl', 'rb') as f:
    rf_3model = pickle.load(f)         # For Relay3
with open('random_forest_model_4.pkl', 'rb') as f:
    rf_4model = pickle.load(f)         # For Relay4
with open('xgboost_overall_model.pkl', 'rb') as f:
    overall_model = pickle.load(f)     # For Overall Prediction

# ------------------------------------------------
# Parameter Definitions for Individual Relay Pages
# ------------------------------------------------
relay1_params = [
    "R1-PA1:VH", "R1-PM1:V", "R1-PA2:VH", "R1-PM2:V",
    "R1-PA3:VH", "R1-PM3:V", "R1-PA4:IH", "R1-PM4:I",
    "R1-PA5:IH", "R1-PM5:I", "R1-PA6:IH", "R1-PM6:I",
    "R1-PA7:VH", "R1-PM7:V", "R1-PA8:VH", "R1-PM8:V",
    "R1-PA9:VH", "R1-PM9:V", "R1-PA10:IH", "R1-PM10:I",
    "R1-PA11:IH", "R1-PM11:I", "R1-PA12:IH", "R1-PM12:I",
    "R1:F", "R1:DF", "R1-PA:Z", "R1-PA:ZH", "R1:S"
]

relay2_params = [
    "R2-PA1:VH", "R2-PM1:V", "R2-PA2:VH", "R2-PM2:V",
    "R2-PA3:VH", "R2-PM3:V", "R2-PA4:IH", "R2-PM4:I",
    "R2-PA5:IH", "R2-PM5:I", "R2-PA6:IH", "R2-PM6:I",
    "R2-PA7:VH", "R2-PM7:V", "R2-PA8:VH", "R2-PM8:V",
    "R2-PA9:VH", "R2-PM9:V", "R2-PA10:IH", "R2-PM10:I",
    "R2-PA11:IH", "R2-PM11:I", "R2-PA12:IH", "R2-PM12:I",
    "R2:F", "R2:DF", "R2-PA:Z", "R2-PA:ZH", "R2:S"   # Overall training did NOT include "R2:S"
]

relay3_params = [
    "R3-PA1:VH", "R3-PM1:V", "R3-PA2:VH", "R3-PM2:V",
    "R3-PA3:VH", "R3-PM3:V", "R3-PA4:IH", "R3-PM4:I",
    "R3-PA5:IH", "R3-PM5:I", "R3-PA6:IH", "R3-PM6:I",
    "R3-PA7:VH", "R3-PM7:V", "R3-PA8:VH", "R3-PM8:V",
    "R3-PA9:VH", "R3-PM9:V", "R3-PA10:IH", "R3-PM10:I",
    "R3-PA11:IH", "R3-PM11:I", "R3-PA12:IH", "R3-PM12:I",
    "R3:F", "R3:DF", "R3-PA:Z", "R3-PA:ZH", "R3:S"
]

relay4_params = [
    "R4-PA1:VH", "R4-PM1:V", "R4-PA2:VH", "R4-PM2:V",
    "R4-PA3:VH", "R4-PM3:V", "R4-PA4:IH", "R4-PM4:I",
    "R4-PA5:IH", "R4-PM5:I", "R4-PA6:IH", "R4-PM6:I",
    "R4-PA7:VH", "R4-PM7:V", "R4-PA8:VH", "R4-PM8:V",
    "R4-PA9:VH", "R4-PM9:V", "R4-PA10:IH", "R4-PM10:I",
    "R4-PA11:IH", "R4-PM11:I", "R4-PA12:IH", "R4-PM12:I",
    "R4:F", "R4:DF", "R4-PA:Z", "R4-PA:ZH", "R4:S"
]

# ------------------------------------------------
# Overall Prediction Feature Set (Separate List)
# ------------------------------------------------
# Create a separate list containing all overall features.
overall_features = relay1_params + relay2_params + relay3_params + relay4_params + [
    "relay1_log", "relay2_log", "relay3_log", "relay4_log"
]
# Remove "R2:S" if present (since training overall model didn't include it)
overall_features = [feat for feat in overall_features if feat != "R2:S"]

# ------------------------------------------------
# In-Memory Dataset for Overall Relay Prediction
# ------------------------------------------------
if "user_data" not in st.session_state:
    st.session_state["user_data"] = pd.DataFrame(columns=overall_features)

# ------------------------------------------------
# Unique Graphs for Each Relay (Unchanged)
# ------------------------------------------------
def relay1_graph():
    try:
        analytics_df = pd.read_csv("analytics.csv")
        if "marker" in analytics_df.columns:
            st.markdown("**Marker Distribution (Bar Chart)**")
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.countplot(data=analytics_df, x="marker", ax=ax)
            ax.set_title("Event Types Distribution")
            st.pyplot(fig)
        else:
            st.warning("Column 'marker' not found in analytics.csv.")
    except FileNotFoundError:
        st.warning("analytics.csv not found. Provide the file to view the bar chart.")

def relay2_graph():
    try:
        analytics_df = pd.read_csv("analytics.csv")
        if "R1:F" in analytics_df.columns:
            st.markdown("**Histogram of R1:F**")
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.histplot(analytics_df["R1:F"], kde=True, ax=ax)
            ax.set_title("R1:F Frequency")
            st.pyplot(fig)
        else:
            st.warning("Column 'R1:F' not found in analytics.csv.")
    except FileNotFoundError:
        st.warning("analytics.csv not found. Provide the file to view the histogram.")

def relay3_graph():
    try:
        analytics_df = pd.read_csv("analytics.csv")
        st.markdown("**Bar Chart: Average PM1:V per Relay**")
        relay_pm1 = {}
        for r in ["R1", "R2", "R3", "R4"]:
            col_name = f"{r}-PM1:V"
            if col_name in analytics_df.columns:
                relay_pm1[r] = analytics_df[col_name].mean()
        if relay_pm1:
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.barplot(x=list(relay_pm1.keys()), y=list(relay_pm1.values()), ax=ax)
            ax.set_title("PM1:V by Relay")
            st.pyplot(fig)
        else:
            st.warning("Required PM1:V columns not found in analytics.csv.")
    except FileNotFoundError:
        st.warning("analytics.csv not found. Provide the file to view average PM1:V chart.")

def relay4_graph():
    try:
        analytics_df = pd.read_csv("analytics.csv")
        if all(col in analytics_df.columns for col in ["R1-PA1:VH", "R1-PM1:V"]):
            st.markdown("**Joint Plot: R1-PA1:VH vs R1-PM1:V**")
            jp = sns.jointplot(data=analytics_df, x="R1-PA1:VH", y="R1-PM1:V",
                               kind="scatter", height=3)
            jp.fig.suptitle("R1-PA1:VH vs R1-PM1:V", y=1.02)
            st.pyplot(jp.fig)
        else:
            st.warning("Columns 'R1-PA1:VH' or 'R1-PM1:V' not found in analytics.csv.")
    except FileNotFoundError:
        st.warning("analytics.csv not found. Provide the file to view the joint plot.")

def show_all_four_graphs():
    st.markdown("### 1) Marker Distribution (Bar Chart)")
    relay1_graph()
    st.markdown("### 2) Histogram of R1:F")
    relay2_graph()
    st.markdown("### 3) Bar Chart of Average PM1:V per Relay")
    relay3_graph()
    st.markdown("### 4) Joint Plot: R1-PA1:VH vs R1-PM1:V")
    relay4_graph()

# ------------------------------------------------
# Helper: Show Relay Input & Predict Button for Individual Pages
# ------------------------------------------------
def relay_input_section(relay_params, relay_key_prefix, relay_title, model_to_use):
    st.subheader(relay_title)
    inputs = {}
    for param in relay_params:
        if param.endswith(":S"):
            inputs[param] = st.number_input(param, value=0, step=1, key=f"{relay_key_prefix}_{param}")
        else:
            inputs[param] = st.number_input(param, value=0.0, key=f"{relay_key_prefix}_{param}")
    if st.button(f"Predict {relay_title}", key=f"predict_{relay_key_prefix}"):
        input_df = pd.DataFrame([inputs])
        input_df = input_df.astype(float)
        prediction = model_to_use.predict(input_df)[0]
        result = "Attack" if prediction != 0 else "Natural"
        st.success(f"{relay_title} Prediction: **{result}**")

# ------------------------------------------------
# Sidebar Options for Individual Relay Predictions
# ------------------------------------------------
if option == "Relay1 Prediction":
    relay_input_section(relay1_params, "r1", "Relay 1", model)
    relay1_graph()
elif option == "Relay2 Prediction":
    relay_input_section(relay2_params, "r2", "Relay 2", rf_model)
    relay2_graph()
elif option == "Relay3 Prediction":
    relay_input_section(relay3_params, "r3", "Relay 3", rf_3model)
    relay3_graph()
elif option == "Relay4 Prediction":
    relay_input_section(relay4_params, "r4", "Relay 4", rf_4model)
    relay4_graph()

# ------------------------------------------------
# Overall Relay Prediction Section (Using Overall Feature Set)
# ------------------------------------------------
else:
    st.subheader("Overall Relay Prediction")
    st.write("Enter all overall features for prediction. Ensure all fields are completed.")
    
    overall_inputs = {}
    for feature in overall_features:
        overall_inputs[feature] = st.number_input(feature, value=0.0, key=f"overall_{feature}")
    
    if st.button("Predict Overall"):
        input_df = pd.DataFrame([overall_inputs])
        input_df = input_df.astype(float)
        prediction = overall_model.predict(input_df)[0]
        result = "Attack" if prediction != 0 else "Natural"
        st.success(f"Overall Prediction: **{result}**")
    
    st.write("### Overall Input Data")
    st.dataframe(pd.DataFrame([overall_inputs]))
    
    st.header("Analytics (All 4 Graphs)")
    show_all_four_graphs()
