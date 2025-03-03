import streamlit as st
import numpy as np
from perceptron import Perceptron, PerceptronSoftmax

# *********************** Util Function
def add_data(training_data_input, target_input):
    st.session_state.target_data.append(target_input)
    st.session_state.training_data.append(training_data_input)

def create_model():
    training_data = np.array(st.session_state.training_data)
    target_data = np.array(st.session_state.target_data)
    PerceptronModel = Perceptron(training_data, target_data)
    return PerceptronModel

def train_step(): 
    message = st.session_state.perceptron_model.train_step()
    if message:
        st.session_state.model_converge = message
        

def train():
    message = st.session_state.perceptron_model.train()
    if message:
        st.session_state.model_converge = message
    
def set_up_training_data():
    if not st.session_state.is_setup_training_data:
        st.session_state.is_setup_training_data = True
        st.session_state.perceptron_model = create_model()

def predict():
    prediction = st.session_state.perceptron_model.predict([st.session_state.X1_test, st.session_state.X2_test])
    st.session_state.predict_result = prediction


# *********************** Streamlit Section
st.title("Simulasi Model Perceptron untuk Pembelajaran Asosiatif")

# *************************** Setup Section
if "setup_status" not in st.session_state:
    st.session_state.setup_status = False
if "input_options" not in st.session_state:
    st.session_state.input_options = [1, -1]
st.header("Setup Model Perceptron")
st.selectbox("Pilih Jenis Data Input", ["Biner", "Bipolar"], key="input_type", disabled=st.session_state.setup_status)
st.number_input("Masukkan Threshold", min_value=0.1, max_value=1.0,value=0.2, key="threshold", disabled=st.session_state.setup_status)
st.number_input("Masukkan Learning Rate", min_value=0.1, max_value=1.0, value=0.1, key="learning_rate", disabled=st.session_state.setup_status)
st.button("Setup Model", key="setup_button",on_click=lambda: st.session_state.update({"setup_status": True}), disabled=st.session_state.setup_status)
if st.session_state.setup_status:
    st.write("Model Perceptron sudah di setup")
    if st.session_state.input_type == "Biner":
        st.session_state.input_options = [0,1]
    elif st.session_state.input_type == "Bipolar":
        st.session_state.input_options = [-1,1]
    st.write("threshold: ", st.session_state.threshold)
    st.write("learning_rate: ", st.session_state.learning_rate)
    st.write("input_type: ", st.session_state.input_type)

#*************************** Training Data Section
st.header("Training Model Perceptron")
st.subheader("Input Training Data")
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'target_data' not in st.session_state:
    st.session_state.target_data = []
if 'perceptron_model' not in st.session_state:
    st.session_state.perceptron_model = None  
if "is_setup_training_data" not in st.session_state:
    st.session_state.is_setup_training_data = False
if "model_converge" not in st.session_state:
    st.session_state.model_converge = None
if "predict_result" not in st.session_state:
    st.session_state.predict_result = None

col = st.columns(2)
with col[0]:
    cols = st.columns(2, gap="small")
    with cols[0]:
        x1 = st.selectbox("X1", key="x1", options=st.session_state.input_options, placeholder="Select X1", disabled=st.session_state.is_setup_training_data or not st.session_state.setup_status)
    with cols[1]:
        x2 = st.selectbox("X2", key="x2", options=st.session_state.input_options, placeholder="Select X2", disabled=st.session_state.is_setup_training_data or not st.session_state.setup_status)

with col[1]:
    target_input = st.selectbox("Select Target", [-1,1], placeholder="Select Target", key="target", disabled=st.session_state.is_setup_training_data or not st.session_state.setup_status)
    
st.button("Tambah Data", on_click=add_data, args=([x1,x2], target_input), disabled=st.session_state.is_setup_training_data or not st.session_state.setup_status)

if len(st.session_state.training_data) >=2 and -1 in st.session_state.target_data and 1 in st.session_state.target_data:
    st.button("Set Up Training Data", on_click=set_up_training_data, disabled=st.session_state.is_setup_training_data)

st.subheader("Training Data")
if st.session_state.training_data:
    for idx, data in enumerate(st.session_state.training_data):
        cols = st.columns([2, 1, 1], gap="small")
        with cols[0]:
            st.write(f"Training Data {idx + 1}:")
            st.write("X1: ", data[0])
            st.write("X2: ", data[1])
        with cols[1]:
            st.write(f"Target Data {idx + 1}:")
            st.write(st.session_state.target_data[idx])
        if st.button(f"Delete", key=f"delete_{idx}", disabled=st.session_state.is_setup_training_data):
            del st.session_state.training_data[idx]
            del st.session_state.target_data[idx]
            st.rerun()
else:
    st.write("No Data")

#*************************** Model Information
st.subheader("Model Perceptron Information")


if st.session_state.perceptron_model:
    cols = st.columns(2)
    st.write("Model Name:", st.session_state.perceptron_model.model_name)
    with cols[0]:
        st.write("Current Weights:", st.session_state.perceptron_model.get_weights())
    with cols[1]:
        st.write("Current Data:", st.session_state.perceptron_model.training_data[st.session_state.perceptron_model.current_data])
    st.write("Bias:", st.session_state.perceptron_model.get_bias())
    st.write("Epochs:", st.session_state.perceptron_model.get_epoch())
else:
    st.write("Please setup training data first")

#*************************** Training Section
st.subheader("Train Model")
is_button_disable =  st.session_state.perceptron_model is None or len(st.session_state.training_data) < 2

if is_button_disable:
    st.write("Minimal 2 data training untuk melatih model")

cols = st.columns([1,1,5], gap="small")
with cols[0]:
    st.button("Step", on_click=train_step, disabled=is_button_disable or bool(st.session_state.model_converge))
with cols[1]:
    st.button("Train", on_click=train, disabled=is_button_disable or bool(st.session_state.model_converge))

if st.session_state.model_converge:
    st.write(st.session_state.model_converge)

#*************************** Prediction Section
st.header("Model Test")
X1_test = st.selectbox("X1 Test", options=st.session_state.input_options, key="X1_test", disabled=is_button_disable)
X2_test = st.selectbox("X2 Test", options=st.session_state.input_options, key="X2_test", disabled=is_button_disable)

st.write("Prediction: ", st.session_state.predict_result)
st.button("Predict", on_click=predict, disabled=is_button_disable)