import streamlit as st
def dataset_info():
    st.sidebar.info(\"\"\"
    **FER2013 Dataset Info**
    Original dataset: 287MB, 35,887 images
    - Download from Kaggle
    - Used for model training
    - Not included in deployment
    App uses pre-trained model.
    \"\"\")
