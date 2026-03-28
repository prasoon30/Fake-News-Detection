import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lreg_model.jb')

st.title("FAKE NEWS DETECTOR")
st.write("Enter a news Article below to check weather it is Fake or Real. ")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    
    else:
        st.warning("Please enter some text to analyse. ")