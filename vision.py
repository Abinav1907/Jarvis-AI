import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


import streamlit as st
model = genai.GenerativeModel('gemini-1.5-flash')
chat_session = model.start_chat(
    history=[]
)
def get_gemini_response(text, image=None):
    if text:
        if image:
            response = chat_session.send_message([text, image])
        else:
            response = chat_session.send_message(text)
    elif image:
        response = chat_session.send_message(image)
    else:
        response = "No input provided."
    return response.text
st.set_page_config(page_title="Royce AI")
st.header("Royce AI")
input = st.text_input("Input: ", key="input") 
uploaded_file = st.file_uploader("Choose an Image (if required)...",type=["jpeg","jpg",'png'])
image = ""
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Ask Royce")



if submit:
  response = get_gemini_response([input,image])
  st.subheader("The Response is")
  st.write(response)
