import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from langchain_community.vectorstores import FAISS
from docx import Document
from pptx import Presentation

# Load API key and configure Google Generative AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_word_text(word_docs):
    text = ""
    for doc in word_docs:
        document = Document(doc)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    return text

def get_ppt_text(ppt_docs):
    text = ""
    for ppt in ppt_docs:
        presentation = Presentation(ppt)
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_name)

def get_conversational_chain():
    prompt_template = """
Based on the context provided, answer the following question in detail. If the answer is not present in the context, state that clearly.
Context:\n {context}\n
Question:\n {question}\n
Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_pdf_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        return "No relevant documents found. Please ask a different question."
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def handle_word_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index_word", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        return "No relevant documents found. Please ask a different question."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def handle_ppt_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index_ppt", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        return "No relevant documents found. Please ask a different question."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_gemini_response(user_input_text, image=None):
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat_session = model.start_chat(history=[])
    
    if user_input_text:
        if image:
            response = chat_session.send_message([user_input_text, image])
        else:
            response = chat_session.send_message(user_input_text)
    elif image:
        response = chat_session.send_message(image)
    else:
        response = "No input provided."
    return response.text

def main():
    
    st.set_page_config(page_title="Royce AI")
    st.header("Royce AI 🤖")
    # Indicator for uploading documents
    st.markdown(
        """
        <style>
            .upload-indicator {
                font-size: 20px;
                font-weight: bold;
                color: #FFFFFF; /* For text on dark backgrounds */
                background-color: #1E1E1E; /* Dark background */
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
        <div class="upload-indicator">
            Please use the sidebar to upload your Image, PDF, Word, and PowerPoint files!😄
            (If you don't find it, Check for a Small arrow at the top left corner of your device)
        </div>
        """, unsafe_allow_html=True
    )


    # Sidebar for PDF, Word, and PowerPoint upload
    with st.sidebar:
        st.title("Upload Section")

        # PDF Upload
        pdf_docs = st.file_uploader("Upload your PDF Files",type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
        if st.button("Upload PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, "faiss_index")  # Specify index name for PDFs
                    st.success("PDFs processed and stored for search.")
            else:
                st.warning("Please upload PDF files.")

        # Word Upload
        word_docs = st.file_uploader("Upload your Word Files", type=["docx"], accept_multiple_files=True, key="word_uploader")
        if st.button("Upload Word Files"):
            if word_docs:
                with st.spinner("Processing Word files..."):
                    raw_text = get_word_text(word_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, "faiss_index_word")  # Specify index name for Word files
                    st.success("Word files processed and stored for search.")
            else:
                st.warning("Please upload Word files.")

        # PowerPoint Upload
        ppt_docs = st.file_uploader("Upload your PowerPoint Files", type=["pptx"], accept_multiple_files=True, key="ppt_uploader")
        if st.button("Upload PowerPoint Files"):
            if ppt_docs:
                with st.spinner("Processing PowerPoint files..."):
                    raw_text = get_ppt_text(ppt_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, "faiss_index_ppt")  # Specify index name for PowerPoint files
                    st.success("PowerPoint files processed and stored for search.")
            else:
                st.warning("Please upload PowerPoint files.")

        # Image Upload
        uploaded_image = st.file_uploader("Upload an image (Optional)", type=["jpeg", "jpg", "png"], key="image_uploader")
        if st.button("Upload Image"):
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            else:
                st.warning("Please upload an image.")

    # Single input box for both query and chat
    user_input_text = st.text_input("Ask a question or chat with Royce")

    # Initialize image variable
    image = None
    if uploaded_image:
        image = Image.open(uploaded_image)

    if st.button("Ask Royce"):
    # Check if any documents were uploaded
      if pdf_docs or word_docs or ppt_docs:
          # Search PDFs
          if pdf_docs:
              with st.spinner("Searching PDFs..."):
                  pdf_response = handle_pdf_query(user_input_text)
                  if pdf_response:
                      st.subheader("PDF Response:")
                      st.write(pdf_response)
                  else:
                      st.subheader("PDF Response:")
                      st.write("No relevant information found in the PDF.")

          # Search Word files
          if word_docs:
              with st.spinner("Searching Word files..."):
                  word_response = handle_word_query(user_input_text)
                  if word_response:
                      st.subheader("Word Response:")
                      st.write(word_response)
                  else:
                      st.subheader("Word Response:")
                      st.write("No relevant information found in the Document provided.")

          # Search PowerPoint files
          if ppt_docs:
              with st.spinner("Searching PowerPoint files..."):
                  ppt_response = handle_ppt_query(user_input_text)
                  if ppt_response:
                      st.subheader("PowerPoint Response:")
                      st.write(ppt_response)
                  else:
                      st.subheader("PPT Response:")
                      st.write("No relevant information found in the Document provided.")

          # Prompt for further input if user_input_text is empty
          if not user_input_text:
              st.warning("You can ask questions about the uploaded documents or chat normally.")
      elif uploaded_image:
          # If only an image is uploaded
          chat_response = get_gemini_response(user_input_text, image)
          st.subheader("Royce's Response:")
          st.write(chat_response)
      else:
          # If no documents or image are uploaded
          if user_input_text:
              chat_response = get_gemini_response(user_input_text, image)
              st.subheader("Royce's Response:")
              st.write(chat_response)
          else:
              st.warning("Please upload files or ask a question.")



if __name__ == "__main__":
    main()
