import streamlit as st 
import base64


### FUCNTION TO DISPLAY THE PDF WHEN UPLOADED IN UI ###
def displayPDF(file,width=700,height=750):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width=100% height="{height}" type="application/pdf">'
    #pdf_display = F'<div><iframe src="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width: 100%;></iframe></div>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)