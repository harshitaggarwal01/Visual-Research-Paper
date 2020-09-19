import streamlit as st
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import cv2
import os
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Reasearch Papers are BORING!")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

OUTPUT_FOLDER="C:/Users/Harshit/Desktop/python/MICHACK/pdfimages"

images = convert_from_bytes(uploaded_file.read(),output_folder=OUTPUT_FOLDER) 

# index = 1
# for image in images:
#     image.save("page_" + str(index) + ".jpg")
#     index += 1


pdfimages = []
for filename in os.listdir(OUTPUT_FOLDER):
    img = cv2.imread(os.path.join(OUTPUT_FOLDER,filename))
    if img is not None:
        pdfimages.append(img)
        st.image(img)

"images", pdfimages