import os

import streamlit as st

from utilities.helper_functions import get_pdf

# set up file structure
cwd = os.getcwd()
pdf_name = 'MOPTA_AIMMS_2023.pdf'
pdf_path = os.path.join(cwd, pdf_name)

# Page
st.set_page_config(page_title='ChargED - Documentation', page_icon=':closed_book:')
st.title('Documentation')
st.write('Read, and download our report below.')


st.markdown(get_pdf(pdf_path), unsafe_allow_html=True)

with open(pdf_path, "rb") as f:
    st.download_button(label='Download', data=f, file_name='MOPTA-Team_ChargED.pdf')