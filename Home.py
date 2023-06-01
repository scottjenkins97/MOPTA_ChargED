import streamlit as st
from pathlib import Path
import os
import markdown
cwd = os.getcwd()

st.set_page_config(page_title='ChargED', page_icon=':zap:')

st.title('MOPTA 2023')
st.subheader('Planning Electric Vehicle Charging Stations')
st.subheader('ChargED: Heidi Wolles, Scott Jenkins and Kilian Wolff, University of Edinburgh')
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

# Display Problem Introduction from markdown file
intro_markdown = read_markdown_file(os.path.join(cwd, "README.md"))
st.markdown(intro_markdown, unsafe_allow_html=True)

# To launch the interactive interface in streamlit, type the following into your terminal:
# streamlit run Home.py