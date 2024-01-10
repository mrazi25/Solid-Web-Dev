# Import necessary libraries
import streamlit as st
from PIL import Image
# Page title
st.title("About Us")
img_Anjong = Image.open("assets/anjong.png")
img_naufal = Image.open("assets/naufal.png")
img_shougi = Image.open("assets/shougi.png")

# Introduction
st.write(
    "Welcome to the 'About Us' page. Get to know the individuals behind the Project."
)

# Using st.columns to create three columns
col1, col2, col3 = st.columns(3)

# Team member 1 in the first column
with col1:
    st.header("Anjong ")
    st.image(img_Anjong, caption="Back-end", use_column_width=True)
    st.link_button("Go to Instagram", "https://www.instagram.com/aan_yahanjong?igsh=MW1zeWRtcm9uZDIxOA%3D%3D&utm_source=qr")


# Team member 2 in the second column
with col2:
    st.header("Naufal")
    st.image(img_naufal, caption="Front-end", use_column_width=True)
    st.link_button("Go to Instagram", "https://www.instagram.com/naufal4929?igsh=MXUyNGs1M2F1aHR5dg==")

# Team member 3 in the third column
with col3:
    st.header("Shougi")
    st.image(img_shougi, caption="UI/UX", use_column_width=True)
    st.link_button("Go to Instagram", "https://www.instagram.com/shougi.bawahab/")

# Footer
st.write(
    "If you have any inquiries or would like to get in touch with our team, please email us at "
    "[contact@sepsisinfohub.com]. We appreciate your interest in our mission."
)
st.divider()     
st.write("""
Copyright © 2023 - SOLID, All Rights Reserved.
""")
