
# Import necessary libraries
import json
import requests
from streamlit_lottie import st_lottie
import streamlit as st
import pandas as pd

    # Menampilkan tulisan dengan ukuran teks yang berbeda
st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: greenyellow; font-size: 50px;">WELCOME TO SEPSIS RECOMMENDATION TREATMENT</h1>
            <p style="font-size: 18px;">This application provides recommendations for sepsis treatment based on several clinical parameters.</p>
        </div>
    """, unsafe_allow_html=True)
def load_lottiefile(path: str):
    with open(path) as f:
        data = json.load(f)
    return data
lottie_file = 'assets/animasi.json'
lottie_json = load_lottiefile(lottie_file)
st_lottie(lottie_file, key="hello")
logo = 'assets/logo.png'
st.sidebar.image(logo)
icon = "cast"
st.divider() 
    # Create a DataFrame from the provided data
data = {
    'Label': ['case', 'Death'],
    '2017': [48.9, 11],
    'Seri 3': [None, None],  # Fill with your data
    'Seri 4': [None, None],  # Fill with your data
    'Seri 5': [None, None]   # Fill with your data
}

df = pd.DataFrame(data)

# Set the Label column as the index for better plotting
df.set_index('Label', inplace=True)

# Streamlit App
st.title('Charts from Data')


# Bar chart
st.bar_chart(df)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.write("## What is Sepsis?")
st.write("Sepsis is a severe illness caused by the body's response to an infection. It can lead to organ failure and, if not treated promptly, can be fatal. Early detection and intervention are crucial for successful treatment.")

st.write("## Symptoms of Sepsis")
st.write("Sepsis symptoms can vary, but common signs include fever, elevated heart rate, rapid breathing, confusion, and extreme discomfort. If you suspect sepsis, seek medical attention immediately.")

st.write("## Risk Factors")
st.write("Certain factors increase the risk of developing sepsis, including age, weakened immune system, chronic medical conditions, and recent surgery or invasive procedures.")

st.write("## Prevention and Treatment")
st.write("Preventing infections, practicing good hygiene, and seeking prompt medical attention for infections can help prevent sepsis. Treatment involves antibiotics, supportive care, and addressing the underlying cause.")

st.write("## Additional Resources")
st.write("For more detailed information and resources about sepsis, you can visit the following websites:")
st.write("- [Sepsis Alliance](https://www.sepsis.org/)")
st.write("- [World Health Organization (WHO) - Sepsis](https://www.who.int/news-room/questions-and-answers/item/sepsis)")
st.write("- [Centers for Disease Control and Prevention (CDC) - Sepsis](https://www.cdc.gov/sepsis/index.html)")

import streamlit as st
import matplotlib.pyplot as plt

# Data
labels = ['2017', 'Of all global deaths']
values = [80, 20]

# Colors with alpha values for transparency
#colors = ['rgba(255, 255, 255, 0.0)', 'rgba(255, 165, 0, 0.8)']  # White for 80, Orange for 20

# Streamlit App
st.title('Pie Chart from Data')

st.markdown(
    """
    <style>
        body {
            background-color: #001F3F; /* Dark blue color */
            color: white; /* Text color */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Plotting Pie Chart
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

# Display the pie chart without white background
st.pyplot(fig, clear_figure=True)


st.write("This information hub is created using Streamlit. If you have any questions or feedback, please contact [Your Name] at [Your Email Address].")
st.divider()     
st.write("""
Copyright © 2023 - SOLID, All Rights Reserved.
""")
