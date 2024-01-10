import streamlit as st

logo = 'image/logo.png'
st.sidebar.image(logo)
    # Menampilkan tulisan dengan ukuran teks yang berbeda
st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: greenyellow; font-size: 50px;">WELCOME TO SEPSIS RECOMMENDATION TREATMENT</h1>
            <p style="font-size: 18px;">This application provides recommendations for sepsis treatment based on several clinical parameters.</p>
        </div>
    """, unsafe_allow_html=True)
icon = "cast"
st.divider() 
    # Create a DataFrame from the provided data
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

st.write("This information hub is created using Streamlit. If you have any questions or feedback, please contact Anjong at student@anjong.ac.id..")
st.divider()     
st.write("""
Copyright © 2023 - SOLID, All Rights Reserved. This information hub is created using Streamlit. If you have any questions or feedback, please contact Anjong at student@anjong.ac.id..
""")
