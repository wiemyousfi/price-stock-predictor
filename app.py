import streamlit as st
from multiapp import MultiApp
from apps import home
from apps import amazon
from apps import tesla
from apps import apple
from apps import microsoft
from apps import google



app = MultiApp()



st.markdown("""
# Trading App
This app is using the neural networks to predict stock prive 
""")




# Add all your application here
app.add_app("home", home.app)
app.add_app("amazon", amazon.app)
app.add_app("apple", apple.app)
app.add_app("tesla", tesla.app)
app.add_app("google", google.app)
app.add_app("microsoft", microsoft.app)



# The main app
app.run()