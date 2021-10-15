import pandas as pd
import yfinance as yf
import streamlit as st
from PIL import Image
import tkinter
import base64

from datetime import datetime
from datetime import date

import mplfinance as fplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
    
from sklearn import preprocessing
from sklearn import metrics
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def app():

#having todays date

    local_dt = datetime.now().date()

    st.write("""
    # Stock Price prediction App
    """)


    #uploat dataset updated everyday from yahoofinance
    tickerSymbol = 'AMZN'
    #get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    #get the historical prices for this ticker
    df= tickerData.history(period="max")
# Open	High	Low	Close	Volume	Dividends	Stock Splits


    image = Image.open('amazon.jpg')
    st.image(image, caption='Amazon stock market')




    def get_table_download_link_csv(df):
  
        csv = df.to_csv().encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="amazon.csv" target="_blank">Download amazon trades up to today</a>'
        return href

    st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

#importing linear regression tools

#importing visualisation tool
    import matplotlib.pyplot as plt



#creating a variable (all the values we will be giving to a model)
    x=df[['Open']].values
#creating the variable we are trying to predict (the adjusted closing price will give you a better idea of the overall value of the stock)
    y=df[['Close']].values



#splitting our dataset to training and testing datasets keeping order
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state = 42,shuffle=False)

#creation the variable regressor and adjusting it with our train set
    regressor= LinearRegression()
    regressor.fit(x_train,y_train)

#creation the variable predicted and giving it the values of the test set
    predicted= regressor.predict(x_test)


    df_test=df.tail(predicted.shape[0])

    df_test=df_test[['Close']]
    df_test=df_test.rename(columns={'Close': 'Actual'})
    df_test.insert(1, 'Predicted', predicted, True)


 



    st.sidebar.header('choose a period to compare actual price to the predicted one')




    import pandas as pd

    df1=df.tail(50)



    def show(d):
            show= fplt.plot( d, type='candle', style='charles', title='amazon price stock chart in candle up to today', ylabel='Price ($)',     volume=True, ylabel_lower='closing', show_nontrading=True)
            return(show)
    a=show(df1)
    st.pyplot(a)

    st.set_option('deprecation.showPyplotGlobalUse', False)




    st.write("""
   # Shown are the stock closing price real and predicted of amazon!""", unsafe_allow_html=True
   )



#having a period of time given by the user it ends by today 

    date1 = date(int(2016), int(7), 10)
    date2 = local_dt
    d3 = st.sidebar.date_input("", [], min_value=date1 , max_value=date2 , key=None)
    if(len(d3)==2):
        df_toshow=df_test.loc[d3[0]:d3[1], :]
        df_toshow.reset_index(inplace=True)
        df_toshow=df_toshow.loc[:, ['Actual','Predicted']]

    #sdf_toshow.index=df_toshow.index.date()
        st.sidebar.write(df_toshow)
    


    st.sidebar.write("today is: ", local_dt)
#st.sidebar.write(local_dt)

    dt=df_test.tail(1)
    a=dt.values[0][0]
    st.sidebar.write("the market opened on: ", a)


    end=dt.values[0][0].reshape(-1, 1)
    end= regressor.predict(end)
    b=end[0][0]
    if st.sidebar.button('show prodicted closing for today'):
        st.sidebar.write(b)
        if(a>b):
            st.sidebar.write("the stock price is down dont buy today!")
        else:
            st.sidebar.write("the stock price is up buy today!")

    
    
    st.line_chart(df_test)

    

    if st.sidebar.button('show prodictions for'):
        st.sidebar.write(b)
        if(a>b):
            st.sidebar.write("the stock price is down dont buy today!")
        else:
            st.sidebar.write("the stock price is up buy today!")
    st.write("""
    # Now the future predictions!
    """)

            
            
    def future(n):
        df['label']=df['Close'].shift(-n)
        data= df.drop(['label'], axis=1)
        X=data.values
        X=preprocessing.scale(X)
        X=X[:-n]
        df.dropna(inplace=True)
        target=df.label
        Y= target.values
        X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state = 42,shuffle=False)
        regr= LinearRegression()
        regr.fit(X_train,y_train)
        regr.score(X_test,y_test)
        X_predict=X[-n:]
        forecast= regr.predict(X_predict)
        Date=np.array(df.index)
        last_Date=Date[len(Date)-1]
        trangle= pd.date_range('2021-05-21',periods=n,freq='d')
        predict_df=pd.DataFrame(forecast, index=trangle)
        predict_df.columns=['forecast']
        return predict_df
    v = st.number_input('give period in days to predict', value=1)
    st.area_chart(future(v))
    
    def get_table_download_link_csv(d):
    #csv = df.to_csv(index=False)
        csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="amazon:PredVSAct.csv" target="_blank">Download csv next month predictions</a>'
        return href
    dp=future(30)
    d=get_table_download_link_csv(dp)
    st.markdown(get_table_download_link_csv(d), unsafe_allow_html=True)



    main_bg = "ima.jpg"
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True
    )















