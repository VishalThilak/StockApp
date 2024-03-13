import streamlit as st
from datetime import datetime, date
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.error import HTTPError

# Setting up page title and icon as the first command
st.set_page_config(page_title='Stock Quest', page_icon='📈', layout='wide')

# Reading the csv that contains all the company names and ticker symbols 
# and putting them into lists
tickers = pd.read_csv('Tickers_List.csv', usecols=['Symbol'])
tickers_list = tickers.Symbol.to_list()
companies  = pd.read_csv('Tickers_List.csv', usecols=['Name'])
comp_list = companies.Name.to_list()

def custimize_streamlit():
    '''
    Custimizing the existing presets in streamlit
    param: nothing
    return: nothing
    '''
    # Hiding the inbuilt streamlit menu and the "made by streamlit" text
    st.markdown(''' <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> ''', unsafe_allow_html=True)

    # Removing inbuilt streamlit red bar
    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
        </style>
    '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True) 

    # Decreaseing padding at the top
    st.write('<style>div.block-container{padding-top:0;}</style>', unsafe_allow_html=True)

    # Custimizing the font in the title 
    st.markdown('<h1 style="text-align:center;color:black;font-weight:bolder;font-size:100px;">Stock Quest</h1>',unsafe_allow_html=True)

def selected_stock():
    '''
    Creating dropdown menu from which users select the stock they want
    param: nothing
    return: stock
    '''
    stock = st.selectbox("Select a stock",comp_list)
    st.write("You selected: ", stock)
    return stock

@st.cache_data
def history(ticker):
    '''
    Generate history based on ticker
    param: ticker of company
    return: dataframe of the stock history
    '''
    # Abitrary date selected
    start_time = "2019-01-01"

    # End time is the current time
    end_time = date.today().strftime('%Y-%m-%d')

    # Download market data 
    df = yf.download(ticker, start=start_time, end=end_time)

    # Creates an index starting from zero
    df.reset_index(inplace=True)

    return df

def current_graph(df):
    '''
    Graph current stock prices
    param: dataframe of a stock
    return: nothing
    '''
    # Getting the closing prices and dates from the dataframe
    Closing = df['Close']
    Date = df['Date']

    # Create graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Date, y=Closing))
    fig.layout.update(xaxis_title="<b>Date</b>",yaxis_title="<b>Price</b>", title_text="<i>Current Stock Price</i>",xaxis_rangeslider_visible=True)
    fig.update_layout(width=1400, height=500)
    
    # Creating seperation line
    st.markdown("***")

    # Titling the graph
    st.subheader("Current Stock Price")
    st.plotly_chart(fig)

def pred_graph(df):
    '''
    Create and plot the prediction graph
    param: dataframe of a stock
    return: nothing
    '''
    # Titling the graph before the graph generates to let the user know that there is a prediction graph
    st.subheader("Prediction of Stock")
    
    # Training the model with the closed stock values
    train = df[['Close', 'Date']]
    train = train.rename(columns={"Date":"ds", "Close":"y"})

    # Initializing model
    m = Prophet(uncertainty_samples = 0)
    m.fit(train)

    # Make a prediction 200 days into the future
    future = m.make_future_dataframe(periods=200)
    forcast = m.predict(future)

    # Seperation line
    st.markdown("***")

    # Ploting prediction
    fig = plot_plotly(m, forcast)
    fig.layout.update(title_text='<i>Prediction of Stock</i>')
    fig.update_xaxes(title_text='<b>Date</b>')
    fig.update_yaxes(title_text='<b>Stock Price</b>')
    fig.update_layout(width=1400, height=500)
    st.plotly_chart(fig)

def sentiment_chart(ticker):
    '''
    Fetch and parse news headlines and timestamps for the given ticker, then perform sentiment analysis.
    If the webpage for the ticker cannot be found, outputs a message indicating no data is available.
    '''

    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker
    
    try:
        req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(req)

    except HTTPError as e:
        st.error(f"No data available for {ticker}. HTTP Error {e.code}: {e.reason}")
        return

    html = BeautifulSoup(response, 'html.parser')

    news_table = html.findAll('tr', class_='cursor-pointer has-label')
    news_parsed = []

    for row in news_table:

        timestamp = row.find('td', align='right').text.strip()
        headline = row.find('a', class_='tab-link-news').get_text().strip()
        news_parsed.append([timestamp, headline])

    if not news_parsed:
        
        st.error(f"No news data available for {ticker}.")
        return

    # Sentiment Analysis
    vader = SentimentIntensityAnalyzer()
    dataframe = pd.DataFrame(news_parsed, columns=['date', 'headline'])
    dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce').dt.date
    dataframe['compound'] = dataframe['headline'].apply(lambda x: vader.polarity_scores(x)['compound'])

    # Group by date and calculate mean sentiment score
    mean_scores = dataframe.groupby('date')['compound'].mean().reset_index()
    mean_scores['compound'] = 100 * mean_scores['compound']
    mean_scores['color'] = np.where(mean_scores['compound'] < 0, 'red', 'blue')

    # Sentiment Chart
    st.subheader("Sentiment Chart")
    st.caption("No bar means neutral assessment")
    fig = go.Figure(go.Bar(x=mean_scores['date'], y=mean_scores['compound'], marker_color=mean_scores['color']))
    fig.update_layout(barmode='relative', width=740, height=500)
    fig.update_xaxes(title_text='<b>Date</b>')
    fig.update_yaxes(title_text='<b>Overall Sentiment</b>')
    fig.layout.update(title_text='<i>Sentiment Bar Chart</i>')
    fig.update_layout(width=1400, height=500)
    st.plotly_chart(fig)

    # Showing all the headlines and dates
    st.subheader("Headlines of News Articles")
    df = dataframe[['date', 'headline']]
    st.dataframe(df)


# Run functions
custimize_streamlit()
stock = selected_stock()
ticker_index = comp_list.index(stock)
market_data = history(tickers_list[ticker_index])
sentiment_chart(tickers_list[ticker_index])
current_graph(market_data)
pred_graph(market_data)
