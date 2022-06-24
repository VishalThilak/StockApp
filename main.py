import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

@st.cache
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
    Create and plot a bar chart showing the sentiment of a stock on various days
    param: ticker of company
    return: nothing
    '''
    # Website from which the stock news data is extracted from
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    url = finwiz_url + ticker

    # Requesting the html data
    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    
    # Opening the request
    response = urlopen(req)

    # Passing the response to get the source code of the website
    html = BeautifulSoup(response)

    # The news artile titles in the html in the id "tab-link-news"
    news_table = html.find(id='news-table')

    # List to keep track of the headlines and dates
    news_parsed = []

    # Interate through all the tr tags
    for row in news_table.findAll('tr'):
        # News headlines
        headline = row.a.get_text()    

        # Date
        date_text = row.td.text.split(' ')

        # If length is only 1 that means theres only time
        if len(date_text) == 1: 
            time = date_text[0]

        else:
            date = date_text[0]
            time = date_text[1]

        # Appended to the news_parsed list
        news_parsed.append([date, headline, time])

    # Sentiment Analyses
    vader = SentimentIntensityAnalyzer()
    dataframe = pd.DataFrame(news_parsed, columns=['date', 'headline', 'time'])
    headlines = dataframe['headline']
    scores = (headlines).apply(vader.polarity_scores).tolist()
    dataframe = dataframe.join(pd.DataFrame(scores), rsuffix=' right')

    # Convert the date to datatime year-month-date
    dataframe['date'] = pd.to_datetime(dataframe.date).dt.date

    # Average the scores for the same dates
    mean_scores = dataframe.groupby(['date']).mean()
    mean_scores.reset_index(inplace=True)

    # Droping everything but date and compoud
    mean_scores.drop(['neg', 'neu', 'pos'], axis=1, inplace=True)

    # Multiplying all the sentiment scores for 100 to make it more visible
    mean_scores['compound'] = 100 * mean_scores['compound']
    
    # Adding column color for if its negative or positive
    mean_scores['Color'] = np.where(mean_scores['compound']<0, 'red', 'blue')

    # Creating the bar chart
    st.subheader("Sentiment Chart")
    st.caption("No bar means neutral assessment")
    fig = go.Figure(go.Bar(x=mean_scores['date'], y=mean_scores['compound'], marker_color=mean_scores['Color']))
    fig.update_layout(barmode='relative', width=740, height=500)
    fig.update_xaxes(title_text='<b>Date</b>')
    fig.update_yaxes(title_text='<b>Overall Sentiment</b>')
    fig.layout.update(title_text='<i>Sentiment Bar Chart</i>')
    fig.update_layout(width=1400, height=500)
    st.plotly_chart(fig)
    
    # Showing all the headlines and dates
    st.subheader("Headlines of News Articles")
    df = dataframe[["date", "headline"]]
    st.dataframe(df)

# Run functions
custimize_streamlit()
stock = selected_stock()
ticker_index = comp_list.index(stock)
market_data = history(tickers_list[ticker_index])
sentiment_chart(tickers_list[ticker_index])
current_graph(market_data)
pred_graph(market_data)

