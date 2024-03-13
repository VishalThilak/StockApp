# Stock Web Application

## Description

A stock web application that uses machine learning to predict the forcast of a stock. The user simple selects
the stock of their choosing, and then they see a timeline of the stocks trend. In addition, they will also see 
the current sentiment about stock. This is using Yahoo's machine learning model and is deployed using AWS lambdas.
In addition, I have created my own LSTM model attempting to predict the trend of a stock and seeing how accurate it is.

## Getting Started

### Dependencies

beautifulsoup4==4.12.3
numpy==1.24.4
pandas==2.0.3
plotly==5.20.0
prophet==1.1.5
Requests==2.31.0
streamlit==1.32.1
vaderSentiment==3.3.2
vaderSentiment==3.3.2
yfinance==0.2.35

### Installing

* Download all the files and run through vscode 
* Or simply run the program by going to the website: http://3.17.162.224:8501/
  
### Executing program

* After running the program in vscode the program is fairly intuitive
* Select any stock of your choosing in the drop down menu
* Let the site load the prediction
* Then see all the prediction graphs
* Theres even a slider to select a specific timeline

## Authors

Vishal Thilak

## Version History

* 0.1
    * Initial Release
 * 0.2
    * Added an LSTM model to attempt to create my own AI
 * 0.3
    * Deployment   
