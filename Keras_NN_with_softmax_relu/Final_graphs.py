## Connecting to fxcm 

import fxcmpy
import datetime as dt
import pandas as pd
import time
import plotly.graph_objects as go
import chart_studio
import chart_studio.plotly as py
from datetime import datetime
import chart_studio

def make_plot(dataframe,name):
    username = 'suvir6'
    api_key = '1Zma9SqWldYavKdgI1Fl'
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    new_name = name[:3] + name[4:]
    df = dataframe.iloc[:]
    df['Date'] = df.index
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['bidopen'],
                    high=df['bidhigh'],
                    low=df['bidlow'],
                    close=df['bidclose'])])
    fig.update_layout(
        title=name + " Live data",
        xaxis_title="Date",
        yaxis_title="Price",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    
    py.plot(fig, filename = new_name + ' 4 Hour data', auto_open=False)



if __name__ == '__main__':
    lister = ['EUR/USD', 'GBP/USD', 'EUR/AUD', 'AUD/USD']
    token = "a051ed2f89fe5ae6c3c2a0070af4243cb55d7357"
    con = fxcmpy.fxcmpy(access_token=token, log_level='error', server='demo', log_file= None)

    print("Fxcm api is connected!")
    while True:
		#now = datetime.now()
        for name in lister:
            data = con.get_candles(name, period = 'H4', number = 100)
            make_plot(data, name)
		    
        print(name + " graph has been updated in PHP!")
        time.sleep(3600)