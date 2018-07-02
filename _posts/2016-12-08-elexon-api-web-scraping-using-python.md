---
title: Elexon API Web Scraping using Python
date: 2016-12-08
categories:
  - Machine Learning 
excerpt: Scraping UK grid data using Python

---
This post is the second in a series applying machine learning techniques to an energy problem.  The goal of this series is to develop models to forecast the UK Imbalance Price.  

- Introduction - [What is the Imbalance Price](http://adgefficiency.com/what-is-the-uk-imbalance-price/)
- Getting Data - [Scraping the ELEXON API](http://adgefficiency.com/elexon-api-web-scraping-using-python/)

---

This post shows how to scrape UK Grid data using the ELEXON API.  [If you just want to skip to the code - the script is here.](https://github.com/ADGEfficiency/electricity_price_forecasting/blob/master/elexon_data_scraping.py)  This worked is inspired by work from the [excellent Patrick Avis of the Energy Analyst](http://energyanalyst.co.uk/).  

ELEXON provides technical data such as generation, along with market volumes and prices. A full detail of available data is given in the [Elexon API guide](https://www.elexon.co.uk/guidance-note/bmrs-api-data-push-user-guide/).

Accessing data requires an API key, [available by setting up a free Elexon account](https://www.elexonportal.co.uk/registration/newuser).  The API is accessed by passing a URL with the API key and report parameters.  The API will return either an XML or a CSV document.

## scraping the ELEXON API

The functionality to scrape ELEXON data is held within a Python script `elexon_data_scraping.py`.  This script is located in the [GitHub repo for the electricity price forecasting project](https://github.com/ADGEfficiency/electricity_price_forecasting/blob/master/elexon_data_scraping.py).

I make use of the argparse library to send the API key into the script

```bash
$ cd electricity_price_forecasting
$ python elexon_data_scraping.py --key 'YOUR_API_KEY'
```

A class ReportGrabber uses the `requests` library to get the XML response from the ELEXON API.  The `xml` library is used to process and parse the XML object for the data we want into a dictionary.  

We then make use of `pandas` to create dataframes

```python
#  to join data across multiple days (single report)
all_dates = pd.concat(dataframes, axis=0)

#  to join data across multiple reports
all_reports = pd.concat(dataframes, axis=1)
```

A simple example of using the ReportGrabber class

```python
#  start python in interactive mode
$ python -i elexon_data_scraping.py --key 'YOUR_API_KEY'

#  create a ReportGrabber object for the imbalance price
>>> report = ReportGrabber('B1770', ['imbalancePriceAmountGBP'], APIKEY)

#  create a dictionary with the API data
>>> output_dict = report.scrape_report('2017-01-01')
scraping B1770 2017-01-01

#  create a dataframe
>>> output = report.create_dataframe(output_dict)

>>> output.head()
   time_stamp        imbalancePriceAmountGBP                      
2017-01-01 00:30:00                 40.00000
2017-01-01 01:00:00                 46.86500
2017-01-01 01:30:00                 40.24737
2017-01-01 02:00:00                 40.05000
2017-01-01 02:30:00                 40.01602
```

Thanks for reading!
