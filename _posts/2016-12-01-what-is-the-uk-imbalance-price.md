---
id: 204
title: What is the UK Imbalance Price?
date: 2016-12-01T12:56:08+00:00
author: Adam Green
layout: post
guid: http://adgefficiency.com/?p=204
permalink: /what-is-the-uk-imbalance-price/
categories:
  - Electricity Price Forecasting

---

This post is the start of a series applying machine learning to solving an energy problem.  The goal of this series is to develop supervised machine learning models to forecast the UK Imbalance Price.

- Introduction - [What is the Imbalance Price](http://adgefficiency.com/what-is-the-uk-imbalance-price/)
- Getting Data - [Scraping the ELEXON API](http://adgefficiency.com/elexon-api-web-scraping-using-python/)

### What is the Imbalance Price?

The Imbalance Price is the price of electricity that generators or suppliers pay for imbalance on the UK electricity grid.

In the UK generators and suppliers (known as Parties) contract with each other for the supply of electricity.  Generators sell electricity to suppliers who then sell electricity to residential, commercial and industrial customers.

As System Operator National Grid handles real time balancing of the UK grid.  Parties submit details of their contracts to National Grid one hour before delivery.  This allows National Grid to understand the expected imbalance.

National Grid will then take actions to correct any predicted imbalance.  For example the Balancing Mechanism allows Parties to submit Bids or Offers to change their position by a certain volume at a certain price.

National Grid also has the ability to balance the system using actions outside the Balancing Mechanism, such as:
- Short Term Operating Reserve.
- Frequency Response plants used to balance real time.
- Reserve Services.
- In more drastic scenarios National Grid may call upon closed power plants or disconnect customers.  

National Grid attempts to minimize balancing costs within technical constraints.  Parties submit their expected positions one hour before delivery.  For a number of reasons parties do not always meet their contracted positions.

A supplier may underestimate their customers demand.  A power plant might face an unexpected outage.  The difference between the contracted and actual position is charged using the Imbalance Price.

ELEXON uses the costs that National Grid incurs in correcting imbalance to calculate the Imbalance Price.  This is then used to charge Parties for being out of balance with their contracts. ELEXON details the process for the [calculation of the Imbalance Price here](https://www.elexon.co.uk/reference/credit-pricing/imbalance-pricing/).

### What data is available?

A key reason behind the stunning success of modern supervised machine learning is the [availability of training data](http://adgefficiency.com/machine-learning-in-energy-part-one/).  Understanding how to apply supervised machine learning depends on what data is available.  

Data for the UK grid is available through the [ELEXON API](https://www.elexon.co.uk/change/new-balancing-mechanism-reporting-service-bmrs/).  

### Next steps

[The next post in this series develops the Python code to scrape data from the ELEXON API](http://adgefficiency.com/elexon-api-web-scraping-using-python/).
