---
title: 'What is the UK Imbalance Price?'
date: 2016-12-01
date_updated: 2022-03-24
categories:
  - Electricity Price Forecasting
classes: wide
excerpt: An introduction to how the UK recovers electricity grid balancing costs.

---

## What is the Imbalance Price?

The *Imbalance Price* is the price of electricity that generators or suppliers pay for imbalance on the UK electricity grid.

In the UK generators and suppliers (known as parties) contract with each other for the supply of electricity.  Generators sell electricity to suppliers who then sell electricity to residential, commercial and industrial customers.

As System Operator National Grid handles real time balancing of the UK grid.  Parties submit details of their contracts to National Grid one hour before delivery - allowing National Grid to understand the expected imbalance.

National Grid will then take actions to correct any predicted imbalance.  Balancing Mechanism allows parties to submit bids or offers to change their position by a certain volume at a certain price - National Grid will select from these bids and offers to balance the grid in a safe, low cost way.

## Other UK Grid Support Services

National Grid also has the ability to balance the system using actions outside the Balancing Mechanism, such as:

- Short Term Operating Reserve (STOR),
- Frequency Response plants used to balance real time,
- Reserve Services,
- In more drastic scenarios National Grid may call upon closed power plants or disconnect customers.  

National Grid attempts to minimize balancing costs within technical constraints.  Parties submit their expected positions one hour before delivery.  For a number of reasons parties do not always meet their contracted positions.

A supplier may underestimate their customers demand.  A power plant might face an unexpected outage.  The difference between the contracted and actual position is charged using the Imbalance Price.

Elexon uses the costs that National Grid incurs in correcting imbalance to calculate the Imbalance Price.  This is then used to charge parties for being out of balance with their contracts. Elexon details the process for the [calculation of the Imbalance Price here](https://www.elexon.co.uk/reference/credit-pricing/imbalance-pricing/).


## What data is available?

Data for the UK grid is available through the [ELEXON API](https://www.elexon.co.uk/change/new-balancing-mechanism-reporting-service-bmrs/) - [see here for a guide on how to access the Elexon API data in Python](https://adgefficiency.com/elexon-api-uk-electricity-grid-data-with-python/).
