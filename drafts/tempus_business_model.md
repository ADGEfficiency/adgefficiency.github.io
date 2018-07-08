# Tempus availability fee business model - first look

Tempus is paid an availability fee every month, conditional on meeting KPIs.

Advantages of to Origin
- certantity about the cost of using the Tempus technology (the max upper limit is if we hit every KPI)
- all of the upside (if we avoid a `$14,000` spike then Origin gets 100% of the benefit beyond the availability fee)

Advantages to Tempus
- potential to not take risk on price volatility 
- stable revenue
- clear understanding of what we need to do (the KPI)
- calculating a fair availability fee requires only knowing our total costs & the number and size of customers

## problems with the baseline approach

1. tension betweeen data collection and delivering value
If we flex we lose the oppourtunity to learn what the baseline would have been.  Having to balance between data generation and revenue is not ideal.  Tempus should aim to be able to flex straight away, generating high quality and useful data no matter what.

2. customer data is heavily non iid

iid is a fundamental assumption made in statistical learning, that data is independently sampled and identically distributed.

Most customer sites don't have historical data.  After we setup data collection our sampling is not indepedent, it is only for the current month.  Our sampling is not spread independently throughout the year.  Likewise the distribution of the data we have is not the same as the true distribution.  We don't have data for what a hot summer will look like.

3. unexplainable variation in customer data

Occupancy is a key variable behind demand for cooling.  If we don't have any variables (i.e. a timetable) to use to predict high demand.  Some sites may have a timetable, some won't. 

## Key performance indicators (KPI)

- avoid consuming any electricity during the five highest price periods each day/week/month
- a gurantee on maximum average cost `$/MWh` to supply a customer across a day/week/month

KPI's can be aggregated across different time periods - the choice of time period is a key decision.

## non financial benefits 

Customer engagement & reducing churn.

The value of the customer data we are collecting improves over time
1. data is more valuable when there is more of it (i.e. an entire year is more than 12x valuable than 1 month)
2. the future will bring more valuable tools & approaches to extract infomation from this data

