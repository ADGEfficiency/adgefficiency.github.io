---
id: 285
title: 'Negative effects of high return temperature'
date: 2016-12-16T11:12:55+00:00
author: Adam Green
layout: post
guid: http://adgefficiency.com/?p=285
permalink: /energy-basics-four-negative-effects-of-high-return-temperatures/
categories:
  - Energy Basics
---

High return temperatures are a major problem in district heating (DH) networks.  High return temperatures mean:

- Increased flow rate of water pumped around the network.
- Lowered capacity of the network to deliver heat.
- Increased heat losses.
- Decreased heat recovery from gas engines and biomass boilers.

Figure 1 shows a simple flow diagram for a district heating network.  The system delivers heat to the building heating system via a heat exchanger.  Hot water is pumped around the district heating network and then returned to the energy centre for heating.

![Figure 1]({{"/assets/return_temps/fig1.png"}})
**Figure 1 – District heating system operating with a return temperature of 50 °C**

What then are the four negative impacts of a high return temperature?

### Increased flow rate of water pumped around the network

Most district heating networks operate with a fixed flow temperature. This is set by the temperature of the water generated in boilers or CHP plants.

A high return temperature means that the temperature difference across the network (`TFLOW – TRETURN`) will decrease.  A smaller temperature difference means pumping more water to deliver the same amount of heat. [See this earlier post if you are not clear how this relationship works](http://adgefficiency.com/energy-basics-q-m-cp-dt/).

Pumping more water means more electricity consumed by the pumps. This means increased electricity cost and carbon emissions from the scheme.

### Lowered capacity of the network to deliver heat

Pipe sizes limit the capacity of a district heating network to deliver water.

At peak flow rate a small temperature difference means we can deliver much less heat than the same network with a high temperature difference.   A scheme with a temperature difference half of the design means we are doubling the effective capital cost of our network per MW of heat capacity.

A larger temperature difference means we may be able to avoid installing new pipework (and the associated capital cost!) as our network expands.  Design of new networks with large temperature differences would mean smaller pipes. Smaller pipes means less capital cost and lower heat losses.

### Increased heat losses

Heat losses are a function of the pipe surface area and the difference in temperature between the pipe and ambient. A higher return temperature means more heat losses in the return pipes.

Heat losses are a drawback of DH schemes versus local gas boilers. DH schemes lose a lot more heat due to the long length of the network pipes versus local systems. Minimizing heat losses is crucial in operating an efficient DH network.

Increased heat losses means more heat generation required in the energy centre. This means higher gas consumption and carbon emissions.

### Decreased heat recovery from gas engines and biomass boilers

District heating schemes bring a net benefit to customers and the environment by the use of low carbon generation in the energy centre.

The efficient use of technologies such as gas engines or biomass boilers is central to the success of district heating.  The benefits of using low carbon generation can offset heat lost from the DH network.

District heating schemes use gas engines to generate heat and power together.  Gas engines generate roughly half of their recoverable heat as hot exhaust gases (> 500 °C) and half as low temperature (<100 °C). Biomass boilers generate only a hot exhaust gas.

The thermodynamic reasons for the loss of heat recovery are the same for of these three heat sources.  An increased DH return temperature increases the final temperature the heat source can be cooled to.

This means that less heat is transferred between the heat source and the DH water.  Below we will look at the example of recovering gas engine low temperature heat.

Gas engines operate with a low temperature hot water circuit.  This water circuit removes the jacket water and lube oil from the engine.  This heat can generate hot DH water for use in the scheme.

Figure 2 shows that a network return temperature (85 °C) leads to us only being able to cool the engine circuit to 85 °C.  This limits heat recovery in the heat exchanger.

![Figure 2]({{"/assets/return_temps/fig2.png"}})
**Figure 2 – Gas engine low temperature waste heat recovery with a high return temperature**

It also forces us to use a dump radiator to cool the engine circuit to the 70 °C required by the engine.  If the scheme was not fitted with a dump radiator then the engine would be forced to reduce generation or shut down.

Figure 3 shows the temperature versus heat (T-Q) diagram for the heat exchanger when return temperature is low (50 °C).  Operating with a low return temperature means we recover a full 1 MW from the engine water circut.

![Figure 3]({{"/assets/return_temps/fig3.png"}})
**Figure 3 – Heat recovery from engine with a low network return temperature (50 °C)**

Now look what happens when return temperature is high (80 °C).  Figure 4 shows that we now only recover 400 kW of heat.

![Figure 4]({{"/assets/return_temps/fig4.png"}})
**Figure 4 – Heat recovery from engine with a high network return temperature (80 °C)**

Gas boilers will need to generate the additional 600 kW of heat required by the network.  This means increased gas consumption and carbon emissions.

The same principle applies to the recovery of heat from higher temperature sources such as gas engine exhaust or biomass boiler combustion products.  A high DH return temperature will limit heat recovery.

### Why do high return temperatures occur?

High network return temperatures can occur for variety of reasons.  Most commonly it is due to heating systems designed for local gas boilers connected to DH networks.

A major issue is the use of bypasses.  Bypasses divert a small amount of the hot DH water being fed to a heat exchanger directly from the flow into the return.  Figure 5 shows a bypass increasing network return temperature from 80 to 95 °C.

![Figure 5]({{"/assets/return_temps/fig5.png"}})
**Figure 5 – Bypass causing high return temperature**

Bypasses are installed to maintain a minimum amount of flow across the network when demand for heat is low.  This prevents starving pumps at low heat demands.

Bypasses cause no issues in local boiler building heating systems but are a major problem in district heating.

These bypasses are pipes designed to only allow a small amount of water to bypass the heat exchanger.  However when network flow is low they also have a proportionally large effect on the return temperature!

Instead of installing bypasses pump systems should operate with higher turndowns.  This can be achieved through multiple pump systems.

Another reason for high network return temperatures is building circuits which use higher temperature water than they require.  For example local hot water cylinders require temperatures above 60 °C to prevent legionella.

Local water storage does not make sense on a DH network – heat storage should occur in the energy centre.  This will allow the DH network operators to optimally manage the heat storage.

Local hot water cylinders can also cause peaks in demand if they are set to charge at the same time.  This will be seen as a huge peak in heat demand on the entire network.  Peak demands can be difficult for DH network operators to deal with.

Thanks for reading!
