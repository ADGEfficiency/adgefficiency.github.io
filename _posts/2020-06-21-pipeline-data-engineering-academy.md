---
title: 'Warming up to data engineering on the Pipeline Summer Camp'
author: Adam Green
categories:
  - Data Science
excerpt: My experience being a part of Pipeline's week long data engineering adventure.

---

*Hello Peter :) - this is still a draf t- interested to get your thoughts on structure, content etc - nothing is set in stone*

---

In May 2020 I took part in the Pipeline Summer Camp - [the inagural offering from the Berlin data engineering bootcamp](https://www.dataengineering.academy/).

We ended up with - [see it deployed here](http://adamg33.pythonanywhere.com/).

<center>
	<img src="/assets/pipeline/app.png">
	<figcaption>The Summer Camp - I'm the poorly lit person in the upper middle</figcaption>
</center>

I've also been able to use lessons & learnings in other projects - notably the [climate-newspaper-downloader](https://github.com/ADGEfficiency/climate-newspaper-downloader).


## What was the Summer Camp?

The Summer Camp was a one week course in data engineering, held online by [Peter Fabian](https://www.linkedin.com/in/peter-fabian-000/) & [Daniel Molnar](https://www.linkedin.com/in/soobrosa/), the founders of [Pipeline Data Engineering Academy](https://www.dataengineering.academy/).  The course was delivered in May 2020, when much of the world's population was in lockdown.

<center>
	<img src="/assets/pipeline/class.png">
	<figcaption>The Summer Camp - I'm the poorly lit person in the upper middle</figcaption>
</center>

The effects of corona have been felt worldwide - yet there is opportunity in this crisis.  Pipeline have shown already they are capable of seeing opportunity in a crisis.

The course is a great example of a lean approach - a minimum viable product with lots of customer feedback.  It's great to be a part of, and I'm looking forward to seeing Pipeline expand their offering in data engineering education.


The course was delivered remotely via Zoom & Slack - roughly half teaching time, half project work.


## What can you learn in a week anyway?

It's not easy to teach online - the last few months at [Data Science Retreat](https://www.datascienceretreat.com/) were spent developing a coronavirus stragety for the school, including how to deliver classes online.  The course content (mainly lectures over slides) was delivered skillfully and technically relevant.  Students all felt comfortable enough to ask questions, which is a sign of a well delivered class.

The technical content was opinionated (without being religious), and full of useful perspectives on data engineering:
- access to data
- connect things together
- `get / load, store, deploy`

There was also a strong emphasis on being pragmatic, avoiding absolutes and realizing there isn't a one size fits all solution.  This 

There is no good place to be on the cloud
- all have downsides
- best stragety = be prepared

Enriching data to make it valuable to the business - including a brilliant example of converting UNIX time to data with semantic meaning:
- UNIX epoch time
- datetime (5th Jan 2020 etc)
- thursday (calculable)
- the fall of the Berlin wall (resolvable with enricher)

Three types of use of data
- analytical (what will happen)
- forensic (what happened)
- monitoring (what is happening)

9/10 problems occur with external data

Filter then join!

Be defensive
- Check! Similar to 

Old tools (Python, SQL, SQLite, UNIXy)
Err on the side of understandable

Separation of concerns / dependency in how you structure data on disk

The other half of the course was project work - deploying a data engineering product in a week.


## What can you build in a week anyway?

Here is what we had to show - its a  simple stack that delivered a full data product ([this commit]()):
- Python
- SQLite
- Flask
- Docker

As someone who has taught data science for a while, the most impressive thing was the simplicity of the stack.  It is very easy when teaching to complicate things for students, or to teach complex tools that confuse more than help.

Teams, rewarding to see it working

It can't be understated the power of leaving after five days with a working product.

A month or so after, some technical debt was repaid:
- Makefile
- Jinja & Bootstrap templating


## What did I learn from the project?

<center>
	<img src="/assets/pipeline/pipeline.gif">
	<figcaption>Deploying the app on PythonAnywhere</figcaption>
</center>

SQLite (available on Mac) = big suprise

Also part of the course was
- Datasette
- GitHub actions to deploy after code is pushed to master

Usefulness of Makefiles to build DAGs.


## Where to go next

If you are someone considering transitioning into a data professional, consider data engineering with [Pipeline](https://www.dataengineering.academy/).

Thanks for reading!
