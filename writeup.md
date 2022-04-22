# Homework 6: Data Visualization!

## Visualization Scenarios

------------------------------------------------------------------------------------------------------------

### Stage 1: Data Visualization in Data Exploration 

1. **Aspect One**:
    - **Question**: *What is the breakdown of outcomes for a stop?*
    - **Graph & Analysis**: *The Graph goes as follows:
    ![Composition of Outcomes for Police Stops](StopOutcomes.png)
    Our default target for Rhode Island traffic stops data is "stop outcomes." So I think to get a could feel for the data it is appropriate to visualize the categorical composition of the stop outcomes. This provided good intuition moving forward since we can be aware that the overwhelming majority of stops result in citations. It also helps visualize the data, though, because we can see that other categories are actually within pretty close comparison of one another, and its only citations which so greatly outnumber the rest.*

2. **Aspect Two**:
    - **Question**: *Your question of choice here*
    - **Graph & Analysis**: *![Composition of Outcomes for Police Stops](VSForForged.png)
    In the bank data we only really have two categories of data points, forgeries and non forgeries. These points each come, however with four moments of distribution. Since the moments are all we can compare amongst this data, I thought it useful to compare the variance and skewness, the first and second moment of distribution, for forgeries and non forgeries independentally. We can see that the variance for forgeries is much larger than that of non forgeries, yet the skewness of the non forgeries is much greater than that of the forgery points. The forged points skew very negative and the non forged points skew very positive... as would be expected.*

3. **Aspect Three**:
    - **Question**: *What is the racial composition of all traffic stops in Rhode Island?*
    - **Graph & Analysis**: *The Graph goes as follows:
    ![Racial Composition of Police Stops](CompOfPoliceStops.png)
    We can see from this pie chart that the vast majority of traffic stops are from white people. Followed by black, hispanic, and a small sliver for Asian. Thankfully this chart is fairly representative of the population of Rhode Island, however, it does seem that minority groups may be slightly overrepresented in traffic stops as compared to their composition of the population itself. Besides the point, I felt as if this was an important part of the data to highlight as to just get a feel of what we are to see in the data in terms of racial factors. Any time we are analyzing police data, race unfortunately is something we should look into very quickly.*

------------------------------------------------------------------------------------------------------------

### Stage 2: Data Visualization for Model Exploration

1. **Aspect One**:
    - **Question**: *Given Race and Gender, can we predict the outcome of the stop?*
    - **Graph & Analysis**: *![Decision Tree, outcome of stop](Tree_Graph.png)
    It is often found in police records that men are treated more harshly by police, and that people of color are treated more harshly by police... And men of color are ultimately treated the harshest. If we know someone's race and gender in 	Rhode Island, can we predict the outcome of a police officer stopping them? The answer is yes and with an accuracy of about .86. The reason for this however has much to do with the first graph I displayed in Stage 1. Since the overwhelming majority of stops result in a citation (i.e. around .86/1), if the model predicts citation, it is very likely to get it right.*

2. **Aspect Two**:
    - **Question**: *Is there a correlation between stop time and the age of person being stopped... And if so, does increasing the k value in K-Means result in better accuracy?*
    - **Graph & Analysis**: *![Increasing KMeans vs Accuracy Graph](KMeans.png)
    We find... that there is not a great correlation between the age of someone being stopped and the amount of time they are stopped for! One could have expected to different scenarios. Maybe younger people are stopped for longer durations of time because they're assumed to be potentially more dangerous. Or maybe older people are stopped for longer since, well... they move a little slower. It does not seem, however that there is much of a correlation between the two and the best our model can do for prediction is only .35. Even with 7 K's in a KMeans, our algorithm does not seem to find any useful logic for predicting stop time given age.*

3. **Aspect Three**:
    - **Question**: *Your question of choice here*
    - **Graph & Analysis**: *![Increasing KMeans vs Accuracy Graph](HeatMap.png)
    In this portion we analyze the relationship between a search being conducted and the age of the driver. My hypothesis would be that the younger drivers would have a higher chance of being searched... and this is the case! In the graph I display the desired logistic regression Heat Map. It is a little wonky however because there are only two values for "search conducted" and those are 1 and 0. Anyway the graph still provides some interesting results.*

------------------------------------------------------------------------------------------------------------

### Stage 3: Data Visualization for *SICK* Applications 

*Your geographic plot using Plotly goes here!*

------------------------------------------------------------------------------------------------------------

## Socially Responsible Computing


### Ethics Part I
1. **Question 1**:
- *One digital tool which helps increase accessibility is white point reduction. Many people, including myself, have eyes which are sensitive to light. Oftentimes even a reduction in screen brightness does not suffice for someone with sensitive eyesight. After prolonged periods of usage, someone with sensitive eyes could be impaired from using the technology. This is where white point reduction makes the technology much more accessible and is typically a component of Apple's smartphones. Reducing white point on a screen makes it much easier on the eyes allowing people with sensitive eyes to use the technology much more effectively.*

2. **Question 2**:
- *I think maybe the most important scenario is the one we touched upon in the readings - colorblindness. So many people in so many different walks of life suffer some form of colorblindness. Often times even our most prominent decision makers will suffer from this. It is vitally important then that we make data which can be visualized by people who suffer from color blindness since the condition bleeds into so many different areas.*

3. **Question 3**:
- *In my opinion this is a pretty remarkable visualization. Now the content of this data is likely found by tracking people, essentially, which is likely a severe invasion of privacy. Also, representing the demographic data as a number of figure does reduce the humanity of the individuals who comprise the data. With that said, sometimes in order to gain macro insights about large scale events, each individual within the demographic is so small in comparison to the entirety, that if conducted in the right manner, it may be ok to reduce them to a number. There are certain data I would not create a visual of, likely when pertaining to certain content. Content which is deemed hurtful should likely not be visualized unless in very specific scenarios.*


### Ethics Part II
1. **Question 1**:
- *One user group that should find my graphs accessible are data science students because they study this stuff and work on much more complicated graphs than I made. Another group which should find my graphs accessible are people with the ability to zoom on their computer science I couldn't get the decision tree to zoom in... so the user needs to do it themselves.*
-*One user group who may have trouble viewing my graphs are people who do not have a zoom feature on their device. As mentioned previously, I couldn't blow up the decision tree and so people need to zoom in. Another demographic who may have trouble are blind people because my graphs skew very heavily towards the visual rather than hearing side of the senses spectrum.*
- *Given more time and resources I would have loved to make my visualizations in such a way where people could focus their attention on specific portions of the graph much easier. This could help people with low attention disorder as they would not get distracted from the mundaneness. How I would go about doing this is by making certain aspect grow or zoom when you hover over them. another feature I would implement is the Alt Text feature. The ability for my graphs to be read out loud would be super cool and make them so much more accessible to my viewers*

2. **Question 2**:
- *I found picking the right chart to be the easiest part of the accessible visualization process. It is important to pick the right type of graph or else people's disabilities will only be amplified by a hard to read graph. Another factor I found easy was labelling the graph in the right way to convey my message. This is important because of people analyzing your graphs have low attention spans, it is important to give them the gist immediately.*
