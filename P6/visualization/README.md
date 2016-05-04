A data set containing 1,157 baseball players including their 
handedness (right or left handed), height (in inches), weight (in pounds), 
batting average, and home runs.
Create a visualization that shows differences among the performance of the 
baseball players.






Summary - in no more than 4 sentences, briefly introduce your data visualization and 
add any context that can help readers understand it

Baseball players are measured in part by their batting average and number of home runs.
This visualization illustrates the relationship between these two variables,
showing that the two exhibit a positive correlation.







Design - explain any design choices you made including changes to the visualization 
after collecting feedback

Many design choices were made as a result of the feedback.  The chart type,
axes, color scale, data point opacity and font were all changed given feedback.

The chart type was changed from a simple bar to a scatterplot.  The choice to do the 
initial bar chart was to succinctly summarize data by category of handedness
and show potential results of handedness and average number of home runs hit
or average batting average.  The result of the bar chart without confidence 
intervals suggested a difference in average home runs based on handedness, 
however, upon further inspection, this result was spurious.  A scatterplot was chosen
to fully suss out the effects of handedness on home runs and batting average.
The scatterplot allowed each data point to be shown as a relationship of home runs
and batting average tagged by handedness.  If clusters of handedness appeared in the
scatterplot, the bar chart would have potentially served as an accurate 
representation of the relationship of handedness on performance (either home runs,
batting average or both).  However, the scatterplot showed no trend of handedness 
but instead did show a strong relationship between batting average and home runs hit. 
This particular design choice ended up illuminating an effect I did not set out 
to investigate.

Styling was changed based on feedback that the data was hard to see and axes hard to
read.  While it was clear to me, I was too close to the chart engineering process
to come at it with fresh eyes.  Incorporating the feedback shows a much cleaner
and easier to interpret chart.  The most dramatic shifts were tweaking the data
point opacity so that clusters of data became more apparent and draws the eye, and
switching the axes such that the log scale of home runs ran along the y axis.

Finally adding clearer titles to the axes and manipulating their font
size and family made the whole chart more easily interpretable.  Again,
I was too close to the data and had forgotten about tweaking this before getting
feedback.

I toyed with the idea of removing the x-axis gridlines but decided to keep them in.
With those gridlines, it is easier to see the difference in home runs between batters
with an average in the range of .20 to .25 and batters with an average in the range
of .25 to .30.





Feedback - include all feedback you received from others on your visualization 
from the first sketch to the final visualization

Steen provided the first feedback that the simple bar chart of average number of 
home runs based on handedness did not show any confidence intervals.  Without
this information, we wouldn't really be able to tell if the difference in the bar
chart was a true effect or not.  Before showing the chart to anyone else, I made a
scatterplot of batting average and home runs split by handedness and did not see
any clear trend of handedness.  The bar chart was scrapped and I began tweaking
the scatter plot as it did show a positive relationship between batting average
and home runs.

Adrianne saw the most iterations of the chart and provided the bulk of the feedback.  As
a former graphic designer, this feedback was quite helpful.  Initially Adrianne
complained about the color scale.  "Too neon" was the comment.  Given this was
a highlight of the lessons in the course, I tweaked the colors without any positive
feedback from Adrianne.  I then came to realize it may be more helpful to tweak
the opacity as up until now each data point was brightly lit.  This yielded an
immediate positive reaction.  Adrianne then indicated that the x axis was confusing.
Up until this point the x axis was a log scale of the number of home runs hit.
Switching the axes not only looked cleaner but the effect was more apparent.

Adrianne made comments about changing the font of the title and axes both in size
and family.

Suzanne saw a near final version of the chart and mirrored what Adrianne had
commented regarding font size and font family.  She remarked on her being old enough
that the original weighting of the axis was nearly invisible to her.

Suzanne also mentioned some of the data points almost looked invisible and she
wondered if her computer was the culprit.  I decreased the transparency a little
and the problem was resolved.


 




Resources - list any sources you consulted to create your visualization


https://github.com/PMSI-AlignAlytics/dimple/wiki --- all of my formatting came from here
colorpicker.com --- tried out some colors before just going with standard 'blue'
http://dimplejs.org/ --- brainstormed chart types looking at the different examples here