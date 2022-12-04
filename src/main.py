# Script to run the main program
# Suppose to handle every pages and the main gradio interface

# test progess bar nested in a progress bar
import progress.bar as pb
import time 

# bar 1
bar1 = pb.Bar('Processing', max=10)

# bar 2
bar2 = pb.Bar('Processing', max=10)

for i in range(10):
    time.sleep(0.1)
    bar1.next()
    for j in range(10):
        time.sleep(0.1)
        bar2.next()
    bar2.finish()

bar1.finish()
