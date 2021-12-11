## Q-learning

To run a normal experiment and/or calculate the Q* matrix make sure line 2 of q_learning_main.py is set to 
~~~
from q_learning_skeleton import *
~~~

To instead run the code for question 8, with optimistic initialization, change line 2 to
~~~
from q_learning_skeleton_optimistic_initialization import *
~~~
## Deep Q-learning

The code has been set up in a way that the question number can be chosen in line 70 of deep_q_learning_main.py to run the code corresponding to that part. 
~~~
ques_no = 9 # Set to 9, 10 or 11
~~~

The default is set to 11, which is the latest version with target network implementation. 

The value can be made (only to) 9 or 10 as well.
