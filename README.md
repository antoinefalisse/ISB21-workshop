ISB2021 workshop on optimal control 
===================================

Welcome to this workshop on optimal control!

The hands-on part of this workshop will demonstrate how to generate a predictive simulation of walking with a planar five-link biped model. The workshop is based on the walking example from the following paper: "An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation" by Matthew Kelly. (SIAM Review, Vol. 59, No. 4, pp. 849-904).

The model we will use consists of 5 segments (tibias, femurs, and torso) and is driven by ankle, knee, and hip torques. The stance foot is fixed to the ground and the gait pattern is imposed to have no double stance and no flight phase. This deeply simplifies the problem as there is then no need for contact models or different equations for different phases of the gait cycle. 

The predictive simulation will be formulated as a trajectory optimization problem and solved using a direct collocation method. The goal of the problem will be to find the model states and controls that satisfy the model dynamics while minimizing the sum of the squared joint torques. You will be invited to test different cost functions, add some constraints, and adjust some variables so as to produce a variety of gait patterns. The aim is to demonstrate the potential of optimal control to address neuro-mechanical research questions.

We have set up a challenge for those who are interested: "Find the cost function that minimizes the maximum peak torque". We will provide you with more information about the challenge during the workshop.

The example is available in MATLAB and Python, so feel free to select whatever tool you prefer. You will find the install requirements as well as more info in the READMEs in the MATLAB and Python folders.

The problem should solve in only a couple of seconds on a standard laptop computer, which is great for testing the effect of different parameters and modeling choices on the predicted gait pattern.

If you have any questions, please feel free to contact us:
- Antoine Falisse: afalisse@stanford.edu
- Gil Serrancoli: gil.serrancoli@upc.edu
- Friedl De Groote: friedl.degroote@kuleuven.be

Credit to Tom Van Wouwe for the initial implementation of the code used for this workshop.
