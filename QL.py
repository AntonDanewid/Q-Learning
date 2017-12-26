'''
HEADER:
Introduction to Artificial Intelligence: Assignment 3, Reinforcement Learning(QLearning)
Teach the kid to wear clothes in a proper manner.


(1) Representations of states:
At any given state of the program the state is defined as a 4 character string, where each character denotes the position of the cloth denoted by that character position.
Clothes Order in string: 0: shirt, 1: sweater, 2: socks, 3: shoes.
Example:
RRRR: Denotes that all the clothes are in the room. (initial state)
UUFF: Denotes that all the clothes have been worn in a proper manner. The shirt and sweater are on the upper body and the socks and shoes are on the feet.(Final State)


(2) Transition Diagram:
The transition Diagram is stored in a Dictionary where each key denotes a node in the graph and
the value for eack key is a list of all nodes that have a connection from that node.
For example: For the graph (http://www.mrgeek.me/wp-content/uploads/2014/04/directed-graph.png) the dictionary would look like:

tDiag = {
	"A":["B"],
	"B":["C"],
	"C":["E"],
	"D":["B"],
	"E":["D","F"],
	"F":[]
}

'''

#Libraries allowed: Numpy, Matplotlib
#Installed using: pip install numpy matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
#import pdb
'''
All possible locations for the clothes: "R: Room", "U: Upper Body", "F: Feet"
Clothes to wear along with their type; U: Upper Body, F: Feet:
NOTE: It is "not" required to use this variable.
'''
clothes = {
	0:{"name":"shirt","type":"U","order":1},
	1:{"name":"sweater","type":"U","order":2},
	2:{"name":"socks","type":"F","order":1},
	3:{"name":"shoes","type":"F","order":2}
}

'''
Global variable to store all "possible" states.
Please enter all possible states from part (a) Transition Graph in this variable.
For state reference check HEADER(1)
'''
states = ["RRRR", "URRR", "RRLR", "UURR", "URLR", "RRLL", "UULR", "URLL", "UULL"]


'''
This function is used to build the Transition Diagram.(tDiag)
I/P: states variable, O/P: returns transition dictionary.
For reference check HEADER(2)
'''
def buildTransitionDiag(states):
	tDiag = {
		"RRRR": ["URRR", "RRLR"],
		"URRR": ["URLR", "UURR", "RRRR"],
		"UURR": ["UULR", "URRR"],
		"UULR": ["UULL", "UURR"],
		"RRLR": ["URLR", "RRLL", "RRRR"],
		"RRLL": ["URLL", "RRLR"],
		"URLL": ["UULL", "RRLL", "URLR"],
		"URLR": ["UULR", "URLL", "RRLR", "URRR"],
		"UULL": ["URLL", "UULR"]


	}

	#Edit this part with actual
	#code.
	return tDiag






'''
This function builds the Reward Matrix R.
Penultimate transition are assigned a high score ~ 100.
Possible transitions are assigned 0.
Transitions not possible are assigned -1.
I/P: transition diagram, O/P: returns R matrix.
'''
def buildRMatrix(tD):
	my_matrix = []
	matrix = np.empty((9,9))

	sindex =0
	for s in states:
		my_temp_matrix = []
		qindex = 0
		for q in states:

			r = 0
			if q in tD[s]:
				r = 0
			else:
				r = -1
			if q in tD[s]:
				if q == states[8]:
					r = 100
			matrix[sindex][qindex] = r
			qindex = qindex +1
			my_temp_matrix.append(r)  # eller hur fan du nu vill representera
		# dina entries i matrisen
		sindex = sindex+1
		my_matrix.append(my_temp_matrix)



	#print(my_matrix)


	# Enter your code here.
	print(matrix)
	return matrix

'''
This function returns the path taken while solving the graph by utilizing the Q-Matrix.
I/P: Q-Matrix. O/P: Steps taken to reach the goal state from the initial state.
NOTE: As you probably infer from the code, the break-off point is 50-traversals. You'll probably encounter this while finishing this assignment that at the initial stages of training, it is impossible for the agent to reach the goal stage using Q-Matrix. This break-off point allows your program to not be stuck in a REALLY-LONG loop.
'''
def solveUsingQ(Q):
	start = initial_state
	steps = [start]
	while start != goal_state:
		start = Q[start,].argmax()
		steps.append(start)
		if len(steps) > 50: break
	print("steps is")
	print(steps)
	return steps


'''
Q-Learning Function.
This function takes as input the R-Matrix, gamma, alpha and Number of Episodes to train Q for.
It returns the Q-Matrix as output.
'''
def learn_Q(R, gamma = 0.8, alpha = 0.0, numEpisodes = 0):
	#Write your code to do the work here.

	Q = np.zeros_like(R)
	#pdb.set_trace()
	#current = np.random.randint(9)
	current = random.sample(range(0,9), 1)
	list = [current]
	for i in range(0, numEpisodes):


		next_state = random.sample(range(0,9), 1)
		while R[current,next_state] == -1:
			next_state = random.sample(range(0,9), 1)
		#Q[current, next_state] = (1 - alpha) * Q[current, next_state] + alpha*(R[current, next_state] + gamma * np.max(Q[next_state]))
		Q[current, next_state] = Q[current, next_state] + alpha * (R[current, next_state] + np.max(Q[next_state]) - Q[current, next_state])

		current = next_state
		list.append(current)
			#Q[current, next_state] = Q[current, next_state] + alpha*(R[current, next_state] + gamma* np.max(Q[next_state, :]) - Q[current, next_state])
			#Q[current, next_state] = Q[current, next_state] * (1-alpha) + alpha*(R[current, next_state] + gamma*Q[next_state, ].argmax())
			#Q[current, next_state] = R[current, next_state] + gamma*Q[next_state,].argmax()


	#rn = Q[current][Q[current] > 0] / np.sum(Q[current][Q[current] > 0])
			#Q[current][Q[current] > 0] = rn
			#print(Q)

	print(list)
	return Q


#variables that hold returned values from the defined functions.
tDiag = buildTransitionDiag(states)
R = buildRMatrix(tDiag)

#Define the initial and goal state with the corresponding index they hold in variable "states".
initial_state = 0
goal_state = 8

'''
Problem: Perform 500 episodes of training, and after every 2nd iteration,
use the Q Matrix to solve the problem, and save the number of steps taken.
At the end of training, use the saved step-count to plot a graph: training episode vs # of Moves.

NOTE: Do this for 4 alpha values. alpha = 0.1, 0.5, 0.8, 1.0
'''
trainSteps = []#Variable to save iteration# and step-count.
runs = [i for i in range(10,200,2)]#List contatining the runs from 10 -> 200, with a jump of 2.
for i in runs:
	Q = learn_Q(R, alpha = 1.0, numEpisodes = i)


	stepsTaken = len(solveUsingQ(Q))
	if stepsTaken is 51:
		print(Q)
	trainSteps.append([i,stepsTaken])

#After Training, plotting diagram.
#NOTE: rename diagram accordingly or it will overwrite previous diagram.
x,y = zip(*trainSteps)
plt.plot(x,y,".-")
plt.xlabel("Training Episode")
plt.ylabel("# of Traversals.")
plt.savefig("output.png")

#Save the output for the best possible order, as generated by the code in the FOOTER.
path = solveUsingQ(Q)
print("\nThe best possible order to wear clothes is:\n")
tmp = ""
for i in path:
	tmp += states[i] + " -> "
print(tmp.rstrip(" ->"))
'''
FOOTER: Save program output here:

'''

''''
"RRRR": ["URRR", "RRLR"],
		"URRR": ["URLR", "UURR"],
		"UURR": ["UULR"],
		"UULR": ["UULL"],
		"RRLR": ["URLR", "RRLL"],
		"RRLL": ["URLL"],
		"URLL": ["UULL"],
		"URLR": ["UULR", "URLL"]
'''