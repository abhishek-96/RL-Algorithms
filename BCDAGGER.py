"""F20 10-703 HW2
# 10-703: Homework 2 Part 1-Behavior Cloning & DAGGER

You will implement this assignment in this python file

You are given helper functions to plot all the required graphs
"""

from collections import OrderedDict 
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from imitation import Imitation
	
	

def generate_imitation_results(mode, expert_file, keys=[100], num_seeds=1, num_iterations=100):
	# Number of training iterations. Use a small number
	# (e.g., 10) for debugging, and then try a larger number
	# (e.g., 100).

	# Dictionary mapping number of expert trajectories to a list of rewards.
	# Each is the result of running with a different random seed.
	reward_data = OrderedDict({key: [] for key in keys})
	accuracy_data = OrderedDict({key: [] for key in keys})
	loss_data = OrderedDict({key: [] for key in keys})

	for num_episodes in keys:

		for t in range(num_seeds):

			print('*' * 50)
			print('num_episodes: %s; seed: %d' % (num_episodes, t))

			# Create the environment.
			env = gym.make('CartPole-v0')
			env.seed(t) # set seed
			im = Imitation(env, num_episodes, expert_file)
			expert_reward = im.evaluate(im.expert)
			print('\n\nExpert reward: %.2f' % expert_reward)

			loss_vec = []
			acc_vec = []
			imitation_reward_vec = []
			for i in range(num_iterations):
				
				if mode == 'behavior cloning':
					im.generate_behavior_cloning_data()
				else:
					im.generate_dagger_data()

				loss, acc = im.train()	
				loss_vec.append(loss)
				acc_vec.append(acc)

				reward = im.evaluate(im.model)
				imitation_reward_vec.append(reward)

				print('Iter', i, 'Loss', loss, 'Accuracy', acc, 'Reward', reward)

			loss_data[num_episodes].append(loss_vec)
			accuracy_data[num_episodes].append(acc_vec)
			reward_data[num_episodes].append(imitation_reward_vec)
	
	return reward_data, accuracy_data, loss_data, expert_reward


"""### Experiment: Student vs Expert
In the next two cells, you will compare the performance of the expert policy
to the imitation policies obtained via behavior cloning and DAGGER.
"""
def plot_student_vs_expert(mode, expert_file, keys=[100], num_seeds=1, num_iterations=100):
	assert len(keys) == 1
	reward_data, acc_data, loss_data, expert_reward = \
		generate_imitation_results(mode, expert_file, keys, num_seeds, num_iterations)
	print(type(reward_data))
	print(reward_data)
	print(np.mean(np.array(reward_data[keys]), axis = 0))
	### Plot the results
	plt.figure(figsize=(12, 3))
	# WRITE CODE HERE
	plt.plot(np.mean(np.array(reward_data[keys]), axis = 0), label = "Reward")
	plt.plot(np.mean(np.arryay(accuracy_data[keys]), axis = 0), label = 'Accuracy')
	plt.plot(np.mean(np.array(loss_data[keys]), axis = 0), label = 'Loss')
	plt.xlabel("Iteration")
	plt.ylabel("Function Value")
	plt.legend()
	plt.show()

	# END
	plt.savefig('p1_student_vs_expert_%s.png' % mode, dpi=300)
	# plt.show()

"""Plot the reward, loss, and accuracy for each, remembering to label each line."""
def plot_compare_num_episodes(mode, expert_file, keys, num_seeds=1, num_iterations=100):
	s0 = time.time()
	reward_data, accuracy_data, loss_data, _ = \
		generate_imitation_results(mode, expert_file, keys, num_seeds, num_iterations)
	
	### Plot the results
	plt.figure(figsize=(12, 4))
	# WRITE CODE HERE
	plt.plot(reward_data[keys], label = "Reward")
	plt.plot(accuracy_data[keys], label = 'Accuracy')
	plt.plot(loss_data, label = 'Loss')
	plt.xlabel("Iteration")
	plt.ylabel("Function Value")
	plt.legend()
	plt.show()
	# END
	plt.savefig('p1_expert_data_%s.png' % mode, dpi=300)
	# plt.show()
	print('time cost', time.time() - s0)


def main():
	# generate all plots for Problem 1
	expert_file = 'expert.h5'
	
	# switch mode
	mode = 'behavior cloning'
	# mode = 'dagger'

	# change the list of num_episodes below for testing and different tasks
	keys = [1] # [1, 10, 50, 100]
	num_seeds = 2 # 3
	num_iterations = 1    # Number of training iterations. Use a small number
							# (e.g., 10) for debugging, and then try a larger number
							# (e.g., 100).

	# Q1.1.1, Q1.2.1
	plot_student_vs_expert(mode, expert_file, keys, num_seeds=num_seeds, num_iterations=num_iterations)

	# Q1.1.2, Q1.2.2
	# plot_compare_num_episodes(mode, expert_file, keys, num_seeds=num_seeds, num_iterations=num_iterations)
	

if __name__ == '__main__':
	main()
