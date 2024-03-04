'''
    Main class for MBPO/TD3. Contains the training routine for both MBPO and TD3,
    as well as model rollout, evaluation, and graphing functions.
    You will implement part of this file.
'''
# pylint: disable=W0201, C0103,
import os
import numpy as np
import torch
import pybullet_envs
import gym
import matplotlib.pyplot as plt

from src.utils import ReplayBuffer
from src.td3 import TD3
from src.pe_model import PE
from src.fake_env import FakeEnv

from tqdm import tqdm


class MBPO:
    '''
        The main class for both TD3 and MBPO. Some of the attributes are only
        used for MBPO and not for TD3. But notice that the vast majority
        of code is shared.
    '''
    def __init__(self, train_kwargs, model_kwargs, TD3_kwargs):
        # shared training parameters
        self.enable_MBPO = train_kwargs["enable_MBPO"]
        self.policy_name = train_kwargs["policy"]
        self.env_name = train_kwargs["env_name"]
        self.seed = train_kwargs["seed"] #random-seed
        self.load_model = train_kwargs["load_model"]
        self.max_timesteps = int(train_kwargs["max_timesteps"]) #maximum real-env timestemps
        self.start_timesteps = int(train_kwargs["start_timesteps"]) #burn-in period
        self.batch_size = int(train_kwargs["batch_size"])
        self.eval_freq = int(train_kwargs["eval_freq"]) #Model evaluation frequency
        self.save_model = train_kwargs["save_model"]
        self.expl_noise = train_kwargs["expl_noise"] #TD3 exploration noise

        # MBPO parameters. Pseudocode refers to MBPO pseudocode in writeup.
        self.model_rollout_batch_size = int(train_kwargs["model_rollout_batch_size"])
        self.num_rollouts_per_step = train_kwargs["num_rollouts_per_step"] #M in pseudocode
        self.rollout_horizon = train_kwargs["rollout_horizon"] #k in pseudocode
        self.model_update_freq = train_kwargs["model_update_freq"] #E in pseudocode
        self.num_gradient_updates = train_kwargs["num_gradient_updates"] #G in pseudocode
        self.percentage_real_transition = train_kwargs["percentage_real_transition"]

        # TD3 agent parameters
        self.discount = TD3_kwargs["discount"] #discount factor
        self.tau = TD3_kwargs["tau"] #target network update rate
        self.policy_noise = TD3_kwargs["policy_noise"] #sigma in Target Policy Smoothing
        self.noise_clip = TD3_kwargs["noise_clip"] #c in Target Policy Smoothing
        self.policy_freq = TD3_kwargs["policy_freq"] #d in TD3 pseudocode

        # Dynamics model parameters
        self.num_networks = model_kwargs["num_networks"] #number of networks in ensemble
        self.num_elites = model_kwargs["num_elites"] #number of elites used to predict
        self.model_lr = model_kwargs["model_lr"] #learning rate for dynamics model

        # Since dynamics model remains unchanged every epoch
        # We can perform the following optimization:
        # instead of sampling M rollouts every step for E steps, sample B * M rollouts per
        # epoch, where each epoch is just E environment steps.
        self.rollout_batch_size = self.model_rollout_batch_size * self.num_rollouts_per_step
        # Number of steps in FakeEnv
        self.fake_env_steps = 0

    def eval_policy(self, eval_episodes=10):
        '''
            Runs policy for eval_episodes and returns average reward.
            A fixed seed is used for the eval environment.
            Do not modify.
        '''
        env_name = self.env_name
        seed = self.seed
        policy = self.policy

        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        policy.actor.eval()
        with torch.no_grad():
            for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                    action = policy.select_action(torch.FloatTensor(state.reshape(1, -1)))
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def init_models_and_buffer(self):
        '''
            Initialize the PE dynamics model, the TD3 policy, and the two replay buffers.
            The PE dynamics model and the replay_buffer_Model will not be used if MBPO is disabled.
            Do not modify.
        '''
        self.file_name = f"{self.policy_name}_{self.env_name}_{self.seed}"
        print("---------------------------------------")
        print(f"Policy: {self.policy_name}, Env: {self.env_name}, Seed: {self.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if self.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        td3_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.discount,
            "tau": self.tau,
        }

        # Target policy smoothing is scaled wrt the action scale
        td3_kwargs["policy_noise"] = self.policy_noise * max_action
        td3_kwargs["noise_clip"] = self.noise_clip * max_action
        td3_kwargs["policy_freq"] = self.policy_freq

        model_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "num_networks": self.num_networks,
            "num_elites": self.num_elites,
            "learning_rate": self.model_lr,
        }
        
        self.policy = TD3(**td3_kwargs) #TD3 policy
        self.model = PE(**model_kwargs) #Dynamics model
        self.fake_env = FakeEnv(self.model) #FakeEnv to help model unrolling

        # if self.load_model != "":
        #     policy_file = self.file_name if self.load_model == "default" else self.load_model
        #     self.policy.load(f"./models/{policy_file}")

        self.replay_buffer_Env = ReplayBuffer(state_dim, action_dim)
        self.replay_buffer_Model = ReplayBuffer(state_dim, action_dim)


    def get_action_policy(self, state):
        '''
            Adds exploration noise to an action returned by the TD3 actor.
        '''
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float)
        action = (
            self.policy.select_action(state)
            + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        ).clip(-self.max_action, self.max_action)
        return action

    def get_action_policy_batch(self, state):
        '''
            Adds exploration noise to a batch of actions returned by the TD3 actor.
        '''
        assert len(state.shape) == 2 and state.shape[1] == self.state_dim
        action = (
            self.policy.select_action_batch(state)
            + np.random.normal(0, self.max_action * self.expl_noise,
                               size=(state.shape[0], self.action_dim))
        ).clip(-self.max_action, self.max_action)
        # Numpy array!
        return action

    def model_rollout(self):
        '''
            This function performs the model-rollout in batch mode for MBPO.
            This rollout is performed once per epoch, and we sample B * M rollouts.
            First, sample B * M transitions from the real environment replay buffer.
            We get B * M states from these transitions.
            Next, predict the action with exploration noise at these states using the TD3 actor.
            Then, use the step() function in FakeEnv to get the next state, reward and done signal.
            Add the new transitions from model to the model replay buffer.
            Continue until you rollout k steps for each of your B * M starting states, or you
            reached episode end for all starting states.
        '''
        rollout_batch_size = self.rollout_batch_size
        print('[ Model Rollout ] Starting  Rollout length: {} | Batch size: {}'.format(
            self.rollout_horizon, rollout_batch_size
        ))
        unit_batch_size = self.model_rollout_batch_size

        batch_pass = self.num_rollouts_per_step

        # populate this variable with total number of model transitions collected
        total_steps = 0
        
        self.policy.actor.eval()
        
        for j in tqdm(range(batch_pass)):
            if j == batch_pass - 1 and rollout_batch_size % unit_batch_size != 0:
                unit_batch_size = rollout_batch_size % unit_batch_size
            
            # Sample from the environment replay buffer
            state, _, _, _, _ = self.replay_buffer_Env.sample(unit_batch_size)
            
            # For k steps (or until all states are done, but doesnt matter cause self.rollout_horizon==1)
            # for k in range(self.rollout_horizon):
            with torch.no_grad():
                action = self.get_action_policy_batch(state)
        
            # hint: make use of self.fake_env. Checkout documentation for FakeEnv.py
            next_state, reward, done = self.fake_env.step(state, action)

            # add to replay buffer
            self.replay_buffer_Model.add_batch(state, action, next_state, reward, done)
        
            # increment step count
            total_steps += unit_batch_size
            
        print(f'[ Model Rollout ] Added: {total_steps} | Model pool: {self.replay_buffer_Model.size} (max {self.replay_buffer_Model.max_size})')

        self.fake_env_steps += total_steps

    def prepare_mixed_batch(self):
        '''
            TODO: implement the mixed batch for MBPO
            Prepare a mixed batch of state, action, next_state, reward and not_done for TD3.
            This function should output 5 tf tensors:
            state, shape (self.batch_size, state_dim)
            action, shape (self.batch_size, action_dim)
            next_state, shape (self.batch_size, state_dim)
            reward, shape (self.batch_size, 1)
            not_done, shape (self.batch_size, 1)
            If MBPO is enabled, each of the 5 tensors should a mixture of samples from the
            real environment replay buffer and model replay buffer. Percentage of samples
            from real environment should match self.percentage_real_transition
            If MBPO is disabled, then simply sample a batch from real environment replay buffer.
        '''

        if self.enable_MBPO:
            num_real_transitions = int(self.batch_size * self.percentage_real_transition)
            num_model_transitions = self.batch_size - num_real_transitions
            real_transitions = self.replay_buffer_Env.sample(num_real_transitions)
            model_transitions = self.replay_buffer_Model.sample(num_model_transitions)
            batch_transitions = [torch.cat([real_transitions[i], model_transitions[i]], axis=0) for i in range(len(real_transitions))]
        else:
            batch_transitions = self.replay_buffer_Env.sample(self.batch_size)

        return batch_transitions

    def plot_training_curves(self, evaluations, evaluate_episodes, evaluate_timesteps):
        '''
            Plotting script. You should include these plots in the writeup.
            Do not modify.
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(evaluate_episodes, evaluations)
        ax1.set_xlabel("Training Episodes")
        ax1.set_ylabel("Evaluation Reward")
        ax1.set_title("Reward vs Training Episodes")
        ax2.plot(evaluate_timesteps, evaluations)
        ax2.set_xlabel("Training Timesteps")
        ax2.set_ylabel("Evaluation Reward")
        ax2.set_title("Reward vs Training Timesteps")
        if self.enable_MBPO:
            algo_str = "MBPO"
        else:
            algo_str = "TD3"
        fig.suptitle("Training Curves for " + algo_str, fontsize=20)
        fig.savefig("./results/training_curve_{}.png".format(algo_str))

    def train(self):
        '''
            Main training loop for both TD3 and MBPO. See Figure 2 in writeup.
        '''
        self.init_models_and_buffer()
        env = gym.make(self.env_name)
        # Set seeds
        env.seed(self.seed)

        # Evaluate untrained policy
        evaluations = [self.eval_policy()]

        evaluate_timesteps = [0]
        evaluate_episodes = [0]

        state, done = env.reset(), False

        # You may want to set episode_reward appropriately
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        
        epoch_number = 1

        for t in range(self.max_timesteps):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < self.start_timesteps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.get_action_policy(state)
                
                if self.enable_MBPO and (t-self.start_timesteps) % self.model_update_freq == 0:
                    print(f"Epoch #{epoch_number}")
                    epoch_number += 1
                    self.model.train(self.replay_buffer_Env, self.batch_size)
                    self.model_rollout()

            # Perform action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # Store data in replay buffer
            self.replay_buffer_Env.add(state, action, next_state, reward, done)

            # Train agent after collecting burn-in memory
            if self.enable_MBPO:
                if t >= self.start_timesteps:
                    for g in range(self.num_gradient_updates):
                        batch_transition = self.prepare_mixed_batch()
                        self.policy.train_on_batch(*batch_transition)
            else:
                if t >= self.start_timesteps:
                    batch_transition = self.prepare_mixed_batch()
                    self.policy.train_on_batch(*batch_transition)

            state = next_state                
            
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                evaluations.append(self.eval_policy())
                evaluate_episodes.append(episode_num+1)
                evaluate_timesteps.append(t+1)
                if len(evaluations) > 5 and np.mean(evaluations[-5:]) > 990:
                    self.plot_training_curves(evaluations, evaluate_episodes, evaluate_timesteps)