from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.pdqn.policies import PDQNPolicy


def random_mentor(obs, n_actions):
    return np.random.randint(n_actions, size=1)


class PDQN(OffPolicyAlgorithm):
    """
    Deep Pessimistic Q-Network (PDQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[PDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # JB
        pessimism=10.,
        mentor=random_mentor,
    ):

        super(PDQN, self).__init__(
            policy,
            env,
            PDQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None


        # JB
        self.mentor = mentor
        self.pessimism = pessimism
        self.m_net, self.m_net_target = None, None
        self.queries = []
        print("ENV", self.env)
        self.reward_range = self.env.get_attr("reward_range")[0]
        print("RANGE", self.reward_range)
        # since gym environments seem to not take env.reward_range seriously...
        # (Cartpole is (-inf, inf))
        if not self.reward_range[1] - self.reward_range[0] < np.inf:
            self.reward_range = (-1, 1)  # hardcoded reward
        assert self.reward_range[1] - self.reward_range[0] < np.inf, (
            "env.reward_range is infinite. Manually set env.reward_range = (r_min, r_max)")

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PDQN, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

        self.m_net = self.policy.m_net
        self.m_net_target = self.policy.m_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        self._update_m_learning_rate(self.policy.m_optimizer)

        losses = []
        mentor_losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            mentor_replay_data = self.mentor_replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_m_values = self.m_net_target(mentor_replay_data.next_observations)

                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                next_m_values, _ = next_m_values.max(dim=1)  # only 1 dim

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                next_m_values = next_m_values.reshape(-1, 1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                target_m_values = mentor_replay_data.rewards + (1 - mentor_replay_data.dones) * self.gamma * next_m_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_m_values = self.m_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Correct sampling?
            # TODO - this will become higher dim
            current_m_values = th.gather(current_m_values, dim=1, index=mentor_replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            mentor_loss = F.smooth_l1_loss(current_m_values, target_m_values)

            # Pessimism - only applies to Q network
            if self.pessimism == 0.:
                pass
            elif self.pessimism > 0.:
                min_rew = self.reward_range[0]
                max_rew = self.reward_range[1]
                # Pessimistic Correction TODO - had to set manually before
                assert min_rew == 0 or self.gamma < 1
                if min_rew == 0:
                    min_value = 0
                else:
                    min_value = min_rew / (1 - self.gamma)

                # Geometric sum Q: should we do gamma ^ remaining on top?
                pleasant_surprise = (current_q_values - min_value) / (max_rew - min_rew)

                # print("Training")
                # print("Q", current_q_values, min_value)
                # print(max_rew, min_rew)
                # print("Surprise", pleasant_surprise)

                # TODO - verify time divide value. And will this fail on GPU?
                pess_errors = self.pessimism * th.mean(
                    th.square(pleasant_surprise), dim=1) / np.sqrt(self.num_timesteps)
                loss = th.mean(pess_errors + loss)  # No weights
            else:
                raise NotImplementedError("Optimism not implemented")

            losses.append(loss.item())
            mentor_losses.append(mentor_loss.item())


            # Optimize the policy
            self.policy.optimizer.zero_grad()
            self.policy.m_optimizer.zero_grad()

            loss.backward()
            mentor_loss.backward()

            # Clip gradient norm
            # TODO - find parameters - is it returning for m or Q?
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            # TODO get policy m_parameters ? 
            # print("PARAMETERS", self.policy.parameters())
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizer.step()
            self.policy.m_optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sampling not allowed?"""
        # raise NotImplementedError("Not implemented - need to follow pess_predict")

        # if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
        #     # Warmup phase
        #     unscaled_action = np.array([self.action_space.sample()])
        # else:
        #     # Note: when using continuous actions,
        #     # we assume that the policy uses tanh to scale the action
        #     # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        #     unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        unscaled_action, _, mentor_acted = self.pess_predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        # if isinstance(self.action_space, gym.spaces.Box):
        #     scaled_action = self.policy.scale_action(unscaled_action)

        #     # Add noise to the action (improve exploration)
        #     if action_noise is not None:
        #         scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

        #     # We store the scaled action in the buffer
        #     buffer_action = scaled_action
        #     action = self.policy.unscale_action(scaled_action)
        # else:
        #     # Discrete case, no need to normalize or clip
        #     buffer_action = unscaled_action
        #     action = buffer_action
        # return action, buffer_action
        return unscaled_action, unscaled_action, mentor_acted


    def pess_predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if False:  # not deterministic and np.random.rand() < self.exploration_rate:  NO SAMPLING
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            # WAS
            # action, state = self.policy.predict(observation, state, mask, deterministic)
            ######### from common.policies.BasePolicy #################
            if isinstance(observation, dict):
                observation = ObsDictWrapper.convert_dict(observation)
            else:
                observation = np.array(observation)

            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

            vectorized_env = is_vectorized_observation(observation, self.observation_space)

            observation = observation.reshape((-1,) + self.observation_space.shape)

            observation = th.as_tensor(observation).to(self.device)

            #with th.no_grad():
            #    actions = self.policy._predict(observation, deterministic=deterministic)
            # Convert to numpy
            #actions = actions.cpu().numpy()
            ############################################################


            # Final dim is the world model pred?
            with th.no_grad():
                mentor_value = self.policy._m_predict(observation)  # output = actions + 1
                pess_preds = self.policy._pess_values(observation)
                pess_value, _ = pess_preds.max(dim=-1)
    
            # print(pess_value, "\t", mentor_value)

            mentor_act = (
                mentor_value > pess_value + 0.01
                or self.num_timesteps < 1000  # total_timesteps / 10
            )

            if self.pessimism < 0:  # Optimistic
                mentor_act = False
            if mentor_act:
                action = self.mentor(state, self.env.action_space.n)
                self.queries.append(self.num_timesteps)
            else:
                action = pess_preds.argmax(dim=1).reshape(-1).cpu().numpy()

        return action, state, mentor_act

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PDQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(PDQN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(PDQN, self)._excluded_save_params() + [
            "q_net", "q_net_target", "m_net", "m_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer", "policy.m_optimizer"]

        return state_dicts, []
