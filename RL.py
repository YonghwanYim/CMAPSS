import numpy as np

class Agent:
    def __init__(self, policy, q_function, epsilon=0.8, epsilon_delta=0.0035, min_epsilon=0):
        self.policy = policy
        self.q_function = q_function
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        # epsilon-greedy 정책에 따라 행동을 선택합니다.
        if np.random.rand() < self.epsilon:
            return self.policy.get_random_action()
        else:
            return self.policy.get_greedy_action(state)

    def update_epsilon(self, episode):
        # epsilon을 감소시킵니다.
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_delta)

    def update_q_function(self, state, action, reward, next_state, next_action, gamma=0.999, alpha=0.001):
        # Q 함수를 업데이트합니다.
        td_target = reward + gamma * self.q_function.get_value(next_state, next_action)
        td_error = td_target - self.q_function.get_value(state, action)
        self.q_function.update_value(state, action, alpha * td_error)

class Environment:
    def __init__(self, states, actions, rewards, unit_numbers):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.unit_numbers = unit_numbers
        self.current_state_index = 0

    def get_current_state(self):
        return self.states[self.current_state_index]

    def get_next_state_index(self, action):
        # 특정 행동에 따라 다음 상태의 인덱스를 반환합니다.
        if action == 'continue':
            self.current_state_index += 1
        else:
            unit_number = self.unit_numbers[self.current_state_index]
            self.current_state_index = self.unit_numbers.index(unit_number + 1)
        return self.current_state_index

    def get_reward(self, action):
        # 특정 행동에 대한 보상을 반환합니다.
        return self.rewards[self.current_state_index, action]

class Policy:
    def __init__(self, actions):
        self.actions = actions

    def get_greedy_action(self, state):
        # 현재 상태에 대해 가장 높은 값을 갖는 행동을 반환합니다.
        pass

    def get_random_action(self):
        # 무작위로 행동을 선택합니다.
        pass

class QFunction:
    def __init__(self, num_states, num_actions):
        self.values = np.zeros((num_states, num_actions))

    def get_value(self, state, action):
        # 특정 상태와 행동에 대한 Q 값 반환
        return self.values[state, action]

    def update_value(self, state, action, value):
        # Q 값 업데이트
        self.values[state, action] = value

class Main:
    def __init__(self, agent, environment, max_episodes):
        self.agent = agent
        self.environment = environment
        self.max_episodes = max_episodes

    def train(self):
        for episode in range(self.max_episodes):
            total_reward = 0
            current_state = self.environment.get_current_state()

            for _ in range(len(self.environment.states)):
                action = self.agent.choose_action(current_state)
                next_state_index = self.environment.get_next_state_index(action)
                reward = self.environment.get_reward(action)
                next_state = self.environment.states[next_state_index]
                next_action = self.agent.policy.get_greedy_action(next_state)

                self.agent.update_q_function(current_state, action, reward, next_state, next_action)
                total_reward += reward
                current_state = next_state

            self.agent.update_epsilon(episode)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

