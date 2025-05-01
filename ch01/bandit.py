import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


import seaborn as sns
sns.set_theme(context="notebook", style="darkgrid")

sns.set(font='IPAexGothic')

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        
        #action_size: エージェントが選択できる行動の数(==スロットマシンの台数)
        self.action_size = action_size#
        #各スロットマシンの価値の推定値を格納
        self.Qs = np.zeros(self.action_size)

        #各スロットマシンのプレイ回数を格納
        self.ns = np.zeros(self.action_size)

    def update(self, action, reward):
        #action: 選択したマシンのNo
        #当該マシンのプレイ回数をインクリメント
        self.ns[action] += 1

        #当該マシンの価値の推定値をアップデート
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):

        # 乱数値がepsilonを下回る場合→(確率epsilonで)→ランダムにマシンを選択
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        # それ以外の場合: 価値が最も高いマシンを選択
        return np.argmax(self.Qs)


if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        #勝率
        rates.append(total_reward / (step + 1))


    plt.subplot(1,2,1)
    print(total_reward)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)

    plt.subplot(1,2,2)

    plt.ylabel('勝率')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()
