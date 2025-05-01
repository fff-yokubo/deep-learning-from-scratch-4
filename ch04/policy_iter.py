if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval


def argmax(d):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():

        action_values = {}

        #とりうる行動について繰り返し
        for action in env.actions():
            #状態state, 行動actionを前提に次にとりうる状態を抽出
            next_state = env.next_state(state, action)

            #状態, 行動, 次の状態を前提に報酬を取得
            #実際には、次の状態のみで報酬が決まる
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        #報酬が最もよくなるactionを取得, max_action
        opt_action = argmax(action_values)

        #報酬のもっともよいactionの確率を1, 他をゼロに設定
        action_probs = {
            act: 1 if act == opt_action else 0
            for act in env.actions() 
        }
        #方策を更新
        pi[state] = action_probs

    return pi

import pprint as pp


def policy_iter(env, gamma, threshold=0.001, is_render=True):

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    cnt = 0
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        pp.pprint(pi)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        pi = new_pi

        cnt+= 1

        print("\n#############\n")

    env.render_v(V, pi)

    return pi, cnt


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    pi, cnt = policy_iter(env, gamma, is_render = False)

    print(f"Counts: {cnt}")