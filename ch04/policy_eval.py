if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):
    '''
    反復方策評価
    1Step分
    '''

    #State: グリッドのマス目について繰り返し
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(
        pi,#方策
        V,#価値関数
        env,#環境
        gamma,#割引率
        threshold=0.00001
    ):
    while True:

        #更新前の価値観数
        old_V = V.copy()
        #反復方策評価(1Step)
        V = eval_onestep(pi, V, env, gamma)

        #収束判定
        #状態価値関数Vの値の変化量の最大値がStep前後で閾値以下となった場合に終了する

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        #変化量最大値が閾値以下→break
        if delta < threshold:
            break

    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)

    
