from utils import sgn

# Simple reward: score differential plus a bonus for winning
simple_reward = lambda s: s[-1] + 10 * sgn(s[-1])

def no_lookback_reward(s: list[int], gamma: float = 0.5):
    s_discount = [gamma**i * elt for i, elt in enumerate(s)]
    bonus = 10 * sgn(s[-1])
    return [sum(s_discount[i:]) + bonus for i in range(len(s_discount))]
