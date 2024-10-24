"""
This module implements three UCB policies for deciding which feature to sample an edge from next.
Once a feature is decided, the edge with smallest/largest value of the feature is retrieved
from the sample pool. The reward of an arm pull is 0 if the sampled edge is correct, and
otherwise 1, and this is already reflected by the 'solution_incorrect' column.

We save the first rank of each chosen edge to a 'bandit_rank' column, and additionally keep a
'bandit_resample_rank' column to indicate when the edge was sampled again by the bandit, for
edges that are sampled twice. We also save the name of the feature that led to the edge
being first sampled, in the 'bandit_arm' column, and the name of the second arm in the 
'bandit_resample_arm' column.

We continue sampling until each edge in the solution has been sampled at least once. This means the 
total number of rounds we might play is equal to twice the number of edges in 
the solution, and the number of rounds played at any given time, t, is equal to the number of edges
with a rank in the 'bandit_rank' column plus the number of edges with a rank in the 'bandit_resample_rank'
column.
"""

import os
import numpy as np
import pandas as pd


def get_count_arm_played(played_ranks, arm_name, t, gamma=1):
    """Count the number of times an arm has been played up to round t.

    When gamma=1 this is just a plain count. When gamma<1, this is the
    discounted count using gamma^(t-s) where s is the round the arm was
    played.

    Parameters
    ----------
    played_ranks : Dict[str, List[int]]
        rounds each arm was played at
    arm_name : str
        name of the arm we are counting plays for. Must be a valid value 
        in the 'bandit_arm' column of df.
    t : int
        number of total rounds played until now
    gamma : int, optional
        discount between 0 and 1, by default 1
    """
    arm_played = played_ranks[arm_name]
    played_count = 0
    for rank in arm_played:
        played_count += gamma**(t - rank)
    return played_count

def get_reward_for_arm(rewards, played_ranks, arm_name, t, gamma=1):
    """Get the reward for an arm up to round t.

    When gamma=1 this is just a plain sum. When gamma<1, this is the
    discounted sum using gamma^(t-s) where s is the round the arm was
    played and found in the 'bandit_rank' column and possibly the
    'bandit_resample_rank' column of df.

    Parameters
    ----------
    rewards : Dict[str, List[int]]
        rewards for each arm pull
    played_ranks : Dict[str, List[int]]
        rounds each arm was played at
    arm_name : str
        name of the arm we are counting plays for. Must be a valid value 
        in the 'bandit_arm' column of df.
    t : int
        number of total rounds played until now
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    reward = 0
    arm_rewards = rewards[arm_name]
    ranks = played_ranks[arm_name]
    for i in range(len(arm_rewards)):
        current_reward = arm_rewards[i]
        current_rank = ranks[i]
        reward += current_reward * gamma**(t - current_rank)
    return reward

def get_confidence_bound_for_arm(Nt, eta_t, B=1, epsilon=2):
    """Get the confidence bound for an arm up to round t.

    When gamma=1, B=1 and epsilon=2 this is standard UCB1. When gamma<1,
    this is discounted UCB and B should be maximum reward * 2.

    Parameters
    ----------
    Nt : int
        number of times arm has been played
    eta_t: int
        disounted count of all arms played so far
    B : int, optional
        maximum reward of arm, by default 1
    epsilon : int, optional
        some appropriate constant, by default 2
    """
    if eta_t < 1:
        return np.inf
    to_sqrt = (epsilon * np.log(eta_t)) / Nt
    ct = B * np.sqrt(to_sqrt)
    return ct

def get_ucb_for_arm(rewards, played_ranks, arm_name, Nt, eta_t, t, B=1, epsilon=2, gamma=1):
    """Get the UCB for an arm at round t.

    When gamma=1, B=1 and epsilon=2 this is standard UCB1. When gamma<1,
    this is discounted UCB and B should be maximum reward * 2.

    Parameters
    ----------
    rewards : Dict[str, List[int]]
        rewards for each arm pull
    arm_name : str
        name of the arm we are counting plays for. Must be a valid value
        in the bandit_arm column of df.
    Nt : int
        number of times arm has been played
    eta_t: int
        discounted total number of times arms have been played
    t : int
        number of total rounds played so far
    B : int, optional
        maximum reward of arm, by default 1
    epsilon : int, optional
        some appropriate constant, by default 2
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    if Nt < 1:
        return np.inf
    reward_so_far = get_reward_for_arm(rewards, played_ranks, arm_name, t, gamma)
    average_reward = reward_so_far / Nt

    confidence_bound = get_confidence_bound_for_arm(Nt, eta_t, B, epsilon)

    ucb = average_reward + confidence_bound
    return ucb

def get_arm_to_play(df, played_ranks, rewards, t, B=1, epsilon=2, gamma=1):
    """Get the arm to play at round t.

    Parameters
    ----------
    played_ranks : Dict[str, List[int]]
        rounds each arm was played at
    rewards : Dict[str, List[int]]
        rewards for each arm pull
    t : int
        number of total rounds played so far
    B : int, optional
        maximum reward of arm, by default 1
    epsilon : int, optional
        some appropriate constant, by default 2
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    ucb_values = {}
    times_played = {}
    for arm in set(df.bandit_arm.unique()) - {-1}:
        times_played[arm] = get_count_arm_played(played_ranks, arm, t, gamma)
    eta_t = sum(times_played.values())
    for arm in set(df.bandit_arm.unique()) - {-1}:
        ucb_values[arm] = get_ucb_for_arm(rewards, played_ranks, arm, times_played[arm], eta_t, t, B, epsilon, gamma)
        if np.isnan(ucb_values[arm]):
            print('nan')
    arm_to_play = max(ucb_values, key=ucb_values.get)
    return arm_to_play



def rank_edges_by_ucb(df, B=1, epsilon=2, gamma=1):
    """Rank all edges for sampling using UCB arm draws.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe of edges we are sampling from
    B : int, optional
        maximum reward of arm, by default 1
    epsilon : int, optional
        some appropriate constant, by default 2
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    distance_sorted = df['feature_distance'].sort_values(ascending=False)
    sensitivity_sorted = df['sensitivity_diff'].sort_values(ascending=True)

    df['bandit_rank'] = -1
    df['bandit_resample_rank'] = -1
    df['bandit_arm'] = -1
    df['bandit_resample_arm'] = -1

    played_ranks = {
        'feature_distance': [],
        'sensitivity_diff': []
    }
    rewards = {
        'feature_distance': [],
        'sensitivity_diff': []
    }

    next_index_distance = 0
    next_index_sensitivity = 0
    t = 1
    # play first two arms in "arbitrary" order
    first_edge = distance_sorted.index[next_index_distance]
    df.loc[first_edge, 'bandit_rank'] = t
    df.loc[first_edge, 'bandit_arm'] = 'feature_distance'
    played_ranks['feature_distance'].append(t)
    rewards['feature_distance'].append(int(df.loc[first_edge, 'solution_incorrect']))
    t += 1
    next_index_distance += 1
    second_edge = sensitivity_sorted.index[next_index_sensitivity]
    if df.loc[second_edge, 'bandit_rank'] != -1:
        df.loc[second_edge, 'bandit_resample_rank'] = t
        df.loc[second_edge, 'bandit_resample_arm'] = 'sensitivity_diff'
        played_ranks['sensitivity_diff'].append(t)
        rewards['sensitivity_diff'].append(int(df.loc[second_edge, 'solution_incorrect']))
        t += 1
        next_index_sensitivity += 1
        second_edge = sensitivity_sorted.index[next_index_sensitivity]
    df.loc[second_edge, 'bandit_rank'] = t
    df.loc[second_edge, 'bandit_arm'] = 'sensitivity_diff'
    played_ranks['sensitivity_diff'].append(t)
    rewards['sensitivity_diff'].append(int(df.loc[second_edge, 'solution_incorrect']))
    t += 1
    next_index_sensitivity += 1

    unsampled = (df.bandit_rank == -1).sum()
    while unsampled > 0:
        arm = get_arm_to_play(df, played_ranks, rewards, t, B, epsilon, gamma)
        if arm == 'feature_distance':
            edge = distance_sorted.index[next_index_distance]
            next_index_distance += 1
        else:
            edge = sensitivity_sorted.index[next_index_sensitivity]
            next_index_sensitivity += 1
        if df.loc[edge, 'bandit_rank'] == -1:
            df.loc[edge, 'bandit_rank'] = t
            df.loc[edge, 'bandit_arm'] = arm
        else:
            df.loc[edge, 'bandit_resample_rank'] = t
            df.loc[edge, 'bandit_resample_arm'] = arm
        played_ranks[arm].append(t)
        rewards[arm].append(int(df.loc[edge, 'solution_incorrect']))
        t += 1
        unsampled = (df.bandit_rank == -1).sum()

if __name__ == '__main__':
    all_df_pth = '/home/ddon0001/PhD/experiments/scaled/no_merges_all/solution_edges_datasets_with_FP_WS_FA_FE.csv'
    out_root = '/home/ddon0001/PhD/experiments/ucb_ranking_fixed_gamma/'
    all_df = pd.read_csv(all_df_pth)
    ds_names = all_df.ds_name.unique()
    for ds_name in ds_names:
        ds_df = all_df[all_df['ds_name'] == ds_name].copy()
        ds_out = os.path.join(out_root, ds_name)
        os.makedirs(ds_out, exist_ok=True)

        b = 2
        gamma = 1 - (1 / (4 * np.sqrt(2 * ds_df.shape[0])))
        epsilon = 1/2
        print('Processing', ds_name, 'with gamma', gamma)
        out_pth = os.path.join(ds_out, f'ucb_ranked_{gamma:.4f}.csv')
        if os.path.exists(out_pth):
            print('Already processed', ds_name)
            continue
        rank_edges_by_ucb(ds_df, B=b, epsilon=epsilon, gamma=gamma)
        ds_df.to_csv(out_pth, index=False)
        print(f'Wrote {out_pth}')
        print('#' * 40)