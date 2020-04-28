"""
rl_reward.py
 - reward estimation in reinforcement learning
"""
import numpy as np



def reward_estimation_for_stop(chunk_start_probs, chunk_end_probs,
                               chunk_start_positions, chunk_end_positions, chunk_stop_flags,
                               chunk_yes_no_flag_probs=None, chunk_yes_no_ans_probs=None,
                               batch_yes_no_flags=None, batch_yes_no_answers=None):
    """
    @func: reward estimation for coqa/quac with yes-no questions
    rewards_for_stop: list (bsz, )
    Using probability as rewards
    """
    batch_size = len(chunk_start_probs)
    rewards_for_stop = [0.0] * batch_size
    for i in range(batch_size):
        stop_flag = chunk_stop_flags[i]
        if stop_flag != 1:
            continue
        # not allow_yes_no: QuAC and Trivia
        start_position = chunk_start_positions[i]
        end_position = chunk_end_positions[i]
        rewards_for_stop[i] =  chunk_start_probs[i][start_position] * chunk_end_probs[i][end_position]

        # allow_yes_no: CoQA
        if (chunk_yes_no_flag_probs is not None) and (chunk_yes_no_ans_probs is not None) \
           and (batch_yes_no_flags is not None) and (batch_yes_no_answers is not None):
            yes_no_flag = batch_yes_no_flags[i]
            # yes-no question
            if yes_no_flag == 1:
                yes_no_ans = batch_yes_no_answers[i]
                rewards_for_stop[i] = chunk_yes_no_flag_probs[i][yes_no_flag] \
                                      * chunk_yes_no_ans_probs[i][yes_no_ans]
            # wh- question
            else:
                rewards_for_stop[i] = chunk_yes_no_flag_probs[i][yes_no_flag] \
                                      * chunk_start_probs[i][start_position] \
                                      * chunk_end_probs[i][end_position]
    return rewards_for_stop                  


def reward_estimation(stop_rewards, stop_probs):
    """
    stop_rewards: list (max_read_times, bsz)
    stop_probs: list (max_read_times, bsz)
    """
    # stop_rewards: (bsz, max_read_times)
    stop_rewards = np.transpose(stop_rewards)
    # stop_rewards: (bsz, max_read_times)
    stop_probs = np.transpose(stop_probs)
    q_vals = []
    # calc from the end to the beginning time
    next_q_vals = None #np.zeros(len(stop_rewards))
    for t in reversed(range(1, stop_rewards.shape[1])):
        t_rewards = stop_rewards[:, t]
        t_probs = stop_probs[:, t]
        if next_q_vals is None:
            cur_q_vals = t_rewards
        else:
            cur_q_vals = np.multiply(t_rewards, t_probs) + np.multiply(next_q_vals, 1-t_probs)
        q_vals.append(list(cur_q_vals)[:])
        next_q_vals = cur_q_vals
    # q_vals: (bsz, max_read_times-1)
    q_vals = np.transpose(q_vals)
    return q_vals
