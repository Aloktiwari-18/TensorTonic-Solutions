def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    n_states = len(values)
    new_values = [0.0] * n_states

    for s in range(n_states):
        best = float('-inf')

        for a in range(len(transitions[s])):
            q = rewards[s][a]

            for s_prime in range(n_states):
                prob = transitions[s][a][s_prime]
                q += gamma * prob * values[s_prime]

            best = max(best, q)

        new_values[s] = best

    return new_values
