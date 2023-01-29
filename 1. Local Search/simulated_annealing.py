from turtle import st
import numpy as np
import pandas as pd
from tqdm import tqdm


class SimulatedAnnealing:

    """
    NOTE:
        - S is the set of members.
        - T is the target value.
        - Chromosomes are represented as an array of 0 and 1 with the same length as the set.
        (0 means the member is not included in the subset, 1 means the member is included in the subset)

        Feel free to add any other function you need.
    """

    def __init__(self):
        pass

    def random_state_generator(self, n: int) -> np.ndarray:
        """
        Generate initial state: This function is used to generate the initial state.

        Inputs:
        - n: number of members in S

        Outputs:
        - initial state
        """
        return np.random.binomial(1, 0.5, n) 

    def is_feasible(self, state: np.ndarray, S: np.ndarray, T: int) -> bool:
        """
        This function is used to check if the sum of the subset is equal or less to the target value.

        Inputs:
        - state: state to be evaluated (subset)
        - S: set of members
        - T: target value

        Outputs:
        - True (1) if the sum of the state is equal or less to the target value, False (0) otherwise
        """

        return (np.sum(S[state == 1]) <= T)

    def neighbor_state_generator(self, state: np.ndarray, S: np.ndarray, T: int, prob: float = 0.5) -> np.ndarray:
        """
        Generate neighbor state: This function is used to generate the neighbor state.

        Inputs:
        - state: current state
        - prob: probability of flipping the second bit (described below)

        The neighbor state is generated in the following way:
        - Copy the current state
        - Get a random number between 0 and the length of the state
        - Flip the bit at the random position
        - Generate a random number between 0 and 1
        - If the random number is less than or equal to prob, 
            get another random number between 0 and the length of the state and flip the bit at the random position
        - If the neighbor state is feasible, return it. Else do the previous stages until the neighbor state is feasible

        Outputs:
        - neighbor state
        """
        copy = state.copy()
        while(True):
            index = np.random.randint(0, len(state))
            copy[index] = (1 if copy[index] == 0 else 0)
            if np.random.random() <= prob:
                index = np.random.randint(0, len(state))
                copy[index] = (1 if copy[index] == 0 else 0)
            if self.is_feasible(copy, S, T):
                return copy
        

    def cost_function(self, state: np.ndarray, S: np.ndarray, T: int) -> int:
        """
        Cost function: This function is used to calculate the cost of a certain state.

        Inputs:
        - state: state to be evaluated
        - S: set of members
        - T: target value

        It calculates the absolute difference between the sum of the members included in the subset
            (i.e. sum of S[i]s where state[i] == 1) and the target value.

        Outputs:
        - cost of the state
        """
        return abs(np.sum(S[state == 1]) - T)
        

    def prob_accept(self, state_cost: int, next_state_cost: int, temperature: float) -> float:
        """
        Probability of accepting the next state: This function is used to calculate the probability of accepting the next state.

        Inputs:
        - state_cost: cost of the current state
        - next_state_cost: cost of the next state
        - temperature: current temperature

        It calculates the probability of accepting the next state using the following formula:
            P = exp(- ((next_state_cost - state_cost) ** 1.5) / temperature)

        NOTE: Feel free to change the formula if you want to. (You can also consider other variables in determining the probability of accepting the next state.)

        Outputs:
        - probability of accepting the next state
        """
        return np.exp(- ((next_state_cost - state_cost) ** 1.5) / temperature)
        

    def run_algorithm(self, S: np.ndarray, T: int, neigbor_prob: float = 0.5, stopping_iter: int = 3000, temperature: float = 30):
        """
        Run algorithm: This function is used to run the simulated annealing algorithm.

        Inputs:
        - S: set of members
        - T: target value
        - neigbor_prob: probability of flipping the second bit (described in neighbor_state_generator)
        - stopping_iter: number of iterations to stop the algorithm
        - temperature: initial temperature

        It runs the simulated annealing in the following way:
        - Generate the initial state and set the best state to the initial state and the best cost to the cost of the initial state
        - Calculate the decay rate in the following way: decay_rate = temperature / stopping_iter
        - For each iteration:
        -   Generate the neighbor state
        -   Calculate the cost of the current state and the neighbor state
        -   If the cost of the neighbor state is less than or equal to the cost of the current state:
                set the current state to the neighbor state and the best state to the neighbor state and the best cost to the cost of the neighbor state
        -   Else:
        -       Generate a random number between 0 and 1
        -       If the random number is less than or equal to the probability of accepting the next state, set the current state to the neighbor state
        - Decrease the temperature by its decay rate
        - Append the current best cost and current best solution to the records list
        - Return the best cost, the best solution, and the records


        Outputs:
        - best cost
        - best solution
        - records
        """
        # UPDATE THESE VARIABLES (best_cost, best_solution, records)
        best_cost = np.Inf
        best_solution = None
        records = []

        # YOUR CODE HERE
        state = self.random_state_generator(len(S))
        best_solution = state
        best_cost = self.cost_function(state, S, T)
        decay_rate = temperature / stopping_iter

        for i in tqdm(range(stopping_iter)):
            neighbor_state = self.neighbor_state_generator(best_solution, S, T, neigbor_prob)
            state_cost = self.cost_function(state, S, T)
            neighbor_cost = self.cost_function(neighbor_state, S, T) 
            if (neighbor_cost <= state_cost):
                state = neighbor_state
                best_solution = neighbor_state
                best_cost = neighbor_cost
            else:
                if(np.random.random() < self.prob_accept(state_cost, neighbor_cost, temperature)):
                    state = neighbor_state
            temperature -= decay_rate
            records.append({'iteration': i, 'best_cost': best_cost,
                           'best_solution': best_solution})  # DO NOT REMOVE THIS LINE

        records = pd.DataFrame(records)  # DO NOT REMOVE THIS LINE
        return best_cost, best_solution, records

