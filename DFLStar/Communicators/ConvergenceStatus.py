# round_calculator.py
import numpy as np

class RoundCalculator:
    def __init__(self, window_size=4, epsilon=1e-10):
        self.window_size = window_size
        self.differences = []
        self.epsilon = epsilon

    def add_round(self, parameters):
        #current_round_values = list(parameters.values())

        if len(self.differences) >= self.window_size:
            self.differences.pop(0)

        if len(self.differences) > 0:
            previous_round_values = self.differences[-1]
            #round_difference = [current - previous for current, previous in zip(current_round_values, previous_round_values)]
            round_difference = parameters - previous_round_values
            self.differences.append(round_difference)
        else:
            self.differences.append(parameters)
            
        #print("differences:", len(self.differences))

    def calculate_cs(self):
      #Calculate the l1-norm of the sum of differences
        sum_diff = np.sum(self.differences, axis=0)
        l1_norm_sum_diff = np.sum(np.abs(sum_diff))

        # Calculate the sum of l1-norms of differences
        l1_norm_diff_sum = np.sum([np.sum(np.abs(val)) for val in self.differences])

        # Calculate Pt using the formula
        #epsilon = 1e-10  # Replace with your desired epsilon value
        pt = l1_norm_sum_diff / (self.epsilon + l1_norm_diff_sum)
        
        return pt
        
        # euclidean
        
#    def calculate_cs(self):
#        # Calculate the l2-norm of the sum of differences
#        sum_diff = np.sum(self.differences, axis=0)
#        l2_norm_sum_diff = np.linalg.norm(sum_diff, ord=2)
#
#        # Calculate the sum of l2-norms of differences
#        l2_norm_diff_sum = np.sum([np.linalg.norm(val, ord=2) for val in self.differences])
#
#        # Calculate Pt using the formula
#        pt = l2_norm_sum_diff / (self.epsilon + l2_norm_diff_sum)
#
#        return pt


    def get_differences(self):
        return self.differences
