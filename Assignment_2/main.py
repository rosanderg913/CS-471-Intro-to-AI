import math
import random


def f(x):
    return 2 - (x**2)

def g(x):
    return (0.0051 * x**5) - (0.1367 * x**4) + (1.24 * x**3) - (4.456 * x**2) + (5.66 * x) - 0.287

def hillClimb(function, domain, step, start_position):
    # Set the domain for the function locally to ensure that the function is not called out of bounds
    x_min = domain[0]
    x_max = domain[1]
    # Get the current value of the function
    current = function(start_position)
    # Define the left and right successors
    left_successor = function(start_position - step)
    right_successor = function(start_position + step)
    # Check if the left successor is greater than the current value and if it is within the domain
    if left_successor > current and left_successor >= x_min:
        return hillClimb(function, domain, step, start_position - step)
    # Check if the right successor is greater than the current value and if it is within the domain
    elif right_successor > current and right_successor <= x_max:
        return hillClimb(function, domain, step, start_position + step)
    # If the current value is greater than both the left and right successors, return the current position
    else:
        return start_position
    

def hillClimbRandomRestart(function, domain, step):
    # Declare a maximum value (starting at negative infinity)
    max_value = float("-inf")
    # Loop through the domain and generate a random starting position
    for i in range(domain[1] * 2):
        # Generate a random starting position
        start_position = random.randint(domain[0], domain[1])
        current = hillClimb(function, domain, step, start_position)
        # If the current value is greater than the maximum value, set the maximum value to the current value
        if current > max_value:
            max_value = current
    # Return the maximum value
    return max_value
    
print("Hill Climb with 0.5 step")
print(hillClimb(f, [-10, 10], 0.5, 2))
print("Hill climb with 0.01 step")
print(hillClimb(f, [-10, 10], 0.01, 1))
print("Random restart hill climb, on function g, with 0.5 step")
print(hillClimbRandomRestart(g, [-10, 10], 0.5))
print("Hill climb, on function g, with 0.5 step")
print(hillClimb(g, [-10, 10], 0.5, 4))
