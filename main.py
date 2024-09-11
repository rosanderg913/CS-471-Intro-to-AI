import math

def findIntersection(input):
    # Step 1: Add first circle to the known set
    known_set = [input[0]]

    final_length = len(input)

    input.pop(0)

    # Step 2: Iterate through known set and check if the circle intersects with any other circle, adding to known set if so
    for circle in known_set:
        for unknown_circle in input:
            if findIntersectionOfTwoCircles(unknown_circle, circle):
                known_set.append(unknown_circle)
                input.remove(unknown_circle)
    
    # If the length of the known set is equal to the length of the input, then all circles intersect
    if len(known_set) == final_length:
        return True
    else:
        return False
    


def findIntersectionOfTwoCircles(c1, c2):
    # Step1: Find the distance between the centers of the 2 circles (d = sqrt((x2 - x1)^2 + (y2 - y1)^2))
    distance = math.sqrt(((c2[0] - c1[0]) ** 2) + ((c2[1] - c1[1]) ** 2))

    # Case 1: Circle 1 and Circle 2 are the same
    if distance == 0 and c1[2] == c2[2]:
        return True
    # Case 2: Circle 1 and Circle 2 are concentric
    elif distance == 0 and c1[2] != c2[2]:
        return False
    # Case 3: Circle 1 is inside Circle 2
    elif distance < abs(c1[2] - c2[2]):
        return False
    # Case 4: Circle 1 and Circle 2 are separate
    elif distance > c1[2] + c2[2]:
        return False
    # Case 5: Circle 1 and Circle 2 touch at a single point
    elif distance <= c1[2] + c2[2]:
        return True
    
# Test cases
test_case_1 = [(1, 3, .7), (2, 3, .4), (3, 3, .9)]                  # True
test_case_2 = [(1.5, 1.5, 1.3), (4, 4, 0.7)]                    # False
test_case_3 = [(.5, .5, .5), (1.5, 1.5, 1.1), (.7, .7, .4), (4, 4, .7)] # False
test_case_4 = [(1,1,4), (1,4,0.9), (1,5.5,2)]                           # True
print('Test Case 1')
print(findIntersection(test_case_1))
print('Test Case 2')
print(findIntersection(test_case_2))
print('Test Case 3')
print(findIntersection(test_case_3))
print('Test Case 4')
print(findIntersection(test_case_4))