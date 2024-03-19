# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import copy
import util
from game import Actions, Grid
from game import Directions
from util import foodGridtoDic,Stack,PriorityQueue

from itertools import product

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # "*** YOUR CODE HERE ***"
    stack = Stack()
    visited = set()
    start_state = problem.getStartState()
    
    stack.push((start_state, [], 0))  # (state, path, cost)
    while not stack.isEmpty():
        state, path, _ = stack.pop()

        if state not in visited:
            visited.add(state)

            if problem.isGoalState(state):
                return path

            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, path + [action], 0))
    return []



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Initialize the priority queue, visited list, and start state.
    frontier = PriorityQueue()
    visited = set()
    start_state = problem.getStartState()

    # Push the start state with an empty path and zero cost.
    frontier.push((start_state, [], 0), 0)
    while not frontier.isEmpty():
        # Pop the node with the lowest cost.
        state, path, cost = frontier.pop()

        # Check if the state is the goal state.
        if problem.isGoalState(state):
            return path

        # If the state has not been visited, mark it as visited.
        if state not in visited:
            visited.add(state)

            # Explore the successors of the current state.
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    # Calculate the new cost and push the successor to the frontier.
                    new_cost = cost + step_cost
                    new_path = path + [action]
                    frontier.push((successor, new_path, new_cost), new_cost)

    return []

def nullHeuristic(state, problem=None):
    """
    This heuristic is trivial.
    """
    return 0


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for task1 ***"
    #solution 1. find a food that longest distance for the pacman agent
    # pacmanPosition, foodGrid = state
    # food_list = foodGrid.asList()
    # if len(food_list) == 0:
    #     return 0
    # heuristic = 0
    # for item in food_list:
    #     foodDistance = getmazeDistance(pacmanPosition, item, problem)
    #     if foodDistance > heuristic:
    #         heuristic = foodDistance
    # return heuristic        
    
    #solution 2. find the nearest food distance. next consider the distances from 
    # the nearest food to other foods and add these distances into the calculation of the heuristic value.
    pacmanPosition, foodGrid = state
    food_list = foodGrid.asList()
    if len(food_list) == 0:
        return 0

    # init
    heuristic = 0

    # each distance for the food and pacman
    distances = [getmazeDistance(pacmanPosition, food, problem) for food in food_list]

    # Find the nearest food
    min_distance = min(distances)
    nearest_food = food_list[distances.index(min_distance)]

    # Add the remaining food distance after the closest food arrives to the heuristic calculation
    for food in food_list:
        if food != nearest_food:
            distance = getmazeDistance(nearest_food, food, problem)
            heuristic = max(heuristic, min_distance + distance)

    return heuristic

    #solution 3.
    

def getmazeDistance(point1, point2, problem):
    #start
    x1, y1 = point1
    #end
    x2, y2 = point2
    walls = problem.walls
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    
    # hist = problem.heuristicInfo[] 
    if (point1,point2) in problem.heuristicInfo.keys():
        return  problem.heuristicInfo[(point1,point2)] 
    else:
        
        food_grid = Grid(walls.width, walls.height, initialValue=False)
        food_grid[x2][y2]    = True
        single_food_problem = SingleFoodSearchProblem(point1,food_grid,walls)
        path = len(astar(single_food_problem))
        
        problem.heuristicInfo[(point1,point2)] = path
        return path

class MAPFProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPositions, foodGrid ) where
      pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
      foodGrid:         a Grid (see game.py) of either pacman_name or False, specifying the target food of that pacman_name. For example, foodGrid[x][y] == 'A' means pacman A's target food is at (x, y). Each pacman have exactly one target food at start
    """

    def __init__(self, startingGameState):
        "Initial function"
        "*** WARNING: DO NOT CHANGE!!! ***"
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()

    def getStartState(self):
        "Get start state"
        "*** WARNING: DO NOT CHANGE!!! ***"
        return self.start
    
    def getFoodGrid(self):
        # state = self.getStartState()
        _, foodGrid = self.start
        return foodGrid
    def getAgents(self):
        
        pacmanPosition, _ =  self.start
        return list(pacmanPosition.keys())
    
    def isGoalState(self, state):
        "Return if the state is the goal state"
        "*** YOUR CODE HERE for task2 ***"
        # find all foodGrid for each pacman that can be finded.
        # comment the below line after you implement the function
        
        pacmanPosition, foodGrid = state
        counts = [foodGrid.count(item=name) for name in pacmanPosition.keys()]
        # print(counts)
        # if sum(counts)>0:
        #     return True
        # else:
        #     return False
        return not sum(counts) > 0

        

    
    def getSuccessors(self, state,agent=None):
        """
            Returns successor states, the actions they require, and a cost of 1.
            Input: search_state
            Output: a list of tuples (next_search_state, action_dict, 1)

            A search_state in this problem is a tuple consists of two dictionaries ( pacmanPositions, foodGrid ) where
              pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
              foodGrid:    a Grid (see game.py) of either pacman_name or False, specifying the target food of each pacman.

            An action_dict is {pacman_name: direction} specifying each pacman's move direction, where direction could be one of 5 possible directions in Directions (i.e. Direction.SOUTH, Direction.STOP etc)


        """
        "*** YOUR CODE HERE for task2 ***"
        "Returns successor states, the actions they require, and a cost of 1."
        if agent != None:
            successors = []
            directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
            # self._expanded += 1  # DO NOT CHANGE
            for direction in directions:
                #pacmposition,foodgrid=state
                x, y = state
                dx, dy = Actions.directionToVector(direction)
                next_x, next_y = int(x + dx), int(y + dy)
                if not self.walls[next_x][next_y]:
                    next_pos = (next_x, next_y)
                    successors.append((next_pos, direction, 1))
            return successors
        else:
            successors = []
            pacmanPositions, foodGrid = state
            directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]

            # generation all moves
            list_moves = {
                pacman_name: [
                    (direction, (int(x + dx), int(y + dy)))
                    for direction in directions
                    for dx, dy in [Actions.directionToVector(direction)]
                    if not self.walls[int(x + dx)][int(y + dy)]
                ]
                for pacman_name, (x, y) in pacmanPositions.items()
            }

            # combine all moves
            all_changes = list(product(*list_moves.values()))

            for moves in all_changes:
                if len(set(position for _, position in moves)) == len(moves):
                    next_positions = pacmanPositions.copy()
                    next_food_grid = foodGrid.copy()
                    moves_dict = {}

                    for pacman_name, (direction, next_pos) in zip(pacmanPositions.keys(), moves):
                        next_positions[pacman_name] = next_pos
                        moves_dict[pacman_name] = direction
                        if next_food_grid[next_pos[0]][next_pos[1]] == pacman_name:
                            next_food_grid[next_pos[0]][next_pos[1]] = False

                    successors.append(((next_positions, next_food_grid), moves_dict, 1))

            return successors


   
def manhattan_dist(a, b):
    
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def conflictBasedSearch(problem: MAPFProblem):
    
#     root = {'cost': 0,
#                 'constraints': [],
#                 'paths': [],
#                 'collisions': []}
    
#     for agent in problem.getAgents():
#         goal = getGoalState(problem=problem,agent=agent)
#         print(problem.getFoodGrid(),agent,goal)
        # path,solution =  astartSearchforConflict(problem,agent,root.constraints)
        # find all path for a with constraints
    

def conflictBasedSearch(problem):
    """
    Conflict-based search algorithm.
    Input: MAPFProblem
    Output: A dictionary stores the path for each agent as a list {agent_name: [action1, action2, ...]}.
    """
    root = Node(constraints={}, path={}, solution={}, cost=0)
    for agent in problem.getAgents():
        path, solution = aStarSearchForConflict(problem, agent, root.constraints)
        root.path[agent] = path
        root.solution[agent] = solution
    root.cost = sum(len(path) for path in root.solution.values())

    queue = util.PriorityQueue()
    queue.push(root, root.cost)

    while not queue.isEmpty():
        node = queue.pop()
        conflict = findConflict(node.path)
        if not conflict:
            return node.solution

        for agent in conflict['agents']:
            new_constraints = node.constraints.copy()
            new_constraints[(agent, conflict['x'], conflict['y'], conflict['t'])] = True
            new_node = Node(constraints=new_constraints, path=node.path.copy(), solution=node.solution.copy(), cost=0)
            path, solution = aStarSearchForConflict(problem, agent, new_constraints)
            new_node.path[agent] = path
            new_node.solution[agent] = solution
            new_node.cost = sum(len(path) for path in new_node.solution.values())
            queue.push(new_node, new_node.cost)

    return None


class Node:
    def __init__(self, constraints, path, solution, cost):
        self.constraints = constraints
        self.path = path
        self.solution = solution
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def aStarSearchForConflict(problem, agent, constraints):
    queue = util.PriorityQueue()
    start_state = problem.getStartState()[0][agent]
    goal_state = getGoalState(problem, agent)
    start_node = (start_state, 0, [start_state], [], 0)
    queue.push(start_node, 0)
    visited = set()

    while not queue.isEmpty():
        current_state, current_cost, current_path, current_solution, current_time = queue.pop()
        if current_state == goal_state:
            return current_path, current_solution
        if (current_state, current_time) in visited:
            continue
        visited.add((current_state, current_time))

        for next_state, action, step_cost in problem.getSuccessors(current_state, agent):
            if (agent, next_state[0], next_state[1], current_time + 1) not in constraints:
                new_cost = current_cost + step_cost
                new_path = current_path + [next_state]
                new_solution = current_solution + [action]
                new_time = current_time + 1
                h_cost = manhattanHeuristic(current_state, goal_state)
                total_cost = new_cost + h_cost
                new_node = (next_state, new_cost, new_path, new_solution, new_time)
                queue.push(new_node, total_cost)

    return [], []


def findConflict(paths):
    max_length = max(len(path) for path in paths.values())
    for index in range(max_length):
        arrive_positions = {}
        position_index = {agent: paths[agent][index] for agent in paths if index < len(paths[agent])}
        for agent, position in position_index.items():
            if position in arrive_positions:
                return {'x': position[0], 'y': position[1], 't': index, 'agents': [agent, arrive_positions[position]]}
            arrive_positions[position] = agent
    return {}


def manhattanHeuristic(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def getGoalState(problem, agent):
    for x in range(problem.getFoodGrid().width):
        for y in range(problem.getFoodGrid().height):
            if problem.getFoodGrid()[x][y] == agent:
                return (x, y)
    return None

"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"


class FoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
            #pacmposition,foodgrid=state
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                nextFood = state[1].copy()
                nextFood[next_x][next_y] = False
                successors.append((((next_x, next_y), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class SingleFoodSearchProblem(FoodSearchProblem):
    """
    A special food search problem with only one food and can be generated by passing pacman position, food grid (only one True value in the grid) and wall grid
    """

    def __init__(self, pos, food, walls):
        self.start = (pos, food)
        self.walls = walls
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    Q = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    Q.push(startNode)
    while not Q.isEmpty():
        node = Q.pop()
        state, cost, path = node
        if problem.isGoalState(state):
            return path
        for succ in problem.getSuccessors(state):
            succState, succAction, succCost = succ
            new_cost = cost + succCost
            newNode = (succState, new_cost, path + [succAction])
            Q.push(newNode)

    return None  # Goal not found


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch
