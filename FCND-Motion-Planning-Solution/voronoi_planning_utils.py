
# coding: utf-8

# In[ ]:


from queue import PriorityQueue
import numpy as np
from scipy.spatial import Voronoi
from bresenham import bresenham
import networkx as nx
from planning_utils import heuristic, create_grid, collinearity_prune
from udacidrone.frame_utils import global_to_local
import numpy.linalg as LA
from enum import Enum
from math import sqrt


# In[ ]:


def get_object_centers(data, north_offset, east_offset, drone_altitude, safety_distance):
    """
    Returns a list of the obstacle centers.
    """
    points = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            points.append([north - north_offset, east - east_offset])
    return points


# In[ ]:


def find_open_edges(graph, grid):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    
    edges=[]
    for v in graph.ridge_vertices:
        p1=graph.vertices[v[0]]
        p2=graph.vertices[v[1]]
        cells=list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        hit=False
        for c in cells:
            # First check if we're off the map
            if np.amin(c)<0 or c[0]>=grid.shape[0] or c[1]>=grid.shape[1]:
                hit=True
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                hit=True
                break
        # If the edge does not hit on obstacle
        # add it to the list
        if not hit:
            # array to tuple for future graph creation step
            p1=(p1[0], p1[1])
            p2=(p2[0], p2[1])
            edges.append((p1, p2))
    return edges


# In[ ]:


def create_graph_from_edges(edges):
    """
    Create a graph from the `edges`
    """
    G = nx.Graph()
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        dist = LA.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)
    return G


# In[ ]:


def create_graph(data, drone_altitude, safety_distance):
    """
    Returns a graph from the colloders `data`.
    """
    # Find grid and offsets.
    grid, north_offset, east_offset = create_grid(data, drone_altitude, safety_distance)

    # Find object centers.
    centers = get_object_centers(data, north_offset, east_offset, drone_altitude, safety_distance)

    # Create Voronoid from centers
    voronoi = Voronoi(centers)

    # Find open edges
    edges = find_open_edges(voronoi, grid)

    # Create graph.
    return (create_graph_from_edges(edges), north_offset, east_offset)

class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    SOUTH_EAST=(1, 1, sqrt(2))
    NORTH_EAST=(-1, 1, sqrt(2))
    SOUTH_WEST=(1, -1, sqrt(2))
    NORTH_WEST=(-1, -1, sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])
# In[ ]:
def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.SOUTH_EAST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NORTH_EAST)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] ==1:
        valid_actions.remove(Action.SOUTH_WEST)
    if x - 1 < 0 or y - 1 <0 or grid[x - 1, y - 1] ==1:
        valid_actions.remove(Action.NORTH_WEST)

    return valid_actions


def a_star(graph, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost
# In[ ]:



# In[ ]:


def closest_point(graph, point_3d):
    """
    Compute the closest point in the `graph`
    to the `point_3d`.
    """
    current_point = (point_3d[0], point_3d[1])
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = LA.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point


# In[ ]:


def calculate_waypoints(global_start, global_goal, global_home, data, drone_altitude, safety_distance):
    """
    Calculates the waypoints for the trajectory from `global_start` to `global_goal`.
    Using `global_home` as home and colliders `data`.
    """
    # Calculate graph and offsets
    graph, north_offset, east_offset = create_graph(data, drone_altitude, safety_distance)

    map_offset = np.array([north_offset, east_offset, .0])

    # Convert start position from global to local.
    local_position = global_to_local(global_start, global_home) - map_offset

    # Find closest point to the graph for start
    graph_start = closest_point(graph, local_position)

    # Convert goal postion from global to local
    local_goal = global_to_local(global_goal, global_home) - map_offset

    # Find closest point to the graph for goal
    graph_goal = closest_point(graph, local_goal)

    # Find path
    path, _ = a_star(graph, graph_start, graph_goal)
    path.append(local_goal)

    # Prune path
    path = collinearity_prune(path, epsilon=1e-3)

    # Calculate waypoints
    return [[int(p[0] + north_offset), int(p[1] + east_offset), drone_altitude, 0] for p in path]

