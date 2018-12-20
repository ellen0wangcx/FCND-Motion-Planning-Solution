
# coding: utf-8

# In[ ]:


from skimage.morphology import medial_axis
from skimage.util import invert
from planning_utils import a_star, create_grid
import numpy as np
from bresenham import bresenham
from enum import Enum
from queue import PriorityQueue
import numpy as np
import utm
from bresenham import bresenham


# In[ ]:


def global_to_local(global_position, global_home):
    """
    Convert a global position (lon, lat, up) to a local position (north, east, down) relative to the home position.

    Returns:
        numpy array of the local position [north, east, down]
    """
    (east_home, north_home, _, _) = utm.from_latlon(global_home[1], global_home[0])
    (east, north, _, _) = utm.from_latlon(global_position[1], global_position[0])

    local_position = np.array([north - north_home, east - east_home, -global_position[2]])
    return local_position


def local_to_global(local_position, global_home):
    """
    Convert a local position (north, east, down) relative to the home position to a global position (lon, lat, up)

    Returns:
        numpy array of the global position [longitude, latitude, altitude]
    """
    (east_home, north_home, zone_number, zone_letter) = utm.from_latlon(global_home[1], global_home[0])
    (lat, lon) = utm.to_latlon(east_home + local_position[1], north_home + local_position[0], zone_number, zone_letter)

    lla = np.array([lon, lat, -local_position[2]])
    return lla


# In[ ]:


def create_voxmap(data, voxel_size=5, max_altitude=20):
    """
    Returns a grid representation of a 3D configuration space
    based on given obstacle data.
    
    The `voxel_size` argument sets the resolution of the voxel map. 
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))
    # minimum and maximum east coordinates
    east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))
    alt_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min))) // voxel_size
    east_size = int(np.ceil((east_max - east_min))) // voxel_size
    alt_size = int(alt_max) // voxel_size
    
    voxmap = np.zeros((north_size, east_size, alt_size), dtype=np.bool)
    
    for i in range(data.shape[0]):
        #continue
        # TODO: fill in the voxels that are part of an obstacle with `True`
        #
        # i.e. grid[0:5, 20:26, 2:7] = True
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(north-d_north-north_min)//voxel_size,
            int(north+d_north-north_min)//voxel_size,
            int(east-d_east-east_min)//voxel_size,
            int(east+d_east-east_min)//voxel_size,
        ]
        height=int(alt+d_alt)//voxel_size
        voxmap[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3], 0:height]=True
        
    return voxmap    


# In[ ]:


class Action_3D(Enum):
    """
    Variation on the Action class to support 3D actions.
    To keep the number of actions limited for now, only Actions UP and DOWN are added.

    the delta property returns the x,y,z delta movements. the last element is the cost (i.e. euclidean distance)
    """
    WEST = (0, -1, 0, 1)
    EAST = (0, 1, 0, 1)
    NORTH = (-1, 0, 0, 1)
    SOUTH = (1, 0, 0, 1)
    NORTH_WEST = (-1, -1, 0, np.sqrt(2))
    NORTH_EAST = (-1, 1, 0, np.sqrt(2))
    SOUTH_WEST = (1, -1, 0, np.sqrt(2))
    SOUTH_EAST = (1, 1, 0, np.sqrt(2))
    UP = (0, 0, 1, 1)
    DOWN = (0, 0, -1, 1)
    
    
    @property   
    def cost(self):
        return self.value[3]
    
    @property
    def delta(self):
        return (self.value[0], self.value[1], self.value[2])

def valid_actions_3D(grid, current_node):
    """
    Returns a list of valid 3D actions given a voxel grid and current node.
    """
    valid = list(Action_3D)
    n, m, o = grid.shape[0] - 1, grid.shape[1] - 1, grid.shape[2] - 1
    x, y, z = current_node
    
    # check if the node is off the grid or
    # it's an obstacle
    if x > n or y > m or z > o:
        return []
    
    if x - 1 < 0 or grid[x-1, y, z] == 1:
        valid.remove(Action_3D.NORTH)
    if x + 1 > n or grid[x+1, y, z] == 1:
        valid.remove(Action_3D.SOUTH)
    if y - 1 < 0 or grid[x, y-1, z] == 1:
        valid.remove(Action_3D.WEST)
    if y + 1 > m or grid[x, y+1, z] == 1:
        valid.remove(Action_3D.EAST)
        
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1, z] == 1:
        valid.remove(Action_3D.NORTH_WEST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1, z] == 1:
        valid.remove(Action_3D.NORTH_EAST)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1, z] == 1:
        valid.remove(Action_3D.SOUTH_WEST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1, z] == 1:
        valid.remove(Action_3D.SOUTH_EAST)

    if z - 1 < 0 or grid[x, y, z - 1] == 1:
        valid.remove(Action_3D.DOWN)
    if z + 1 > o or grid[x, y, z + 1] == 1:
        valid.remove(Action_3D.UP)
        
    return valid


# In[ ]:


def a_star_3D(grid, h, start, goal):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    3D variation of a_star.
    the original a_star implementation could be refactored slightly to support both, but keeping the original implementation for now
    
    """
    
    if isinstance(start, (np.ndarray, np.generic)):
        # convert to tuple
        start = tuple(start.astype(int))
    if isinstance(goal, (np.ndarray, np.generic)):
        # convert to tuple
        goal = tuple(goal.astype(int))

    path = []
    path_cost = 0
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
            # Get the new vertexes connected to the current vertex
            for action in valid_actions_3D(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1], current_node[2] + da[2])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (queue_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


# In[ ]:


class Local3DPlanner(object):
    def __init__(self, waypoint_queue, data, north_offset, east_offset, voxel_size, altitude, max_altitude=20,
                 box_size=40):

        self.north_offset = north_offset
        self.east_offset = east_offset
        self.voxel_size = voxel_size
        self.box_size = box_size
        self.altitude = altitude
        self.max_altitude = max_altitude
        self.waypoint_queue = waypoint_queue
        self.next_waypoint = to_3d_numpy_array(self.waypoint_queue.get())
        self.current_position = None
        self.voxmap = create_voxmap(data, self.voxel_size, self.max_altitude)

    def end_reached(self):
        return self.waypoint_queue.empty()

    def update_position(self, current_position, ned=False):
        self.current_position = to_3d_numpy_array(current_position)
        if ned:
            # reverse altitude
            self.current_position[2] *= -1

        # determine local goal in local view
        self.local_box, self.local_goal = get_local_box_and_local_target(self.current_position, self.next_waypoint,
                                                                         size=self.box_size)
        while is_point_within_local_box(self.next_waypoint, self.local_box) and not self.waypoint_queue.empty():
            self.next_waypoint = to_3d_numpy_array(self.waypoint_queue.get())
            self.local_box, self.local_goal = get_local_box_and_local_target(self.current_position, self.next_waypoint,
                                                                             size=self.box_size)
            print("waypoint inside current view, moving on to next waypoint", self.next_waypoint)

        if len(self.local_goal) == 2:
            # if no altitude defined yet, use default altitude
            self.local_goal = np.concatenate([self.local_goal, [self.altitude]])
        self.local_goal = to_3d_numpy_array(self.local_goal)

        self.grid_local_box = [
            ((x1 + self.north_offset, y1 + self.east_offset), (x2 + self.north_offset, y2 + self.east_offset))
            for (x1, y1), (x2, y2) in self.local_box]

        # determine coordinates of local box in voxel map
        self.xmin, self.ymin = np.array(self.grid_local_box)[:, 0, :].min(axis=0)
        self.xmax, self.ymax = np.array(self.grid_local_box)[:, 0, :].max(axis=0)
        self.voxelxmin = int(np.floor(self.xmin)) // self.voxel_size
        self.voxelymin = int(np.floor(self.ymin)) // self.voxel_size
        self.voxelxmax = int(np.ceil(self.xmax)) // self.voxel_size
        self.voxelymax = int(np.ceil(self.ymax)) // self.voxel_size

        # grid coordinates
        self.grid_position = self.current_position + np.array([self.north_offset, self.east_offset, 0])
        self.grid_next_waypoint = self.next_waypoint + np.array([self.north_offset, self.east_offset, 0])
        self.grid_local_goal = self.local_goal + np.array([self.north_offset, self.east_offset, 0])

        # convert current position and local goal in voxmap
        self.grid_voxel_current_position = self.to_voxel_coordinates(self.grid_position)
        self.grid_voxel_local_goal = self.to_voxel_coordinates(self.grid_local_goal)

    def to_voxel_coordinates(self, p):
        v = (p // self.voxel_size) - np.array([self.voxelxmin, self.voxelymin, 0])
        max_voxel_value = (2 * self.box_size) // self.voxel_size
        return np.clip(v, 0, max_voxel_value - 1)

    def to_grid_coordinates(self, p):
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        return (p * self.voxel_size) + np.array([self.xmin, self.ymin, 0])

    def to_waypoints(self, path):
        waypoints = []
        for p in path:
            # add heading
            g = np.concatenate([self.to_grid_coordinates(p), np.array([0])]) - np.array(
                [self.north_offset, self.east_offset, 0, 0])
            waypoints.append(tuple(g))
        return waypoints

    def search(self):
        path, _ = a_star_3D(self.current_view, euclidean_distance, self.grid_voxel_current_position,
                            self.grid_voxel_local_goal)
        return path

    def plan(self):
        """ returns list of waypoints given the current view.
            also returns the unpruned local voxel path for debugging/plotting """
        if self.current_position is None:
            raise "no position set. use Local3DPlanner.update_position(<position>) before Local3DPlanner.plan "
        path = self.search()
        ppath = collinearity_prune(path)

        return self.to_waypoints(ppath), path

    def path_voxels(self, path):
        """ returns voxel view for plotting """
        path_voxels = np.zeros(self.current_view.shape, dtype=bool)
        for p in path:
            path_voxels[p[0], p[1], p[2]] = True
        return path_voxels

    @property
    def current_view(self):
        return self.voxmap[self.voxelxmin:self.voxelxmax, self.voxelymin:self.voxelymax, :]


# In[ ]:


def collinearity_prune(path, epsilon=1e-5):
    """
    Prune path points from `path` using collinearity.
    """
    def point(p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)
    
    def collinearity_check(p1, p2, p3):
        m=np.concatenate((p1, p2, p3), 0)
        det = np.linalg.det(m)
        return abs(det)<epsilon
    pruned_path=[p for p in path]
    i = 0
    while i < len(pruned_path) -2:
        p1=point(pruned_path[i])
        p2=point(pruned_path[i+1])
        p3=point(pruned_path[i+2])
        if collinearity_check(p1, p2, p3):
            pruned_path.remove(pruned_path[i+1])
        else:
            i+=1
    return pruned_path


# In[ ]:


def find_closest_skeleton_point(skeleton_coordinates, p):
    d = np.linalg.norm(skeleton_coordinates - p[:2], axis=1)
    return tuple(skeleton_coordinates[d.argmin()])


def find_start_goal(skeleton, start, goal):
    skeleton_coordinates = np.array(skeleton.nonzero()).T
    near_start = find_closest_skeleton_point(skeleton_coordinates, start)
    near_goal = find_closest_skeleton_point(skeleton_coordinates, goal)
    return near_start, near_goal


def collinearity_2D(p1, p2, p3):
    area = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    return area == 0


def collinearity_check(p1, p2, p3, epsilon=1e-6):
    if p1.shape[1] == 2:
        m = np.concatenate((p1, p2, p3), 0)
    else:
        m = np.concatenate((p1, p2, p3))
    det = np.linalg.det(m)
    return abs(det) < epsilon


# In[ ]:


def get_main_plan(global_home, goal_lat, goal_lon, current_local_pos,
                  safety_distance=3, target_altitude=5, filename='colliders.csv'):
    """
     returns a medial axis global plan.
     also returns data, north_offset & east_offset for reuse later
    """
    # Read in obstacle map
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    # Define a grid for a particular altitude and safety margin around obstacles
    grid, nmin, emin = create_grid(data, target_altitude, safety_distance)
    north_offset = -1 * nmin
    east_offset = -1 * emin
    print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
    # Define starting point on the grid (this is just grid center)
    # TODO: convert start position to current position rather than map center
    grid_start = (int(current_local_pos[0] + north_offset), int(current_local_pos[1] + east_offset))
    local_goal = global_to_local((goal_lon, goal_lat, 0), global_home)
    grid_goal = (int(local_goal[0] + north_offset), int(local_goal[1] + east_offset))

    print("Local Goal: {0}" % local_goal)
    # Run A* to find a path from start to goal
    # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
    # see planning_utils.py
    print('Local Start and Goal: ', grid_start, grid_goal)
    # Medial Axis
    path, _ = get_medial_axis_path(grid, grid_goal, grid_start)
    # TODO: prune path to minimize number of waypoints
    # TODO (if you're feeling ambitious): Try a different approach altogether!
    # step 1: prune collinear points from path
    print("number of waypoints: %d" % len(path))
    path = collinearity_prune(path)
    print("number of waypoints after pruning: %d" % len(path))
    print("number of waypoints after collinearity pruning: %d" % len(path))
    path = prune_path_using_raytracing(grid, path)
    print("number of waypoints after raytracing pruning: %d" % len(path))
    # Convert path to waypoints
    waypoints = [[p[0] - north_offset, p[1] - east_offset, target_altitude, 0] for p in path]
    return waypoints, data, north_offset, east_offset


# In[ ]:


def get_medial_axis_path(grid, grid_goal, grid_start):
    skeleton = medial_axis(invert(grid))
    skel_start, skel_goal = find_start_goal(skeleton, np.array(grid_start), np.array(grid_goal))
    # path, _ = a_star(grid, heuristic, skel_start, skel_goal)
    path, _ = a_star(invert(skeleton).astype(np.int), euclidean_distance, skel_start, skel_goal)
    path = path + [grid_goal]
    return path, skeleton


# In[ ]:


def prune_path_using_raytracing(grid, path):
    """
    optimize the path by looking for longer straight lines. recursive operation which will return the first
    and last point if there is a clear straight path (no obstacles) between them.
      otherwise split the path in the middle and process both partial paths separately and concatenate there results
    :param grid:
    :param path:
    :return: pruned path
    """
    if len(path) < 4:
        return path
    elif has_clear_path(grid, path[0], path[-1]):
        # clear path between given (partial)path, so return first and last waypoints
        return [path[0], path[-1]]
    else:
        # split at midpoint and prune right and left partial paths
        middle_i = len(path) // 2
        left_path = prune_path_using_raytracing(grid, path[:middle_i])
        right_path = prune_path_using_raytracing(grid, path[max(0, middle_i - 1):])
        return left_path + right_path[1:]


# In[ ]:


def euclidean_distance(n1, n2):
    a, b = n1, n2
    if not isinstance(n1, (np.ndarray, np.generic)):
        a = np.array(n1)
    if not isinstance(n2, (np.ndarray, np.generic)):
        b = np.array(n2)
    return np.linalg.norm(a - b)


# In[ ]:


def has_clear_path(grid, p1, p2):
    coord = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    for x, y in coord:
        if grid[x][y] == 1:
            return False
    return True


# In[ ]:


def to_3d_numpy_array(p):
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    if len(p) > 3:
        p = p[:3]
    return p


# In[ ]:


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b



def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


# In[ ]:


def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def prune_path(path):
    if len(path) < 3:
        return path
    way_points = [path[0]]
    next_candidate = path[1]
    for i in np.arange(len(path))[2:]:
        p1 = point(way_points[-1])
        p2 = point(next_candidate)
        p3 = point(path[i])
        if collinearity_check(p1, p2, p3):
            # drop current candidate in favour of new point
            next_candidate = path[i]
        else:
            # promote current candidate to waypoint
            way_points.append(next_candidate)
            next_candidate = path[i]

    if way_points[-1] != next_candidate:
        # add last candidate if not already done
        way_points.append(next_candidate)
    return way_points



# In[ ]:


def get_local_box_and_local_target(current_position, next_waypoint, size=40):
    local_box = get_box_edges(current_position, size=size)
    if is_point_within_local_box(next_waypoint, local_box):
        return local_box, next_waypoint
    else:
        for p1, p2 in local_box:
            local_goal = seg_intersect(np.array(p1), np.array(p2), current_position[:2], next_waypoint[:2])
            if (is_part_of_line_segment(local_goal, current_position[:2], next_waypoint[:2]) and
                    is_part_of_line_segment(local_goal, np.array(p1), np.array(p2))):
                return local_box, local_goal


# In[ ]:


def is_part_of_line_segment(ref, p1, p2, epsilon=1e-6):
    """ returns true if `ref` point is on the line segment between p1 and p2 """
    return abs(euclidean_distance(p1, ref) + euclidean_distance(ref, p2) - euclidean_distance(p1, p2)) < epsilon


from shapely.geometry import Polygon, Point


# In[ ]:


def is_point_within_local_box(p, local_box):
    """ returns true if point `p` is found inside `local_box` """
    return Point(p[0], p[1]).within(Polygon([x for x, y in local_box]))


# In[ ]:


def get_box_edges(current_position, size=40):
    x, y = current_position[:2]
    return [((x + size, y + size), (x - size, y + size)),
            ((x - size, y + size), (x - size, y - size)),
            ((x - size, y - size), (x + size, y - size)),
            ((x + size, y - size), (x + size, y + size))
            ]


# In[ ]:


def get_local_box_and_local_target(current_position, next_waypoint, size=40):
    local_box = get_box_edges(current_position, size=size)
    if is_point_within_local_box(next_waypoint, local_box):
        return local_box, next_waypoint
    else:
        for p1, p2 in local_box:
            local_goal = seg_intersect(np.array(p1), np.array(p2), current_position[:2], next_waypoint[:2])
            if (is_part_of_line_segment(local_goal, current_position[:2], next_waypoint[:2]) and
                    is_part_of_line_segment(local_goal, np.array(p1), np.array(p2))):
                return local_box, local_goal

