import argparse
import time
import msgpack
from enum import Enum
from enum import auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid, collinearity_prune
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, goal_global_position=None):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL
        self.goal_global_position = goal_global_position

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:    #if len(self.all_waypoints)>0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()   #self.takeoff_transition()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()
    
    #def calculate_box(self):
        #print("Setting Home")
        #local_waypoints = [[10.0, 0.0, 3.0], [10.0, 10.0, 3.0], [0.0, 10.0, 3.0], [0.0, 0.0, 3.0]]
        #return local_waypoints

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()
        #self.set_home_position(self.global_position[0], self.global_position[1],
                               #self.global_position[2])  # set the current location to be the home position
        #self.flight_state = States.ARMING


    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])
        
    #def takeoff_transition(self):
        #print("takeoff transition")
        ## self.global_home = np.copy(self.global_position)  # can't write to this variable!
        #target_altitude = 3.0
        #self.target_position[2] = target_altitude
        #self.takeoff(target_altitude)
        #self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as f:
            home_pos_data = f.readline().split(",")
        lat0 = float(home_pos_data[0].strip().split(" ")[1])
        lon0 = float(home_pos_data[1].strip().split(" ")[1])
        print(lat0, lon0)
        
        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)
        self.global_home(lon0, lat0, 0)

        # TODO: retrieve current global position
        local_north, local_east, local_down=global_to_local(self.global_position, self.global_home)
        print("Local => north :" , local_north, "east :", local_east, "down :", local_down)
 
        # TODO: convert to current local position using global_to_local()
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start_north = int(np.ceil(local_north - north_offset))
        grid_start_east = int(np.ceil(local_east - east_offset))
        grid_start = (grid_start_north, grid_start_east)
        # TODO: convert start position to current position rather than map center
        goal_north, goal_east, goal_alt = global_to_local(self.goal_global_position, self.global_home)
        grid_goal = (int(np.ceil(goal_north-north_offset)), int(np.ceil(goal_east-east_offset)))
        
        
        
        # Set goal as some arbitrary position on the grid
        #grid_goal = (-north_offset + 10, -east_offset + 10)
        # TODO: adapt to set goal as latitude / longitude position and convert
        #local_goal[0]=-122.4081522
        #local_goal[1]=37.7942581
        #local_goal[0]=-122.39287756
        #local_goal[1]=37.79606176
        #goal_north, goal_east, goal_alt=global_to_local(self.global_position, self.global_home)
        #grid_goal=(int(np.ceil(goal_north-north_offset)), int(np.ceil(goal_east-east_offset)))
        #print("Local Goal: {0}" % local_goal)

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        # TODO: prune path to minimize number of waypoints
        print("number of waypoints: %d" % len(path))
        path=collinearity_prune(path)
        print("number of waypoints after pruning: %d" % len(path))
        print("number of waypoints after collinearity pruning: %d" % len(path))
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        print(waypoints)
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()
        #super().start()
        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--goal_lon', type=str, help="Goal longitude")
    parser.add_argument('--goal_lat', type=str, help="Goal latitude")
    parser.add_argument('--goal_alt', type=str, help="Goal altitude")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    goal_global_position = np.fromstring(f'{args.goal_lon},{args.goal_lat},{args.goal_alt}', dtype='Float64', sep=',')
    drone = MotionPlanning(conn, goal_global_position=goal_global_position)
    time.sleep(1)

    drone.start()
