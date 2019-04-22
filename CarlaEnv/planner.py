import numpy as np

# ==============================================================================
# -- import route planning (copied CARLA 0.9.4's PythonAPI) --------------------
# ==============================================================================
import carla
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import vector

def compute_lap(start_waypoint, hop_resolution=1.0):
    # Town07 lap is 7 straight turns
    route = [RoadOption.STRAIGHT] * 7

    # Compute route waypoints
    current_waypoint = start_waypoint
    route.append(RoadOption.VOID)
    solution = []
    for i, action in enumerate(route):
        # Generate waypoints to next junction
        wp_choice = [current_waypoint]
        while len(wp_choice) == 1:
            current_waypoint = wp_choice[0]
            solution.append((current_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = current_waypoint.next(hop_resolution)
            # Stop at destination
            if i > 0 and current_waypoint.transform.location.distance(start_waypoint.transform.location) < hop_resolution:
                break
        if action == RoadOption.VOID:
            break

        # Select appropriate path at the junction
        if len(wp_choice) > 1:
            # Current heading vector
            current_transform = current_waypoint.transform
            current_location = current_transform.location
            projected_location = current_location + \
                carla.Location(
                    x=np.cos(np.radians(current_transform.rotation.yaw)),
                    y=np.sin(np.radians(current_transform.rotation.yaw)))
            v_current = vector(current_location, projected_location)

            direction = 0
            if action == RoadOption.LEFT:
                direction = 1
            elif action == RoadOption.RIGHT:
                direction = -1
            elif action == RoadOption.STRAIGHT:
                direction = 0
            select_criteria = float("inf")

            # Choose correct path
            for wp_select in wp_choice:
                v_select = vector(
                    current_location, wp_select.transform.location)
                cross = float("inf")
                if direction == 0:
                    cross = abs(np.cross(v_current, v_select)[-1])
                else:
                    cross = direction * np.cross(v_current, v_select)[-1]
                if cross < select_criteria:
                    select_criteria = cross
                    current_waypoint = wp_select

            # Generate all waypoints within the junction
            # along selected path
            solution.append((current_waypoint, action))
            current_waypoint = current_waypoint.next(hop_resolution)[0]
            while current_waypoint.is_intersection:
                solution.append((current_waypoint, action))
                current_waypoint = current_waypoint.next(hop_resolution)[0]
    assert solution
    return solution

def plan_route(start_waypoint, end_waypoint):
    # Setting up global router
    dao = GlobalRoutePlannerDAO(m)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    # Obtain route plan
    x1 = start_waypoint.transform.location.x
    y1 = start_waypoint.transform.location.y
    x2 = end_waypoint.transform.location.x
    y2 = end_waypoint.transform.location.y
    route = grp.plan_route((x1, y1), (x2, y2))
    route.append(RoadOption.VOID)
    return start_waypoint, end_waypoint, route

def compute_route_waypoints(start_waypoint, end_waypoint, route, hop_resolution=1.0):
    # Compute route waypoints
    solution = []
    current_waypoint = start_waypoint
    for i, action in enumerate(route):
        # Generate waypoints to next junction
        wp_choice = [current_waypoint]
        while len(wp_choice) == 1:
            current_waypoint = wp_choice[0]
            solution.append((current_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = current_waypoint.next(hop_resolution)
            # Stop at destination
            if i > 0 and current_waypoint.transform.location.distance(end_waypoint.transform.location) < hop_resolution:
                break
        if action == RoadOption.VOID:
            break

        # Select appropriate path at the junction
        if len(wp_choice) > 1:
            # Current heading vector
            current_transform = current_waypoint.transform
            current_location = current_transform.location
            projected_location = current_location + \
                carla.Location(
                    x=np.cos(np.radians(current_transform.rotation.yaw)),
                    y=np.sin(np.radians(current_transform.rotation.yaw)))
            v_current = vector(current_location, projected_location)

            direction = 0
            if action == RoadOption.LEFT:
                direction = 1
            elif action == RoadOption.RIGHT:
                direction = -1
            elif action == RoadOption.STRAIGHT:
                direction = 0
            select_criteria = float("inf")

            # Choose correct path
            for wp_select in wp_choice:
                v_select = vector(
                    current_location, wp_select.transform.location)
                cross = float("inf")
                if direction == 0:
                    cross = abs(np.cross(v_current, v_select)[-1])
                else:
                    cross = direction * np.cross(v_current, v_select)[-1]
                if cross < select_criteria:
                    select_criteria = cross
                    current_waypoint = wp_select

            # Generate all waypoints within the junction
            # along selected path
            solution.append((current_waypoint, action))
            current_waypoint = current_waypoint.next(hop_resolution)[0]
            while current_waypoint.is_intersection:
                solution.append((current_waypoint, action))
                current_waypoint = current_waypoint.next(hop_resolution)[0]
    assert solution
    return solution
