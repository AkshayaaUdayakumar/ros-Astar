#!/usr/bin/env python3

import rospy
import math
import heapq
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker

class AStarNode:
    def __init__(self):
        rospy.init_node('astar_node')
        
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.path_pub = rospy.Publisher('/astar_path', Path, queue_size=10)
        self.vis_pub = rospy.Publisher('/astar_marker', Marker, queue_size=10)
        
        self.map = None
        self.goal = None
        self.map_data = None

    def map_callback(self, msg):
        self.map = msg
        self.map_data = self.convert_map_to_grid(msg)
        rospy.loginfo("Map received")

    def convert_map_to_grid(self, msg):
        grid = []
        for y in range(msg.info.height):
            row = []
            for x in range(msg.info.width):
                idx = x + (msg.info.height - y - 1) * msg.info.width
                if msg.data[idx] == 0:
                    row.append(0)
                else:
                    row.append(1)
            grid.append(row)
        return grid

    def goal_callback(self, msg):
        self.goal = msg
        rospy.loginfo("Goal received: (%f, %f)", msg.pose.position.x, msg.pose.position.y)
        if self.map:
            self.run_astar()

    def run_astar(self):
        start = (int((0 - self.map.info.origin.position.x) / self.map.info.resolution), 
                 int((0 - self.map.info.origin.position.y) / self.map.info.resolution))
        goal = (int((self.goal.pose.position.x - self.map.info.origin.position.x) / self.map.info.resolution), 
                int((self.goal.pose.position.y - self.map.info.origin.position.y) / self.map.info.resolution))
        
        path = self.astar(start, goal)
        
        if path:
            rospy.loginfo("Path found: %s", path)
            ros_path = Path()
            ros_path.header.frame_id = "map"
            ros_path.header.stamp = rospy.Time.now()
            for p in path:
                pose = PoseStamped()
                pose.pose.position.x = p[0] * self.map.info.resolution + self.map.info.origin.position.x
                pose.pose.position.y = p[1] * self.map.info.resolution + self.map.info.origin.position.y
                pose.pose.position.z = 0
                ros_path.poses.append(pose)
            self.path_pub.publish(ros_path)
            self.visualize_path(ros_path)
        else:
            rospy.logwarn("No path found")

    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return None

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def distance(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(self, node):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        neighbors = []
        for d in directions:
            neighbor = (node[0] + d[0], node[1] + d[1])
            if self.is_within_bounds(neighbor) and self.is_free(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def is_within_bounds(self, node):
        x, y = node
        return 0 <= x < len(self.map_data[0]) and 0 <= y < len(self.map_data)

    def is_free(self, node):
        x, y = node
        return self.map_data[y][x] == 0

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def visualize_path(self, ros_path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.pose.orientation.w = 1.0

        for pose in ros_path.poses:
            p = Point()
            p.x = pose.pose.position.x
            p.y = pose.pose.position.y
            p.z = 0
            marker.points.append(p)

        self.vis_pub.publish(marker)

if __name__ == '__main__':
    try:
        node = AStarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

