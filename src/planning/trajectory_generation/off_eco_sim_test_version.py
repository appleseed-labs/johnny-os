# import numpy as np


# GRID_RES = 0.25
# def createOccupancy_grid(processed_pc):
#     processed_pc = processed_pc[processed_pc[:, 2] > 0.5]
#     max_x = max(processed_pc[:,0])
#     min_x = min(processed_pc[:,0])
#     max_y = max(processed_pc[:,1])
#     min_y = min(processed_pc[:,1])
#     grid_min_x = int(min_x/GRID_RES) * GRID_RES
#     grid_min_y = int(min_y/GRID_RES) * GRID_RES
#     # print(min_x, min_y)
#     # print(grid_min_x, grid_min_y)
#     # print()

#     width = int((max_x - grid_min_x)/GRID_RES) + 1
#     height = int((max_y - grid_min_y)/GRID_RES) + 1

#     grid = np.zeros((height, width), dtype = np.int8)

#     for (x, y, z) in processed_pc:
#         grid_x = int((x - grid_min_x)/GRID_RES)
#         grid_y = int((y - grid_min_y)/GRID_RES)
#         # print("x,y,z:", x, y, z)
#         # print("grid_y, grid_x",grid_y, grid_x)
#         grid[grid_y, grid_x] = 1
#         # print(grid)
#         print()

#     print(grid)

#     return grid.flatten()


# processed_pc = np.array([[1.678, 0.535, 3.0], [1.235, 0.12, 3.0], [-0.87, 0.89, 3.0]])
# # processed_pc = np.array([[1.678, 0.535, 3.0], [1.235, 0.12, 3.0], [-0.87, 0.89, 3.0], [-0.87, - 1.23, 0.0]])
# # processed_pc = np.array([[1.6, 0, 3.0], [-0.8, 0, 3.0]])
# print(createOccupancy_grid(processed_pc))

# # -------
import heapq
import numpy as np

def heuristic(a, b):
    """Compute Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(occupancy_grid, start, goal):
    goal_occupied = False
    # if obstacle at start position, ignore
    if(occupancy_grid[start]==1):
        occupancy_grid[start] = 0
    # if obstacle at end position, return None
    if(occupancy_grid[goal]==1):
        goal_occupied = True
        print("goal is occupied")
        occupancy_grid[goal] = 0
    
    rows, cols = occupancy_grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_list:
        _, current = heapq.heappop(open_list)

        # Return path in correct order
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            if(goal_occupied):
                # return path expect the goal position
                return path[:0:-1]
            else:
                return path[::-1]  

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and occupancy_grid[neighbor] == 0:
                tentative_g = g_score[current] + 1 

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

# Example occupancy grid (5x5)
occupancy_grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1]
])

start = (0, 0)
goal = (4, 4)

path = astar(occupancy_grid, start, goal)
print("Path:", path)