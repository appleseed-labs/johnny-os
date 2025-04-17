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

# # -----------------


import heapq
import numpy as np
import matplotlib.pyplot as plt
import cv2

def heuristic(a, b):
    # return abs(a[0] - b[0]) + abs(a[1] - b[1])
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** (1/2)

def a_star_algorithm(occupancy_grid, start, goal):
    # if obstacle at start position, ignore
    if(occupancy_grid[start]==1):
        occupancy_grid[goal] = 0
    # if obstacle at end position, return None
    if(occupancy_grid[goal]==1):
        print("goal is occupied")
        return None
    
    cols = len(occupancy_grid)
    rows = len(occupancy_grid[0])
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
            return path[::-1]  

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and occupancy_grid[neighbor] == 0:
                tentative_g = g_score[current] + 1 

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g +  heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

def dilate_grid(grid, dilate_width):
    dilated_kernel = np.ones((3, 3)).astype(np.uint8)
    np_grid = grid.astype(np.uint8)
    dilated_grid = (cv2.dilate((np_grid), dilated_kernel, iterations = dilate_width))
    return dilated_grid

def bresenham_line(p0, p1):
    """Yield integer coordinates on the line from p0 to p1"""
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield (x, y)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            yield (x, y)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield (x1, y1)

def clean_up_path(occupancy_grid, path):
    if not path or len(path) < 3:
        return path

    new_path = [path[0]]  # Always start with the first node
    last_added = path[0]

    for i in range(1, len(path)):
        line = list(bresenham_line(last_added, path[i]))
        print(path[i],":")
        print("line",line)
        print()

        if any(occupancy_grid[p] == 1 for p in line):
            # Obstacle encountered — last safe node is path[i-1]
            new_path.append(path[i - 1])
            last_added = path[i - 1]
            print("last_added:", last_added)

    # Ensure goal is added
    if new_path[-1] != path[-1]:
        new_path.append(path[-1])

    return new_path


        

# # Test 1
# # Example occupancy grid (5x5)
# occupancy_grid = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 1],
#     [1, 0, 0, 0, 1]
# ])

# start = (0, 0)
# goal = (4, 4)

# path = a_star_algorithm(occupancy_grid, start, goal)
# print("Path:", path)


# # Test 2
# # Create grid
# grid_size = 50
# occupancy_grid = np.zeros((grid_size, grid_size), dtype=int)

# # Place a few obstacles
# occupancy_grid[10:25, 10:20] = 1

# # Set start and goal
# start = (grid_size // 2, grid_size // 2)        # Center
# goal = (1, 1)                       # Near top-right
# occupancy_grid[start] = 0
# occupancy_grid[goal] = 0

def generate_circular_obstacles(grid, center, radius):
    """Generates circular obstacles in the grid around the center with the specified radius."""
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                grid[i, j] = 1  # mark as obstacle
    return grid

# Example grid setup
grid_size = 50
occupancy_grid = np.zeros((grid_size, grid_size), dtype=int)

# Set start and goal at the middle and near top-left
start = (grid_size // 2, grid_size // 2)
goal = (1, grid_size - 2)

# Generate circular obstacles between start and goal
# Randomly place 3 circular obstacles between start and goal
obstacle_centers = [(np.random.randint(grid_size), np.random.randint(grid_size)) for _ in range(3)]
for center in obstacle_centers:
    radius = np.random.randint(3, 6)  # Random radius between 3 and 6
    occupancy_grid = generate_circular_obstacles(occupancy_grid, center, radius)

# Ensure start and goal are clear
occupancy_grid[start] = 0
occupancy_grid[goal] = 0

dilated_occupancy_grid = dilate_grid(occupancy_grid, 2)
# Run A* algorithm
raw_path = a_star_algorithm(dilated_occupancy_grid, start, goal)
smooth_path = clean_up_path(dilated_occupancy_grid, raw_path)
print(raw_path)
print(smooth_path)

# Visualize
def visualize_path(original_grid, dilated_grid, raw_path, smooth_path, start, goal):
    # Create a combined grid for visualization
    vis_grid = np.zeros_like(original_grid, dtype=int)

    # 1 = original obstacle, 2 = dilated only
    vis_grid[np.where(dilated_grid == 1)] = 2
    vis_grid[np.where(original_grid == 1)] = 1

    # Define colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'black', 'gray'])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(vis_grid, cmap=cmap, origin='upper')

    # Draw raw path
    if raw_path:
        x_raw, y_raw = zip(*raw_path)
        ax.plot(y_raw, x_raw, color='blue', linewidth=2, label='Raw Path')

    # Draw smoothed path
    if smooth_path:
        x_smooth, y_smooth = zip(*smooth_path)
        ax.plot(y_smooth, x_smooth, color='orange', linewidth=2, linestyle='--', label='Smoothed Path')

    # Draw start and goal
    ax.scatter([start[1]], [start[0]], c='green', s=100, label='Start')
    ax.scatter([goal[1]], [goal[0]], c='red', s=100, label='Goal')

    # Gridlines for clarity
    ax.set_xticks(np.arange(-0.5, original_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, original_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.set_title("Raw vs Smoothed Path with Obstacle & Dilation Overlay")
    ax.legend()
    plt.show()

visualize_path(occupancy_grid, dilated_occupancy_grid, raw_path, smooth_path, start, goal)