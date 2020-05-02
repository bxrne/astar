import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

grid = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Points of Interest
start = (0,0)
goal = (0,19)

#Heuristic - Pythagoras' theorem 
def heuristic(a,b):
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def astar(array, start, goal, movement):
    #possible movemnt
    neighbours = []
    if movement == 1:
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    else:
        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
   #tiles not to choose ever again
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start,goal)}

    #contains all positions considered
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    #checking for open positions
    while oheap:
        #find the one with the smallest f score(overall cost (g+h))
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data
        #forget that one then
        close_set.add(current)
        #calculate g scores for all possible neighbors
        for i,j in neighbors:
            neighbor = current[0] + i, current[1] +j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                     #array bound y walls
                    continue
            else:
                #array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

#implementation
route1 = astar(grid, start, goal, 0)
x1 = []
y1 = []
distance1 = 0

#add route into two seperate arrays for plotting
for i in range(0, len(route1)):
    nx = route1[i][0]
    ny = route1[i][1]
    x1.append(nx)
    y1.append(ny)
    distance1 += np.sqrt((nx)**2 + (ny)**2)
#add start point > not included, skips itself as a step
x1.append(start[0])
y1.append(start[1])
print(distance1)

#implementation
route2 = astar(grid, start, goal, 1)
x2 = []
y2 = []
distance2 = 0

#add route into two seperate arrays for plotting
for i in range(0, len(route2)):
    nx = route2[i][0]
    ny = route2[i][1]
    x2.append(nx)
    y2.append(ny)
    distance2 += np.sqrt((nx)**2 + (ny)**2)
#add start point > not included, skips itself as a step
x2.append(start[0])
y2.append(start[1])
print(distance2)


fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(grid, cmap=plt.cm.binary)
ax.plot(y1,x1, color="#bdc3c7", linewidth=3, zorder=10, label=str(round(distance1,1)) + " units")
ax.plot(y2,x2, color="blue", linewidth=3, zorder=10, label =  str(round(distance2,1)) + " units")
ax.scatter(start[1],start[0], marker = "*", color = "#27ae60", s = 200, zorder=20)
ax.scatter(goal[1],goal[0], marker = "*", color = "#c0392b", s = 200, zorder=20)
plt.title("A* implementations")
ax.legend()
plt.show()
