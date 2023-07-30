import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

show = 20
# show  = 10;

ground_truth_x = np.linspace(0, 0.1 * (show - 1), show)
ground_truth_y = np.linspace(0, 0, show)
ground_truth_z = np.linspace(0, 0, show)

initial_x = []
initial_y = []
initial_z = []

solve_x = []
solve_y = []
solve_z = []

land_x = []
land_y = []
land_z = []

# Read in output (50 lines)
file0 = open("cmake-build-debug/initial_guess.txt", "r+")
file1 = open("cmake-build-debug/solved_out.txt", "r+")
file2 = open("cmake-build-debug/landmark_position.txt", "r+")

s0 = file0.readlines()
s1 = file1.readlines()
s2 = file2.readlines()

for i in range(0, show):
    initial_x.append(float(s0[i].split()[0]))
    initial_y.append(float(s0[i].split()[1]))
    initial_z.append(float(s0[i].split()[2]))

    solve_x.append(float(s1[i].split()[0]))
    solve_y.append(float(s1[i].split()[1]))
    solve_z.append(float(s1[i].split()[2]))

for i in range(0, 5):
    land_x.append(float(s2[i].split()[0]))
    land_y.append(float(s2[i].split()[1]))
    land_z.append(float(s2[i].split()[2]))

# plt.scatter(ground_truth_x, ground_truth_y, color="limegreen", marker='x')
# plt.scatter(initial_x, initial_y, color="r", marker="+")
# plt.scatter(solve_x, solve_y, color="deepskyblue", marker="x")
# plt.show()

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(ground_truth_x, ground_truth_y, ground_truth_z, color="limegreen",  marker='x')
ax.scatter3D(initial_x, initial_y, initial_z, color="r", marker="+")
ax.scatter3D(solve_x, solve_y, solve_z, color="deepskyblue", marker="x")
ax.scatter3D(land_x, land_y, land_z, color="yellow", marker="^")
plt.title("simple 3D scatter plot")

# show plot
plt.show()