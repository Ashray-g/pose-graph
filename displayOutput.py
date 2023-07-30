import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

ground_truth_x = np.linspace(0, 4.9, 50)
ground_truth_y = np.linspace(0, 0, 50)

initial_x = []
initial_y = []

solve_x = []
solve_y = []

# Read in output (50 lines)

file0 = open("cmake-build-debug/initial_guess.txt", "r+")
file1 = open("cmake-build-debug/solved_out.txt", "r+")

s0 = file0.readlines()
s1 = file1.readlines()

for i in range(0, 50):
    initial_x.append(float(s0[i].split()[0]))
    initial_y.append(float(s0[i].split()[1]))

    solve_x.append(float(s1[i].split()[0]))
    solve_y.append(float(s1[i].split()[1]))

plt.xlim(-0.1, 5)
plt.ylim(-1, 1)

plt.scatter(ground_truth_x, ground_truth_y, color="limegreen", marker='x')
plt.scatter(initial_x, initial_y, color="r", marker="+")
plt.scatter(solve_x, solve_y, color="deepskyblue", marker="x")
plt.show()