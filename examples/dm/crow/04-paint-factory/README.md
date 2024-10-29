This folder provides an example of a CDL description for a robot paint factory domain.

There are three zones, several bowls, and several blocks on the table.
The robot can use its suction gripper to pick-and-place objects into a designated location.
Blocks have 8 possible colors. Placing objects in a bowl will paint the object to be the same color as
the bowl. The task is to paint the blocks and organize them in the target brown box. Our training-time
goal requires the robot to paint-and-place two objects. The goal contains their colors and their
relationship (e.g., a pink block is left of a yellow block. Demonstration are collected using handcrafted oracle policies.
The offline dataset contains only successful trajectories.


### Overview

Use the following commands to run the example:

```bash
# Runs the example --- loads the domain and prints its content
python3 1-load-crow-domain.py
```
