This folder provides an example of a CDL description for a simple pick-and-place domain.

### Overview

Use the following commands to run the example:

```bash
# Runs the example --- loads the domain and prints its content
python3 1-load-crow-domain.py

# Visualize the domain in the terminal
cdl-highlight ../../../../concepts/dm/crowhat/domains/RWI-pp-simplified.cdl 
```

```bash
# Run the planner for the "pick-and-place" problem
python3 4-solve-pick-and-place.py both

# The previous command will run two processes: one for the simulated world and one for the physical world.
# Alternatively, you can run the simulated and the physical world separately:

# In shell 1
python3 4-solve-pick-and-place.py server
# In shell 2
python3 4-solve-pick-and-place.py main
```

Since this is a robot manipulation domain, it contains a significant amount of infrastructure.

At a high-level, there are two world views:

- The (remote) physical world, where the robot operates.
- The (local) simulated world, where the planner operates.

In a simulated environment, we use two separate processes to simulate the remote and the local world.
In a real-robot environment, the remote world is the real world.

To connect the two worlds, there are two bridges connecting the two worlds:

- The `perception_interface`, which provides the planner with the current state of the remote world.
- The `remote_physical_controller_interface`, which allows the planner to send actions to the remote world.

Inside the robot's brain (the planner), there is another control interface, named `local_physical_controller_interface`.

The outer loop is called `Execution Manager`. In this example, we are using an OpenLoopExecutionManager, which simply executes the plan without any feedback.

```python
def run(goal):
    update_perception()  # Use the perception interface to update the planner's world view
    plan = planner.plan(goal)  # Plan for the given goal, which internally uses the local_physical_controller_interface
    for action in plan:
        send_command(action)  # Use the remote_physical_controller_interface to send the action to the remote world
```


### Additional Scripts

- `2-load-qddl-scene.py`: Loads the scene from a QDDL file. It starts two processes: one for the simulated world and one for the physical world.
- `3-control-panda.py`: A simple script to control the Panda robot in the simulator.
