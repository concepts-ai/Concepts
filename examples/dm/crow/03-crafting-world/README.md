This folder provides an example of a CDL description for a Mini-Minecraft domain.

The Mini-Minecraft domain is a simplified version of the Minecraft domain, where the agent can move in a 2D grid and interact with the environment by mining and crafting objects.
In the default setup, the environments only begin with raw resources and two starting tools (axe and pickaxe).
Agents may need to compose multiple skills to obtain other downstream resources.

### Overview

Use the following commands to run the example:

```bash
# Runs the example --- loads the domain and prints its content
python3 1-load-crow-domain.py

# Visualize the domain in the terminal
cdl-highlight ../../../../concepts/benchmark/gridworld/crafting_world/crow_domains/crafting_world_station_agnostic.cdl
```

```bash
# Run the planner for the "crafting-world" problem
python3 2-solve-crafting-world.py
```

You can specify additional `--target` arguments to run different scenarios. For example, to test "single-goal-item" tasks, run:

```bash
python3 2-solve-crafting-world.py --target single-goal
```

To test "double-goal-item" tasks, run:

```bash
python3 2-solve-crafting-world.py --target double-goal
```

To test a custom goal configuration, run:

```bash
python3 2-solve-crafting-world.py --target custom
```

See the `2-solve-crafting-world.py` script for more details.

You can also directly run the planner with the following command:

```bash
cdl-plan crafting-world-problem-example.cdl --print-stat --is-goal-ordered=False --is-goal-serializable=True --always-commit-skeleton=True --min-search-depth=15
cdl-plan crafting-world-problem-example-with-pragma.cdl --print-stat
```

