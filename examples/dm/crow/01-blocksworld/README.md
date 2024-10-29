This folder provides an example of a CDL description for Blocks World, a classic planning domain.

### Overview

Use the following commands to run the example:

```bash
# Runs the example --- loads the domain and prints its content
python3 1-load-crow-domain.py

# Visualize the domain in the terminal
cdl-highlight blocksworld.cdl
```

```bash
# Run the planner for the "Sussman Anomaly" problem
python3 2-solve-blocksworld.py

# You can significantly improves the performance by using a more "breakdown" specification of the "on" behavior.
python3 2-solve-blocksworld.py --domain blocksworld-breakdown.cdl
```

The only difference between the two commands is the definition for the "put-on" behavior.

In the basic version, we have:

```cdl
behavior r_on(x: block, y: block):
  goal: on(x, y)
  body:
    promotable sequential:
      achieve clear(y)
      achieve holding(x)
    stack(x, y)
  eff:
    clear[y] = False
    holding[x] = False
    on[x, y] = True
    clear[x] = True
    handempty[...] = True
```

Planning results:

```
Plan for: on(B, C) and on(A, B)
Plan 0: unstack(C, A); place_table(C); pickup_table(B); stack(B, C); pickup_table(A); stack(A, B)
{'nr_expanded_nodes': 2121}
Press Enter to continue...
```

In the "breakdown" version, we have:

```cdl
action r_on(x: block, y: block):
  goal: on(x, y)
  body:
    promotable unordered:
      achieve clear(y)
      achieve clear(x)
    achieve_once handempty()
    untrack clear(x)
    achieve holding(x)
    stack(x, y)
  eff:
    ... # Same as before
```

This breakdown version explicitly states the promotion of the "clear" predicate for both blocks.
Planning results:

```
Plan for: on(B, C) and on(A, B)
Plan 0: unstack(C, A); place_table(C); pickup_table(B); stack(B, C); pickup_table(A); stack(A, B)
{'nr_expanded_nodes': 22}
```

You can also run the planner manually:

```bash
# Run the planner
cdl-plan blocksworld.cdl blocksworld-problem-sussman.cdl --print-stat

# Run the planner with the breakdown version
cdl-plan blocksworld-breakdown.cdl blocksworld-problem-sussman.cdl --print-stat

# Run the planner assuming the goal is ordered
cdl-plan blocksworld-breakdown.cdl blocksworld-problem-sussman.cdl --print-stat --is-goal-serializable=False --is-goal-ordered=True --always-commit-skeleton=True

# You can also specify planner options inside the CDL file
cdl-plan blocksworld-problem-sussman-with-pragma.cdl --print-stat

# To use "custom" policies, you can use the following command
cdl-plan blocksworld-problem-sussman-with-custom-policy.cdl --print-stat
```

