This folder provides an example of a CDL description for a domain, including types, functions, and behaviors.

### Overview

Use the following commands to run the example:

```bash
# Runs the example --- loads the domain and prints its content
python3 1-load-crow-domain.py

# Visualize the domain in the terminal
cdl-highlight crow-demo.cdl

# Visualize the domain in the browser. Useful for generating visualizations (e.g., to PowerPoint)
cdl-highlight crow-demo.cdl --html

# Load the domain and summarize its content. This is useful for checking syntax.
cdl-summary crow-demo.cdl
```

### Other Examples

- Usage of string types:
```bash
cdl-plan string-demo.cdl
```
- Usage of the `alternative` statements:
```bash
cdl-plan alternative-demo.cdl
```
- Usage of list types.
```bash
cdl-plan list-type-demo.cdl
```

