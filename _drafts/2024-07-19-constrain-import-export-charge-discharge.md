---
title: Avoiding Simultaneous Import and Export in Linear Programs of Energy Systems
date_created: '2024-07-21'
date: '2023-07-21'
categories:
  - Energy
excerpt: TODO
toc: true
toc_sticky: true
---

This post explains how to constrain simultaneous import and export in linear programs. It's a particular problem that affects most linear program models of energy systems. To solve it, we will introduce two linear programming tricks:

1. Constraining the upper and lower bounds of continuous variables to both a non-zero lower bound and zero,
2. Linking two continuous variables together through binary variables.

The code in this post depends on the PuLP library for a linear programming framework, and Pandas for working with the results.

You can install them both into a Python environment with:

```shell-session
$ pip install pulp pandas
```

# Context

When modelling energy systems, we are often modelling assets where electricity can flow in opposite directions. Examples include:

- a site that can import or export electricity,
- a battery that can charge or discharge electricity.

Electricity cannot flow in opposite directions at the same time. 

Most linear program models of energy systems discretize time - the time between the start and end of an interval is split into intervals, often 30 minutes long.

Energy balances are created for each interval. For a single interval of time, there can only be one of import of export electricity.

In reality, it's possible a site both imports and exports within an interval.  However, there would have been no period during that 30 minutes where both import and export occurred simultaneously.

Making a linear program modelling an energy system deal with this simultaneous import and export problem is not trivial - hence this blog post.

# A Simple Site with Import, Export, Demand and Generation

To demonstrate the problem, we will start with a site that has:

- a electric load,
- a generator,
- a grid connection that can import and export.

The code below sets this scenario as a linear program using the PuLP Python library:

```python
import pulp


def extract_inputs_and_results(inputs: dict[str, float], prob: pulp.LpProblem) -> dict[str, float]:
    results = {f"input--{k}": v for k, v in inputs.items()}
    results["status"] = pulp.LpStatus[prob.status]
    for v in prob.variables():
        results[f"variable--{v.name}"] = v.varValue
    return results


def run_linear_program(generation: float = 25, demand: float = 50) -> dict[str, float]:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, 100)
    site_export = pulp.LpVariable("site_export", 0, 100)

    prob += site_import + generation == site_export + demand
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return extract_inputs_and_results(
        {"generation": generation, "demand": demand}, prob
    )


run_linear_program()
```

```output
{'input--generation': 25,
 'input--demand': 50,
 'status': 'Optimal',
 'variable--__dummy': None,
 'variable--site_export': 0.0,
 'variable--site_import': 25.0}
```

There are a few things to note:

- The program has a single constraint (an energy balance around the site) - it doesn't have an objective function,
- the upper bounds on `site_import` and `site_export` are both set to 100 - if the upper bound is not set, the program can become infeasible.

We can check this works correctly by changing the on-site generation and seeing how the import and export change:

```python
import pandas as pd

pd.DataFrame(
    run_linear_program(generation) for generation in [0, 25, 50, 100]
)[["input--generation", "variable--site_export", "variable--site_import"]]
```

```output
   input--generation  variable--site_export  variable--site_import
0                  0                    0.0                   50.0
1                 25                    0.0                   25.0
2                 50                   -0.0                    0.0
3                100                   50.0                    0.0
```

The program above works fine - as generation increases, we reduce import and increase export.

## Simultaneous Import and Export

We can force our linear program to import and export simultaneously by setting the import price lower than the export price.

This will require adding an objective function to our program, as well as separate import and export electricity prices.

The scenario we setup is one where export prices are lower than import prices. While this rarely happens in practice, it is possible.

A naive implementation will allow simultaneous import and export.  In a scenario where export prices are lower than import prices, the optimal solution is to import and export electricity in the same interval. 

```python
import pandas as pd
import pulp


def run_linear_program(
    generation: float,
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, 100)
    site_export = pulp.LpVariable("site_export", 0, 100)

    prob += site_import + generation == site_export + demand
    prob += site_import * import_price - site_export * export_price
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)

run_linear_program(25)
```

```output
status = Optimal
site_export = 75.0
site_import = 100.0
```

We now see that our program imports power and then exports it within the same interval. 

While this is optimal for how we have setup the program (well done solver!), it's not physically possible.

## Stopping Simultaneous Import and Export

To prevent simultaneous import and export, we can introduce binary variables. This will also turn our linear program into a mixed integer linear program. 

The binary variables link the import and export electricity flow together, with two additional constraints on the import and export (two constraints in total):

```python
import pulp


def run_linear_program(
    generation: float,
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
    site_import_limit: int = 100,
    site_export_limit: int = 100,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, None)
    site_export = pulp.LpVariable("site_export", 0, None)

    prob += site_import * import_price - site_export * export_price
    prob += site_import + generation == site_export + demand

    site_import_binary = pulp.LpVariable("site_import_binary", cat="Binary")
    site_export_binary = pulp.LpVariable("site_export_binary", cat="Binary")

    prob += site_import - site_import_limit * site_import_binary <= 0
    prob += site_export - site_export_limit * site_export_binary <= 0
    prob += site_import_binary + site_export_binary == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)


run_linear_program(25)
```

```output
status = Optimal
site_export = 0.0
site_export_binary = 0.0
site_import = 25.0
site_import_binary = 1.0
```

Now we see that our site only imports electricity.

## Explaining the Tricks

There are two tricks we play to get this to work:

1. Constrain the upper bound on `site_import` and `site_export`,
2. Link two continuous variables together through two binary variables.

### Constraining Upper and Lower Bounds

When defining a linear program variable in PuLP, we can easily set the upper and lower bounds:

```python
import pulp

var = pulp.LpVariable("a-name", 0, 100)
```

What we cannot easily do is create a variable that be discontinuous - a variable that could range from 100 to 50, or be zero.

We can create this variable by introducing a binary variable and two constraints:

```python
prob = pulp.LpProblem("minimize", pulp.LpMinimize)
var = pulp.LpVariable("var", 0, 100)
binary = pulp.LpVariable("binary", cat="Binary")
upper_limit = 100
lower_limit = 50

prob += var - upper_limit * binary <= 0
prob += lower_limit * binary - var <= 0
```

The table below shows the possible states of the continuous and binary variables, along with their feasibility when the binary variable is 1:

| Var | Binary | Upper Limit Constraint | Lower Limit Constraint | Feasible |
|-----|--------|------------------------|------------------------|----------|
| 150 | 1      | 150 - 100 * 1 = 50     | 50 * 1 - 150 = -100    | No       |
| 100 | 1      | 100 - 100 * 1 = 0      | 50 * 1 - 100 = -50     | Yes      |
| 75  | 1      | 75 - 100 * 1 = -25     | 50 * 1 - 75 = -25      | Yes      |
| 50  | 1      | 50 - 100 * 1 = -50     | 50 * 1 - 50 = 0        | Yes      |
| 25  | 1      | 25 - 100 * 1 = -75     | 50 * 1 - 25 = 25       | No       |

Table below does the same for when the binary variable is 0:

| Var | Binary | Upper Limit Constraint | Lower Limit Constraint | Feasible |
|-----|--------|------------------------|------------------------|----------|
| 150 | 0      | 150 - 100 * 0 = 150    | 50 * 0 - 150 = -150    | No       |
| 100 | 0      | 100 - 100 * 0 = 100    | 50 * 0 - 100 = -100    | No       |
| 75  | 0      | 75 - 100 * 0 = 75      | 50 * 0 - 75 = -75      | No       |
| 50  | 0      | 50 - 100 * 0 = 50      | 50 * 0 - 50 = -50      | No       |
| 25  | 0      | 25 - 100 * 0 = 25      | 50 * 0 - 25 = -25      | No       |

### Linking Two Continuous Variables Together

Linking the two continuous variables can be done with a simple sum constraint, that limits the sum of both binary variables to be 1:

```python
prob = pulp.LpProblem("minimize", pulp.LpMinimize)

a = pulp.LpVariable("a", 0, 100)
a_binary = pulp.LpVariable("a_binary", cat="Binary")
a_upper_limit = 100
a_lower_limit = 50

prob += a - a_upper_limit * a_binary <= 0
prob += a_lower_limit * a_binary - a <= 0

b = pulp.LpVariable("b", 0, 100)
b_binary = pulp.LpVariable("b_binary", cat="Binary")
b_upper_limit = 100
b_lower_limit = 50

prob += b - b_upper_limit * b_binary <= 0
prob += b_lower_limit * b_binary - b <= 0

prob += a_binary + b_binary == 1
```

# Full Code

Here is the final working code:

```python
import pulp


def run_linear_program(
    generation: float,
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
    site_import_limit: int = 100,
    site_export_limit: int = 100,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, None)
    site_export = pulp.LpVariable("site_export", 0, None)

    prob += site_import * import_price - site_export * export_price
    prob += site_import + generation == site_export + demand

    site_import_binary = pulp.LpVariable("site_import_binary", cat="Binary")
    site_export_binary = pulp.LpVariable("site_export_binary", cat="Binary")

    prob += site_import - site_import_limit * site_import_binary <= 0
    prob += site_export - site_export_limit * site_export_binary <= 0
    prob += site_import_binary + site_export_binary == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)


run_linear_program(25)
```

## Summary

This post introduced two linear programming tricks that allows us to avoid common problems when modelling energy systems as linear programs.  The tricks are:
1. Constrain the upper and lower bounds of continuous variables to both a non-zero lower bound and zero,
2. Linking two continuous variables together through binary variables.

These tricks allow us to avoid simultaneous import and export power in linear programs of energy systems.

---

Thanks for reading!

If you are interested in linear programming for energy systems, check out [energy-py-linear](https://energypylinear.adgefficiency.com/latest/).
