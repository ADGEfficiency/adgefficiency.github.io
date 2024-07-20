# Constraining Simultaneous Import and Export Power or Battery Charge and Discharge

## Context

When modelling energy systems, we are often modelling assets where electiricyt can flow in opposite directions.  Examples include:

- a site that can import or export electricity,
- a battery that can charge or discharge electricity.

For these cases, electricity cannot flow in opposite directions at the same time.  We want this 

This post explains how to constrain the simultaneous flow of electricity to only go in one direction in a linear program.

TODO - context on LP in energy

## A Simple Site with Import, Export, Demand and Generation

To demonstrate this, we will setup a simple example of a site with:

- a electric load,
- a generator,
- a grid connection that can import and export.

The code below sets this scenario as a linear program using the PuLP Python library:

```python
import pulp

def run_linear_program(
    electric_generation: float,
    demand: float = 50
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, None)
    site_export = pulp.LpVariable("site_export", 0, None)

    prob += site_import + generation == site_export + demand
    prob.solve()

    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)

run_linear_program(25)
```

There are a few things to note:

- The program above has a single constraint - it doesn't have an objective function,
- the upper bounds on `site_import` and `site_export` are set to 100 - if the upper bound is not set, the program can become infeasible.

We can check this works correctly by changing the on site generation and seeing how the import and export change:

```python
import pulp

for generation in [0, 25, 50, 100]:
    print(run_linear_program(generation))
```

## Simultaneous Import and Export

To induce simultaneous import and export, we can setup a scenario where import prices are lower than export prices.

This will require adding an objective function to our program:

```python
import pulp

def run_linear_program(
    electric_generation: float,
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, 100)
    site_export = pulp.LpVariable("site_export", 0, 100)

    prob += site_import + generation == site_export + demand
    prob += site_import * import_price - site_export * export_price
    prob.solve()

    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)

run_linear_program(25)
```

## Stopping Simultaneous Import and Export

To prevent simultaneous import and export, we need to turn our linear program into a mixed integer linear program. 

We will introduce a binary variable that will allow us to stop simultaneous import and export.

```python
import pulp

def run_linear_program(
    electric_generation: float,
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
    site_import_limit: int =100,
    site_export_limit: int =100
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

    prob.solve()
    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)

run_linear_program(25)
```

Now we see that our 

## Explaining the Trick

Constrain the upper bound on `site_import` and `site_export` 

Link two continuous variables together through two binary variables.

## Other Use Cases for this Trick

Constraining simultaneous battery charge and discharge

Constraining electric generation to a non-zero lower bound or zero.
