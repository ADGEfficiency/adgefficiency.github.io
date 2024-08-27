
## Constraining Non-Zero Upper and Lower Bounds of Generators

### Constraining Upper and Lower Bounds

Let's demonstrate this with a simple example.  We will setup a site with a generator and demand:

```python
import pulp


def run_linear_program(
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
    site_import_limit: int = 100,
    site_export_limit: int = 100,
    generator_upper_limit: int = 100,
    generator_cost: float = 100,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, None)
    site_export = pulp.LpVariable("site_export", 0, None)
    generation = pulp.LpVariable("generation", 0, generator_upper_limit)

    prob += (
        site_import * import_price
        - site_export * export_price
        + generation * generator_cost
    )
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


run_linear_program(demand=10, import_price=100, export_price=0)
```

Above our generator has been dispatched at `10`.  In the real world, generators have minimum outputs - often around 50% of their full load.

We already have our upper limit constrained by an upper bound on the `generation` variable. We can introduce a lower bound in the same way, but it won't work as we want:


```python
import pulp


def run_linear_program(
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
    site_import_limit: int = 100,
    site_export_limit: int = 100,
    generator_upper_limit: int = 100,
    generator_lower_limit: int = 50,
    generator_cost: float = 100,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, None)
    site_export = pulp.LpVariable("site_export", 0, None)
    generation = pulp.LpVariable(
        "generation", generator_lower_limit, generator_upper_limit
    )

    prob += (
        site_import * import_price
        - site_export * export_price
        + generation * generator_cost
    )
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


run_linear_program(demand=10, import_price=100, export_price=0)
```

Now we see that our generator is forced to run - it's not able to run lower than `50`.

Even in a scenario where import power is free, the generator will run:

```python
run_linear_program(demand=10, import_price=0, export_price=0)
```

We can fix this by introducing a similar looking constraint

```python
import pulp


def run_linear_program(
    demand: float = 50,
    import_price: float = 50,
    export_price: float = 500,
    site_import_limit: int = 100,
    site_export_limit: int = 100,
    generator_upper_limit: int = 100,
    generator_lower_limit: int = 50,
    generator_cost: float = 100,
) -> None:
    prob = pulp.LpProblem("minimize", pulp.LpMinimize)

    site_import = pulp.LpVariable("site_import", 0, None)
    site_export = pulp.LpVariable("site_export", 0, None)
    generation = pulp.LpVariable("generation", 0, generator_upper_limit)

    prob += (
        site_import * import_price
        - site_export * export_price
        + generation * generator_cost
    )
    prob += site_import + generation == site_export + demand

    site_import_binary = pulp.LpVariable("site_import_binary", cat="Binary")
    site_export_binary = pulp.LpVariable("site_export_binary", cat="Binary")
    generator_binary = pulp.LpVariable("generator_binary", cat="Binary")

    prob += site_import - site_import_limit * site_import_binary <= 0
    prob += site_export - site_export_limit * site_export_binary <= 0
    prob += site_import_binary + site_export_binary == 1

    prob += generator_binary * generator_lower_limit - generation <= 0
    prob += generation - generator_upper_limit * generator_binary <= 0

    prob.solve()
    print(f"status = {pulp.LpStatus[prob.status]}")
    for v in prob.variables():
        print(v.name, "=", v.varValue)


run_linear_program(demand=10, import_price=1000, export_price=0)
```

# Full Code

Here is all the code:

```python

```

## Summary

This post introduced a simple linear programming trick that allows us to avoid common problems when modelling energy systems as linear programs.

This trick allows us to:

- avoid simultaneous import and export power,
- constrain generators to have non-zero lower bounds,
- avoid simultaneous battery charge and discharge.

---

Thanks for reading!

If you are interested in linear programming for energy systems, check out [energy-py-linear](https://energypylinear.adgefficiency.com/latest/).
