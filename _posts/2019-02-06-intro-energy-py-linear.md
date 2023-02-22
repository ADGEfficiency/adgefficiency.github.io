---
title: Introducing energy-py-linear
date_created: 2019-02-06
date: 2023-01-30
date_updated: 2023-01-30
categories: 
- Energy
excerpt: A Python library for optimizing energy systems using mixed integer linear programming.

---

This post introduces [energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear) - a Python library for optimizing energy assets using mixed integer linear programming (MILP).

## Why Linear Programming?

Linear programming is a popular choice for solving many energy industry problems - many energy systems can be modelled as linear, and suitable for optimization using linear solvers.

Linear models have the quality that if a feasible solution exists, it exists on the boundary of a constraint.  This makes solving linear programs fast in practice. The optimization itself is also deterministic - it doesn't rely on randomness like gradient descent.

## What can `energypylinear` do?

1. optimize the dispatch of electric batteries, electric vehicle charging and gas fired CHP generators,
2. optimize for either price or carbon,
3. calculate the variance between two simulations.

You can find the source code for `energypylinear` at [ADGEfficiency/energy-py-linear](https://github.com/ADGEfficiency/energy-py-linear).
