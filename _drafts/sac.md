# DEBUG NOTES

## 15/10/20

Large Q func values - add a reward scale


## 17/10/20

First thing was looking at transitions - we can see that the obs isn't being reset after each step

2020-10-17 10:27:05,962 - climatedb - DEBUG - [[-0.46744808 -0.88402053 -0.40620497]], [[-0.4119937]], [-4.2492437], [[-0.5193096 -0.8545862 -1.1928185]]
2020-10-17 10:27:05,963 - climatedb - DEBUG - [[-0.46744808 -0.88402053 -0.40620497]], [[-0.8316113]], [-4.6260557], [[-0.6053495  -0.79595983 -2.0832415 ]]
2020-10-17 10:27:05,964 - climatedb - DEBUG - [[-0.46744808 -0.88402053 -0.40620497]], [[0.6828145]], [-5.3687005], [[-0.6989821  -0.71513915 -2.475367  ]]
2020-10-17 10:27:05,964 - climatedb - DEBUG - [[-0.46744808 -0.88402053 -0.40620497]], [[0.36186516]], [-6.111212], [[-0.79507494 -0.6065112  -2.903162  ]]
2020-10-17 10:27:05,965 - climatedb - DEBUG - [[-0.46744808 -0.88402053 -0.40620497]], [[0.33263853]], [-7.0430164], [[-0.88291925 -0.46952483 -3.2582538 ]]

added obs = next obs
## 17-01

Look at Q1 / Q2 loss

had

for ep in episodes:
 ep()
 update_pol()
 update_q()

update_target_net()

added writing of target net weights

## 17-02

policy loss much larger than qfunc loss

rescale of the log prob loss to 1.0, added log prob to tensorborad

## 17-03

first signs of life
- now to speed up the learning with a smaller net

## 17-04

looks great

changes for next run:

- increase num ep to 15000
- increase to 5 updates per epsiode


## 18-04

looks great - on to the next env

## ??

See old tensorboards

## 01-12

Found mistake with log standard deviation clipping

# NOTES

Learn three functions
- pi, Q1, Q2

Similar to TD3

- MSBE minimization (regress both Q func to shared target)
- shared target computed using target nets (two???) & clipped double Q trick
- target nets computed by polyak averaging the Q parameters

Dobule Q trick
- using minimum q value between Q1, Q2

Different from TD3

- entropy regularization
- next state actions come from current policy (not from target policy, not from experience!)
- target policy smoothing is not explicit - noise comes from stochastic policy

$a bar prime$ = next action sampled fresh

MSBE = mean square bellman error

Policy optimization
- max(V) = Q - alpha * log pi
- uses repamaretrization trick (take expectation over action -> expectation over noise)
- squashed (tanh) Gaussian policy
- parametrize both mean & stdev

Test time = use mean action (instead of sampling)


## entropy regularization 

Tradeoff between performance & policy randomness (for exploration)

Changes the optimal policy

- alpha = tradeoff coeff (controls exploration)
- can be fixed or 'entropy constrained'

Changes the value func & Q func

- V includes entropy bonus from all timesteps
- Q func includes entropy bonus from all timesteps except first (but this can be diffeerent)

Can define the value func in terms of Q & H

Can define the Q func in terms of V & reward
