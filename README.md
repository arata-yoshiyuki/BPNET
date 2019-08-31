# BPNET

Bankruptcy Propagation on a Network (BPNET) implementation.

* BPNET is a C++ program to estimate the effect of bankruptcy propagation on an inter-firm transaction network.
* I applied this program to Japanese inter-firm transaction network data in [Arata (2018)](https://www.rieti.go.jp/jp/publications/dp/18e040.pdf) and uploaded it here for replication purpose.
* This program is still under construction and any comments are welcome.

## 1. Overview

* Firms are interrelated with each other via customer-supplier relationships, which generate a huge and complex network.
* Because of the relationships the bankruptcy of a firm may lead to the bankruptcy of its suppliers and/or customers (i.e., bankruptcy propagation).
* First, by using survival analysis, I estimate an increase in bankruptcy probability when its supplier or customer goes bankrupt.
* Second, based on the estimates, I simulate this model and quantify the effect of bankruptcy propagation on the network.

## 2. Parameter estimation

* Folder *Cascade_base_model* is for the point estimate.
* *Data.h* collects functions for data preprocessing.
* *Cascade.cpp* is the main file.
  * The log-likelihood for given parameter values is calculated.
  * The log-likelihood function is parallelized.
  * Nelder–Mead method is used to maximize the log-likelihood function.
* Folder *Cascade_base_model_Var* is for statistical inference, i.e., for the standard error.

## 3. Simulation

* Folder *Simulation* has two sub-folders *Full* and *Null*.
* *Full* is for simulation with parameter estimates obtained above.
* *Null* is for simulation for the case when the effect of bankruptcy propagation is *switched off*. It serves as a null model.

## 4. References

* Arata, Y. 2018. Bankruptcy propagation on a customer-supplier network: An empirical analysis in Japan. RIETI DP, 18-E-040.

<!-- $$\begin{aligned}
    &= \\
    &=
\end{aligned}
$$  -->
