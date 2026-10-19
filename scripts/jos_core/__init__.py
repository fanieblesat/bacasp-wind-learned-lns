"""jos_core — executable form of the corrected BACASP-under-wind model (Section 3)
and the learned search control of Section 4.

Modules
  params      model constants (W1, Wlim, alpha, phi, rho, Omega) — ONE place
  instances   vessel generator, wind (OU) generator with berth exposure
  simulator   Solution representation + period-indexed evaluation implementing
              Eqs (moor_def)..(completion); checker for MIP decision variables
  search      initial solution, six operators, classical baselines
  controller  LNS environment (54 actions, 16 features), DQN, training, variants
"""
