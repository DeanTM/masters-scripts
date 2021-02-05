N = 1000
N_E = int(N * 0.8)  # pyramidal neurons
N_I = int(N * 0.2)
f = 0.1  # fractions of neurons in each selective population
p = 2  # number of selective populations
N_sub = int(N_E * f)
N_non = N_E - p*N_sub
w_plus = 2.1
w_minus = 1.0 - f*(w_plus - 1.0) / (1.0 - f)
C_ext = 800
C_E = N_E
C_I = N_I

runtime = 0.4