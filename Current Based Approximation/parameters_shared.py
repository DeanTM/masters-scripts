N = 1000
N_E = int(N * 0.8)  # pyramidal neurons
N_I = int(N * 0.2)
f = 0.1  # fractions of neurons in each selective population
p = 5  # number of selective populations
N_sub = int(N_E * f)
N_non = N_E - p*N_sub
w_plus = 2.1
w_minus = 1.0 - f*(w_plus - 1.0) / (1.0 - f)
C_ext = 800
C_E = N_E
C_I = N_I
rate_ext = 3.

runtime = 0.4

parameters_shared_dict = dict(
    N=N,
    N_E=N_E,
    N_I=N_I,
    f=f,
    p=p,
    N_sub=N_sub,
    N_non=N_non,
    w_plus=w_plus,
    w_minus=w_minus,
    C_ext=C_ext,
    C_E=C_E,
    C_I=C_I,
    rate_ext=rate_ext,
    runtime=runtime
)