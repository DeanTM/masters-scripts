import numpy as np
import matplotlib.pyplot as plt


def dwdt_bcm(nu_j, nu_i, theta):
    return nu_i * (nu_i - theta) * nu_j / theta


tau_1=0.034
tau_2=0.014
F=-0.51
G=1.03
def C_x_izh_desai(nu_j, nu_i, tau_1=tau_1, tau_2=tau_2,F=F,G=G):
    return nu_i * nu_j * (F/(nu_i + 1/tau_1) + G/(nu_i + 1/tau_2))

if __name__ == '__main__':
    theta = 7.5**2
    nu_i = np.linspace(0, 100, 100)
    nu_j = 10.
    dw_vals = dwdt_bcm(nu_j, nu_i, theta)
    min_dw = np.min(dw_vals)

    fig = plt.figure(figsize=(4,3))
    plt.plot(nu_i, dw_vals, color='k', label='$dw/dt$')
    plt.vlines(theta, min_dw, -min_dw, color='red', ls='--', label=r'$\theta_{thr}$')
    plt.arrow(theta + 1., -0.8*min_dw, 4., 0., color='red', length_includes_head=False, width=7., head_length=1)
    plt.arrow(theta - 1., -0.8*min_dw, -4., 0., color='red', length_includes_head=False, width=7., head_length=1)
    plt.xticks([0.])
    plt.yticks([0.])
    ylabel = plt.ylabel(r"$\frac{dw}{dt}$")
    ylabel.set_rotation(0)
    plt.xlabel(r"$\nu_i$")
    plt.grid(ls=':', alpha=0.5)
    plt.legend()
    fig.tight_layout()
    fig.savefig('images_and_animations/bcm_curve.png')
    plt.show()

    fig = plt.figure(figsize=(4,3))
    nu_i = np.linspace(0, 40, 100)
    nu_j = 10.
    theta = -(G/tau_1 + F/tau_2) / (F + G)
    dw_vals = C_x_izh_desai(nu_j, nu_i)
    min_dw = np.min(dw_vals)
    plt.plot(nu_i, dw_vals, color='k', label=r'$\langle dw/dt\rangle $')
    plt.vlines(theta, min_dw, -min_dw, color='red', ls='--', label=r'$\theta_{thr}$')
    plt.xticks([0.])
    plt.yticks([0.])
    ylabel = plt.ylabel(r"$\left\langle\frac{dw}{dt}\right\rangle$")
    ylabel.set_rotation(0)
    plt.xlabel(r"$\nu_i$")
    plt.grid(ls=':', alpha=0.5)
    plt.legend()
    fig.tight_layout()
    fig.savefig('images_and_animations/izh_desai_curve.png')
    plt.show()