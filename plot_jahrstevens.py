from functions import *
import matplotlib.pyplot as plt

V = np.linspace(-70e-3, -20e-3, 100)
V_ = np.array([-55e-3])

g_V = (V - V_E) / J(V)
g_V_linearised = (V_ - V_E) / J(V_) + (V - V_)*J_2(V_)

J_2V = J_2(V)
zero_idx = np.argmin(np.abs(J_2V))
zero_V = V[zero_idx]

print(f"J_2 zero near {zero_V*1000:.2f}")
min_g = np.min(g_V)
max_g = np.max(g_V)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(V*1000, g_V*1000, label='exact') #, linewidth=2.5)
axes[0].plot(V*1000, g_V_linearised*1000, label='Linearised',ls=':') # linewidth=2.5, )

axes[1].plot(V*1000, J_2V)#, linewidth=2.5)
axes[1].plot([zero_V*1000], [0.], 'ko')


axes[0].axvline(V_thr*1000, color='k', linestyle=':', label=r'$V_{thr/reset}$')
axes[0].axvline(V_reset*1000, color='k', linestyle=':')
axes[0].set_xlabel('$V$ (mV)')
axes[1].set_xlabel('$V$ (mV)')
axes[0].set_ylabel(r'$(V-V_E)g_{NMDA}(V)$ (mV)')
axes[1].set_yticks([0.])
axes[0].set_title("Jahr-Stevens Approximation")
axes[1].set_title("$J_2(V)$")
axes[0].legend()
axes[0].grid(ls=':', alpha=.3)
axes[1].grid(ls=':', alpha=.3)
fig.tight_layout()
plt.savefig('images_and_animations/jahrstevens.png')
plt.show()