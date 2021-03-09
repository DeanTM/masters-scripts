from os import listdir
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if 'wong2006.py' not in listdir():
        import requests
        resp = requests.get("https://raw.githubusercontent.com/xjwanglab/book/master/wong2006/wong2006.py")
        filecontent = resp.text.replace('xrange', 'range')
        with open('wong2006.py', 'w') as f:
            f.write(filecontent)
    from wong2006 import modelparams, Model
    
    wong2006model = Model(modelparams)
    wong2006model.run(coh=10, n_trial=2)
    
    fig, axes = plt.subplots(2,1,figsize=(8,6), sharex=True)
    axes[0].plot(wong2006model.t, wong2006model.I1, 'black')
    axes[0].plot(wong2006model.t, wong2006model.I2, 'red')
    axes[0].set_ylabel('input (nA)')
    axes[1].plot(wong2006model.t, wong2006model.r1, color='black', alpha=0.6)
    axes[1].plot(wong2006model.t, wong2006model.r2, color='red', alpha=0.6)
#     axes[1].plot(wong2006model.t, wong2006model.r1[:,0], color='black')
#     axes[1].plot(wong2006model.t, wong2006model.r2[:,0], color='red')
#     axes[1].plot(wong2006model.t, wong2006model.r1[:,1], color='black', ls=':')
#     axes[1].plot(wong2006model.t, wong2006model.r2[:,1], color='red', ls=':')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('activity (Hz)')
    fig.tight_layout()
    fig.savefig('images_and_animations/wongplot_trajectory.png')
    
    fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)
    for k, coh in enumerate([1, 3, 5, 15]):
        wong2006model.run(coh=coh, n_trial=10)
        axes[int(k>=2),k%2].plot(
            wong2006model.r1, wong2006model.r2,
            color='black', alpha=0.1)
        axes[int(k>=2),k%2].set_title(f"Coherence: {coh}%")
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Rate 1 (Hz)")
    plt.ylabel("Rate 2 (Hz)")
#     fig.text(0.5, 0.04, 'Rate 1 (Hz)', ha='center')
#     fig.text(0.04, 0.5, 'Rate 2 (Hz)', va='center', rotation='vertical')
    fig.tight_layout()
    fig.savefig('images_and_animations/wongplot_statespace.png')
    plt.show()
