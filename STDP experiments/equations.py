# This file contains equations for the Brian2 STDP experiments
# There is one neuron model we'll consider
neuron_model = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''
# # There are various synapse models for the different learning rules
# synapse_model_STDP = '''
#     w : 1
#     dApre/dt = -Apre / taupre : 1 (event-driven)
#     dApost/dt = -Apost / taupost : 1 (event-driven)
#     '''
# synapse_model_Triplet = '''
#     w : 1
#     dApre/dt = -Apre / taupre : 1 (event-driven)
#     dApost/dt = -Apost / taupost : 1 (event-driven)
#     dApost_slow/dt = -Apost_slow / taupost_slow : 1 (event-driven)
#     '''

# responsive post-synaptic neurons are induced to fire
on_pre_responsive = '''
    ge += w
    '''

# # # various learning rules have different on_pre and on_post behaviour
# # on_pre_STDP = '''
# #     Apre += dApre
# #     w = clip(w + Apost, 0, gmax)
# #     '''
# # on_post_STDP = '''
# #     Apost += dApost
# #     w = clip(w + Apre, 0, gmax)
# #     '''

# # # mu-STDP is the rule with (exponentially) scaled weight dependence
# # on_pre_muSTDP = '''
# #     Apre += dApre
# #     w = clip(w + Apost*(w**mu), 0, gmax)
# #     '''
# # on_post_muSTDP = '''
# #     Apost += dApost
# #     w = clip(w + Apre*(gmax-w)**mu, 0, gmax)
# #     '''

# # # W-STDP has the same potentiation dynamics as STDP
# # on_pre_WSTDP = '''
# #     Apre += dApre
# #     w = clip(w + Apost*w, 0, gmax)
# #     '''
# # on_post_WSTDP = on_post_STDP

# # we only consider the "minimal Triplet rule" where LTP is triplet-driven
# on_pre_Triplet = '''
#     Apre += dApre
#     w = clip(w + Apost, 0, gmax)
#     '''
# on_post_Triplet = '''
#     Apost += dApost
#     w = clip(w + Apre*Apost_slow, 0, gmax)
#     Apost_slow += dApost  # increment last
#     '''

# # mu-Triplet is to the mu-STDP Rule as Triplet is to the STDP rule
# on_pre_muTriplet = on_pre_muSTDP
# on_post_muTriplet = '''
#     Apost += dApost
#     w = clip(w + Apre*Apost_slow*(gmax-w)**mu, 0, gmax)
#     Apost_slow += dApost_slow
#     '''


######
# re-writing rules a la Gerstner et al.
######
# There are various synapse models for the different learning rules
synapse_model_STDP = '''
    w : 1
    dxj/dt = -xj / taupre : 1 (event-driven)
    dyi/dt = -yi / taupost : 1 (event-driven)
    '''
synapse_model_Triplet = '''
    w : 1
    dxj/dt = -xj / taupre : 1 (event-driven)
    dyi/dt = -yi / taupost : 1 (event-driven)
    dyi2/dt = -yi2 / taupost_slow : 1 (event-driven)
    '''

# various learning rules have different on_pre and on_post behaviour
on_pre_STDP = '''
    xj += 1
    w = clip(w + Aminus * yi, 0, gmax)
    '''
on_post_STDP = '''
    yi += 1
    w = clip(w + Aplus * xj, 0, gmax)
    '''

# mu-STDP is the rule with (exponentially) scaled weight dependence
on_pre_muSTDP = '''
    xj += 1
    w = clip(w + Aminus * yi * (w**mu), 0, gmax)
    '''
on_post_muSTDP = '''
    yi += 1
    w = clip(w + Aplus * xj * (gmax-w)**mu, 0, gmax)
    '''

# W-STDP has the same potentiation dynamics as STDP
on_pre_WSTDP = '''
    xj += 1
    w = clip(w + Aminus * yi * w, 0, gmax)
    '''
on_post_WSTDP = on_post_STDP

# we only consider the "minimal Triplet rule" where LTP is triplet-driven
on_pre_Triplet = '''
    xj += 1
    w = clip(w + Aminus * yi, 0, gmax)
    '''
on_post_Triplet = '''
    yi += 1
    w = clip(w + Aplus * xj * yi2, 0, gmax)
    yi2 += 1
    '''

# mu-Triplet is to the mu-STDP Rule as Triplet is to the STDP rule
on_pre_muTriplet = on_pre_muSTDP
on_post_muTriplet = '''
    yi += 1
    w = clip(w + Aplus * xj * yi2 * (gmax-w)**mu, 0, gmax)
    yi2 += 1
    '''
