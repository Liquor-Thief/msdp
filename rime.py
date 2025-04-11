import numpy as np
def initialization(pop_size, dim, lb, ub):
    lb = np.array(lb)
    ub = np.array(ub)
    if lb.size == 1 and ub.size == 1:
        pop = np.random.rand(pop_size, dim) * (ub - lb) + lb
    else:
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            pop[:, d] = np.random.rand(pop_size) * (ub[d] - lb[d]) + lb[d]
    return pop

def RIME(N, Max_iter, lb, ub, dim, fobj):
    Best_rime = np.zeros(dim)
    Best_rime_rate = np.inf
    Rimepop = initialization(N, dim, ub, lb)
    Lb = np.array(lb)
    Ub = np.array(ub)
    it = 1
    Convergence_curve = np.zeros(Max_iter)
    Rime_rates = fobj(Rimepop)

    for i in range(N):
        if Rime_rates[i] < Best_rime_rate:
            Best_rime_rate = Rime_rates[i]
            Best_rime = Rimepop[i, :]

    while it <= Max_iter:
        RimeFactor = (np.random.rand() - 0.5) * 2 * np.cos((np.pi * it / (Max_iter / 10))) * (1 - round(it * 5 / Max_iter) / 5)
        E = (it / Max_iter) ** 0.5
        newRimepop = np.copy(Rimepop)
        normalized_rime_rates = Rime_rates / np.linalg.norm(Rime_rates)

        for i in range(N):
            for j in range(dim):
                r1 = np.random.rand()
                if r1 < E:
                    newRimepop[i, j] = Best_rime[j] + RimeFactor * ((Ub[j] - Lb[j]) * np.random.rand() + Lb[j])

                r2 = np.random.rand()
                if r2 < normalized_rime_rates[i]:
                    newRimepop[i, j] = Best_rime[j]

        for i in range(N):
            Flag4ub = newRimepop[i, :] > ub
            Flag4lb = newRimepop[i, :] < lb
            newRimepop[i, :] = newRimepop[i, :] * ~(Flag4ub + Flag4lb) + ub * Flag4ub + lb * Flag4lb
            newRime_rates = np.min(fobj(newRimepop))

            if newRime_rates < Rime_rates[i]:
                Rime_rates[i] = newRime_rates
                Rimepop[i, :] = newRimepop[i, :]
                if newRime_rates < Best_rime_rate:
                    Best_rime_rate = newRime_rates
                    Best_rime = Rimepop[i, :]

        Convergence_curve[it - 1] = Best_rime_rate
        it += 1
    return Best_rime_rate, Best_rime, Convergence_curve
