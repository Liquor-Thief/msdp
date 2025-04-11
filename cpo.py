import numpy as np
def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)
    Positions = np.zeros((SearchAgents_no, dim))
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions

def CPO(Pop_size, Tmax, lb, ub, dim, fobj):
    ub = np.array(ub)
    lb = np.array(lb)

    X = initialization(Pop_size, dim, ub, lb)  # Initialize the positions of search agents
    t = 0
    fitness = np.array([fobj(X[i, :]) for i in range(Pop_size)])
    Gb_Fit = np.min(fitness)
    Gb_Sol = X[np.argmin(fitness), :]

    Xp = X.copy()
    while t < Tmax:
        for i in range(Pop_size):
            U1 = np.random.rand(dim) > np.random.rand()
            if np.random.rand() < np.random.rand():
                if np.random.rand() < np.random.rand():
                    y = (X[i, :] + X[np.random.randint(Pop_size), :]) / 2
                    X[i, :] = X[i, :] + (np.random.randn() * abs(2 * np.random.rand() * Gb_Sol - y))
                else:
                    y = (X[i, :] + X[np.random.randint(Pop_size), :]) / 2
                    X[i, :] = (U1 * X[i, :]) + ((1 - U1) * (y + np.random.rand() * (
                                X[np.random.randint(Pop_size), :] - X[np.random.randint(Pop_size), :])))
            else:
                Yt = 2 * np.random.rand() * (1 - t / Tmax) ** (t / Tmax)
                U2 = (np.random.rand(dim) < 0.5) * 2 - 1
                S = np.random.rand() * U2
                if np.random.rand() < 0.8:
                    St = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    S = S * Yt * St
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (X[np.random.randint(Pop_size), :] + St * (
                                X[np.random.randint(Pop_size), :] - X[np.random.randint(Pop_size), :]) - S)
                else:
                    Mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    vt = X[i, :]
                    Vtp = X[np.random.randint(Pop_size), :]
                    Ft = np.random.rand(dim) * (Mt * (-vt + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = Gb_Sol + 0.2 * (1 - np.random.rand()) + np.random.rand() * (U2 * Gb_Sol - X[i, :]) - S

            X[i, :] = np.clip(X[i, :], lb, ub)
            nF = fobj(X[i, :])
            if fitness[i] < nF:
                X[i, :] = Xp[i, :]
            else:
                Xp[i, :] = X[i, :]
                fitness[i] = nF
                if fitness[i] <= Gb_Fit:
                    Gb_Sol = X[i, :]
                    Gb_Fit = fitness[i]

        t += 1

    return Gb_Fit, Gb_Sol