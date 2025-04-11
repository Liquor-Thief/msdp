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

def rime_step(pop, fitness, best_sol, best_fit, lb, ub, iteration, max_iter,fobj):
    pop_size, dim = pop.shape
    lb = np.array(lb)
    ub = np.array(ub)
    p_soft = np.sqrt(iteration / (max_iter + 1e-9))

    max_f = np.max(fitness)
    if max_f < 1e-9:
        max_f = 1.0
    fit_norm = 1.0 - fitness / max_f

    rime_factor = (np.random.rand() - 0.5) * 2.0

    new_pop = pop.copy()
    for i in range(pop_size):
        if np.random.rand() < p_soft:
            for d in range(dim):
                new_pop[i, d] = best_sol[d] + rime_factor * (ub[d] - lb[d]) * np.random.rand()

        if np.random.rand() < fit_norm[i]:
            alpha = (iteration / max_iter)
            new_pop[i] = new_pop[i] + alpha * (best_sol - new_pop[i]) * np.random.rand()
        new_pop[i] = np.clip(new_pop[i], lb, ub)

    new_fit = np.array([fobj(ind) for ind in new_pop])
    for i in range(pop_size):
        if new_fit[i] < fitness[i]:
            pop[i] = new_pop[i]
            fitness[i] = new_fit[i]
            if fitness[i] < best_fit:
                best_fit = fitness[i]
                best_sol = pop[i].copy()

    return pop, fitness, best_sol, best_fit


def acpo_step(pop, fitness, best_sol, best_fit,
              lb, ub,iteration, max_iter,prev_best_fit, gamma,Nmin,fobj):
    pop_size, dim = pop.shape
    lb = np.array(lb)
    ub = np.array(ub)
    delta_f = abs(best_fit - prev_best_fit)
    # μ(t)
    mu_t = np.exp(-gamma * delta_f)

    new_size = int(Nmin + (pop_size - Nmin)*mu_t)
    if new_size < Nmin:
        new_size = Nmin
    if new_size > pop_size:
        new_size = pop_size
    if new_size < pop_size:
        indices = np.random.choice(pop_size, new_size, replace=False)
        pop = pop[indices]
        fitness = fitness[indices]
        idx_best = np.argmin(fitness)
        best_fit = fitness[idx_best]
        best_sol = pop[idx_best].copy()
    new_pop_size = pop.shape[0]
    for i in range(new_pop_size):
        r = np.random.rand()
        if r < 0.25:
            r_idx = np.random.randint(new_pop_size)
            predator_pos = pop[r_idx]
            factor = np.random.randn()
            new_pos = pop[i] + factor * np.abs(2.0*np.random.rand()*best_sol - predator_pos)
        elif r < 0.50:
            r1, r2 = np.random.randint(new_pop_size, size=2)
            U1 = np.random.rand(dim)
            y = (pop[i] + pop[r1]) / 2.0
            new_pos = (1 - U1)*pop[i] + U1*(y + np.random.rand()*(pop[r1] - pop[r2]))
        elif r < 0.75:
            r1, r2, r3 = np.random.randint(new_pop_size, size=3)
            St_i = np.exp(fitness[i] / (np.sum(fitness)+1e-9))
            new_pos = pop[r1] + St_i*(pop[r2] - pop[r3])
        else:
            alpha = 0.2
            r4 = np.random.rand()
            gamma_t = 2.0*(1 - iteration/(max_iter+1e-9))
            Fi = np.random.rand(dim)*(best_sol - pop[i])
            new_pos = best_sol + (alpha*(1-r4)+r4)*(np.random.rand(dim)*best_sol - pop[i]) - np.random.rand()*gamma_t*Fi

        new_pos = np.clip(new_pos, lb, ub)
        new_fit = fobj(new_pos)
        if new_fit < fitness[i]:
            pop[i] = new_pos
            fitness[i] = new_fit
            if new_fit < best_fit:
                best_fit = new_fit
                best_sol = new_pos.copy()

    return pop, fitness, best_sol, best_fit


def rime_acpo_opt(pop_size, max_iter, dim, lb, ub, gamma, Nmin,fobj):
    if isinstance(lb, (int, float)):
        lb = [lb] * dim
    if isinstance(ub, (int, float)):
        ub = [ub] * dim
    pop = initialization(pop_size, dim, lb, ub)
    fitness = np.array([fobj(ind) for ind in pop])
    best_fit = np.min(fitness)
    best_sol = pop[np.argmin(fitness)].copy()

    prev_best_fit = best_fit

    for iteration in range(1, max_iter+1):
        pop, fitness, best_sol, best_fit = rime_step(
            pop, fitness, best_sol, best_fit, lb, ub,
            iteration, max_iter,fobj
        )
        pop, fitness, best_sol, best_fit = acpo_step(
            pop, fitness, best_sol, best_fit,
            lb, ub,
            iteration, max_iter,
            prev_best_fit,
            gamma=gamma,
            Nmin=Nmin,fobj=fobj
        )

        prev_tmp = best_fit

        prev_best_fit = prev_tmp
    return best_fit, best_sol
