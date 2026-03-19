from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from libmoonRAW.util.constant import get_agg_func, solution_eps, get_hv_ref
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV
import numpy as np
from libmoonRAW.util.gradient import get_moo_Jacobian_batch

import copy

class GradBaseSolver:
    def __init__(self, step_size, epoch, tol, core_solver):
        self.step_size = step_size
        self.epoch = epoch
        self.tol = tol
        self.core_solver = core_solver
        self.is_agg = (self.core_solver.core_name == 'AggCore')

    def solve(self, problem, x , prefs):
        '''
            :param problem:
            :param x:
            :param agg:
            :return:
                is a dict with keys: x, y.
        '''
        def marked_models_update(cur_losseses): # adaptive conversion
            for i in range(self.n_prob):
                cur_losses = cur_losseses[i]
                cur_sol = copy.deepcopy(xs_var).data
                compare_res = np.all((np.array(cur_losses) >= np.array(marked_losses[i])), axis=1) # if all True, not pareto
                row_ids = np.where(compare_res)[0]
                if len(row_ids) != 0: # current point not pareto optimal, assign it averagely
                    # self.marked_model_weights[row_ids[0]] += 1
                    for idx in row_ids:
                        marked_weights[i][idx] += 1 / len(row_ids)
                else:
                    compare_res = np.all((np.array(cur_losses) <= np.array(marked_losses[i])), axis=1)
                    remove_ids = np.where(compare_res)[0]
                    new_weight = 1
                    for idx in remove_ids[::-1]:
                        del marked_losses[i][idx]
                        del marked_solutions[i][idx]
                        new_weight += marked_weights[i][idx]
                        del marked_weights[i][idx]
                    marked_losses[i].append(cur_losses)
                    marked_solutions[i].append(cur_sol[i])
                    marked_weights[i].append(new_weight)
                assert sum(marked_weights[i]) - (epoch_idx + 1) < 1e-5, f"{sum(marked_weights[i])}Wrong total weights!"
                result_solution[i] = sum([weight * sol 
                                          for weight, sol in zip(marked_weights[i], marked_solutions[i])])
                result_solution[i] /= (epoch_idx + 1)

        def project(p):
            p = [p_i.detach().numpy() for p_i in p]
            u = sorted(p, reverse=True)
            res = []
            rho = 0
            for i in range(len(p)):
                if (u[i] + (1.0 / (i + 1)) * (1 - np.sum(np.asarray(u)[:i + 1]))) > 0:
                    rho = i + 1
            lmbd = (1.0 / rho) * (1 - np.sum(np.asarray(u)[:rho]))
            for i in range(len(p)):
                res.append(max(p[i] + lmbd, 0))
            res =  torch.from_numpy(np.array(res)) 
            return res

        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        xs_var = Variable(x, requires_grad=True)
        ########## New methods ##########
        assert torch.sum(prefs[1]) == 1., print(torch.sum(prefs[1]), "You may interpreted the size wrong!")
        dynamic_lambdas = torch.full(prefs.shape, 1 / self.n_obj)
        alpha_array = torch.full(prefs.shape, 1 / self.n_obj)
        marked_losses = [[[float('inf')] * self.n_obj] for _ in range(self.n_prob)]
        marked_solutions = [[copy.deepcopy(xs_var)] for _ in range(self.n_prob)]
        marked_weights = [[0] for _ in range(self.n_prob)]
        result_solution = copy.deepcopy(xs_var).data
        result_solution_loss = [float('inf') for _ in range(self.n_prob)]
        #################################
        optimizer = SGD([xs_var], lr=self.step_size)
        ind = HV(ref_point=get_hv_ref(problem.problem_name))
        hv_arr, y_arr = [], []
        for epoch_idx in tqdm(range(self.epoch)):
            fs_var = problem.evaluate(xs_var)
            y_np = fs_var.detach().numpy()
            y_arr.append(y_np)
            hv_y_np = y_np[~np.isnan(y_np).any(axis=1)]            
            hv_arr.append(ind.do(hv_y_np))
            Jacobian_array = get_moo_Jacobian_batch(xs_var, prefs * fs_var, self.n_obj)
            y_detach = fs_var.detach()
            optimizer.zero_grad()

            if self.is_agg:
                agg_name = self.core_solver.solver_name.split('_')[-1]
                agg_func = get_agg_func(agg_name, self.core_solver.scaler)
                ########## New methods ##########
                if 'omd' in agg_name:
                    agg_val, dynamic_lambdas = agg_func(fs_var, torch.Tensor(prefs).to(fs_var.device), dynamic_lambdas) 
                else:
                    agg_val = agg_func(fs_var, torch.Tensor(prefs).to(fs_var.device))
                #################################
                torch.sum(agg_val).backward()
            else:
                if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                    alpha_array = torch.stack([self.core_solver.get_alpha(Jacobian_array[idx], y_detach[idx], idx) for idx in range( self.n_prob) ])
                elif self.core_solver.core_name == 'CRMOGMCore':
                    new_alpha_array = torch.stack([self.core_solver.get_alpha(Jacobian_array[idx], y_detach[idx], idx) for idx in range( self.n_prob) ])
                    momentum_alpha = max(0., 1 - (epoch_idx + 1) * torch.sum(torch.abs(new_alpha_array - alpha_array)))
                    alpha_array = momentum_alpha * alpha_array + (1 - momentum_alpha) * new_alpha_array
                elif self.core_solver.core_name == 'MocoCore':
                    for i in range(alpha_array.size(0)):
                        alpha_array[i] = project(alpha_array[i] - self.core_solver.eta * 
                                                 (Jacobian_array[i] @ Jacobian_array[i].T + self.core_solver.pho * torch.eye(self.n_obj)) @ alpha_array[i])
                elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'HVGradCore']:
                    # assert False, 'Unknown core_name'
                    if self.core_solver.core_name == 'HVGradCore':
                        alpha_array = self.core_solver.get_alpha_array(y_detach)
                    elif self.core_solver.core_name == 'PMTLCore':
                        alpha_array = self.core_solver.get_alpha_array(Jacobian_array, y_np, epoch_idx)
                    elif self.core_solver.core_name == 'MOOSVGDCore':
                        alpha_array = self.core_solver.get_alpha_array(Jacobian_array, y_detach)
                    else:
                        assert False, 'Unknown core_name'
                elif self.core_solver.core_name == 'FEREROCore':
                    alpha_array = torch.stack([self.core_solver.get_d_pmol(Jacobian_array[idx], y_np[idx], prefs[idx])[0] for idx in range(self.n_prob)])
                elif self.core_solver.core_name == 'ExcessMTLCore':
                    alpha_array = torch.stack([self.core_solver.get_alpha(Jacobian_array[idx], idx) for idx in range(self.n_prob)])
                else:
                    assert False, 'Unknown core_name'

                if self.core_solver.core_name not in ['PMTLCore', 'EPOCore', 'FEREROCore']:
                    torch.sum(alpha_array * prefs * fs_var).backward()
                else:
                    torch.sum(alpha_array * fs_var).backward()

            ########## New methods ##########
            if self.is_agg and 'gomd' in self.core_solver.solver_name:
                marked_models_update(fs_var.detach().cpu().numpy())
            elif self.is_agg and 'omd' in self.core_solver.solver_name:
                result_solution = (result_solution * epoch_idx + xs_var) / (epoch_idx + 1)
            #################################
            optimizer.step()
            
            if 'lbound' in dir(problem):
                xs_var.data = torch.clamp(xs_var.data, torch.Tensor(problem.lbound) + solution_eps, # fix bug: change x to x_var
                                     torch.Tensor(problem.ubound) - solution_eps)
        res = {}
        ########## New methods ##########
        if self.is_agg and 'omd' in self.core_solver.solver_name:
            res['x'] = result_solution.detach().numpy()
            fs_var = problem.evaluate(result_solution)
            y_np = fs_var.detach().numpy()
            res['y'] = y_np
            hv_arr.append(ind.do(y_np))
            res['hv_arr'] = hv_arr
            res['y_arr'] = y_arr
        else:
            res['x'] = xs_var.detach().numpy()
            res['y'] = y_np
            res['hv_arr'] = hv_arr
            res['y_arr'] = y_arr
        #################################
        return res



class GradAggSolver(GradBaseSolver):
    def __init__(self, problem, step_size, epoch, tol, agg):
        self.agg = agg
        self.problem = problem
        super().__init__(step_size, epoch, tol)

    def solve(self, x, prefs):
        x = Variable(x, requires_grad=True)
        ind = HV(ref_point = get_hv_ref(self.problem.problem_name))
        hv_arr = []
        y_arr = []
        x_arr = []
        prefs = Tensor(prefs)
        optimizer = SGD([x], lr=self.step_size)
        agg_func = get_agg_func(self.agg)
        res = {}
        for i in tqdm(range(self.epoch)):
            y = self.problem.evaluate(x)
            hv_arr.append(ind.do(y.detach().numpy()))
            agg_val = agg_func(y, prefs)
            optimizer.zero_grad()
            torch.sum(agg_val).backward()
            optimizer.step()
            y_arr.append(y.detach().numpy())
            if 'lbound' in dir(self.problem):
                x.data = torch.clamp(x.data, torch.Tensor(self.problem.lbound) + solution_eps, torch.Tensor(self.problem.ubound)-solution_eps)


        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_history'] = np.array(hv_arr)
        res['y_history'] = np.array(y_arr)
        res['x_history'] = np.array(y_arr)
        return res