import argparse
import numpy as np
import torch
from libmoonRAW.util import synthetic_init, get_problem, get_uniform_pref
from libmoonRAW.solver.gradient.methods.base_solver import GradBaseSolver
from libmoonRAW.solver.gradient.methods.core.core_solver import EPOCore, MGDAUBCore, CRMOGMCore, MocoCore, RandomCore, AggCore, MOOSVGDCore, HVGradCore, PMTLCore, FEREROCore, ExcessMTLCore
from libmoonRAW.solver.gradient.methods.core.core_solver import PMGDACore

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt
from libmoonRAW.util.constant import FONT_SIZE_2D, FONT_SIZE_3D, color_arr, beautiful_dict, root_name, min_key_array
from libmoonRAW.util.constant import plt_2d_tickle_size, plt_2d_label_size

import seaborn as sns
sns.set_theme(style='whitegrid')


def draw_2d_prefs(prefs, rho):
    prefs_norm2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    for idx, pref in enumerate(prefs_norm2):
        plt.plot([0, pref[0]*rho], [0, pref[1]*rho], color='grey', linewidth=2,
                 linestyle='--', zorder=-1)


def plot_figure_2d(problem):
    fig, ax = plt.subplots()
    if problem.problem_name[0] == 'F':
        ax.set_xlim(-0.05, 1.55)
        ax.set_ylim(-0.05, 1.55)

    if hasattr(problem, 'get_pf'):
        pf = problem.get_pf(n_pareto_points=1000)
        plt.plot(pf[:, 0], pf[:, 1], color='gray', linewidth=2, label='True PF', zorder=-1)
        if problem.problem_name[0] == 'F':
            plt.fill_between(np.concatenate((pf[:, 0], np.linspace(pf[-1,0], plt.xlim()[1], num=100)[1:])), 
                             np.concatenate((pf[:, 1], np.full(99, pf[-1,1]))),
                             y2=plt.ylim()[1], color='lightgray', alpha=0.5, label='Feasible Region', zorder=-2)
        else:
            plt.fill_between(np.concatenate((np.linspace(plt.xlim()[1], pf[0,0], num=100)[:-1], pf[:, 0])), 
                             np.concatenate((np.full(99, pf[0, 1]), pf[:, 1])),
                             y2=plt.ylim()[1], color='lightgray', alpha=0.5, label='Feasible Region', zorder=-2)
        
    y_arr = res['y']
    if problem.problem_name[0] == 'F':
        rho = 1.5
    else:
        # rho = np.max([np.linalg.norm(y) for y in y_arr])
        rho = 1.05
    draw_2d_prefs(prefs, rho)

    colors = {'agg_ls': 'r', 'agg_tche': 'k', 'agg_softtche': 'brown',
              'agg_omdgdtche': 'b', 'agg_gomdgdtche': 'c',
              'agg_omdegtche': 'm', 'agg_gomdegtche': 'plum',
              'gm_mgda': 'orange', 'gm_crmogm': 'y', 'gm_moco': 'gold',
              'pmtl': 'green', 'epo': 'lawngreen', 'ferero': 'springgreen',
              'excessmtl': 'aquamarine'}
    plt.scatter(y_arr[:, 0], y_arr[:, 1], s=75, color=colors[args.solver_name], zorder=1)
    
    ax.set_xlim(-0.05, 1.6 if problem.problem_name[0] == 'F' else 1.05)
    ax.set_ylim(-0.05, 1.6 if problem.problem_name[0] == 'F' else 1.05)
    ax.set_aspect('equal')
    ax.grid(False)
    plt.xticks(fontsize=plt_2d_tickle_size)
    plt.yticks(fontsize=plt_2d_tickle_size)
    plt.xlabel('$f_1$', fontsize=plt_2d_label_size)
    plt.ylabel('$f_2$', fontsize=plt_2d_label_size)
    plt.legend(fontsize=12)


def plot_figure_3d(folder_name):
    sub_sample = 1
    ax = (plt.figure()).add_subplot(projection='3d')
    for idx in range(len(prefs)):
        ax.plot(res['y_history'][::sub_sample, idx, 0], res['y_history'][::sub_sample, idx, 1],
                res['y_history'][::sub_sample, idx, 2],
                color=color_arr[idx])
    prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    for idx, pref in enumerate(prefs_l2):
        ax.scatter(pref[0], pref[1], pref[2], color=color_arr[idx], s=40)
    th1 = np.linspace(0, np.pi / 2, 100)
    th2 = np.linspace(0, np.pi / 2, 100)

    theta, phi = np.meshgrid(th1, th2)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    ax.plot_surface(x, y, z, alpha=0.3)
    ax.axis('equal')
    ax.view_init(30, 45)
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.2])
    ax.set_zlim([0, 1.2])
    ax.set_xlabel('$L_1$', fontsize=FONT_SIZE_3D)
    ax.set_ylabel('$L_2$', fontsize=FONT_SIZE_3D)
    ax.set_zlabel('$L_3$', fontsize=FONT_SIZE_3D)

def save_figures(folder_name):
    # os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join(folder_name, 'res.pdf')
    plt.savefig(fig_name, bbox_inches='tight')
    fig_name_svg = os.path.join(folder_name, 'res.svg')
    plt.savefig(fig_name_svg, bbox_inches='tight')
    print('Save fig to {}'.format(fig_name))
    print('Save fig to {}'.format(fig_name_svg))
    plt.title(beautiful_dict[args.solver_name])
    plt.close()

def save_pickles(folder_name):
    import pickle
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script')
    # mgdaub random epo pmgda agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche pmtl hvgrad moosvgd
    parser.add_argument('--solver-name', type=str, default='agg_cosmos')
    parser.add_argument( '--problem-name', type=str, default='VLMOP1')
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--draw-fig', type=str, default='True')
    parser.add_argument('--n-prob', type=int, default=10 )
    parser.add_argument('--epoch', type=int, default=1000 )
    parser.add_argument('--seed-idx', type=int, default=1)

    parser.add_argument('--seeds', type=str, default="")

    parser.add_argument('--mu', type=float, default=0.1) # soft tche
    parser.add_argument('--eta', type=float, default=0.) # omd tche, moco, excessmtl
    parser.add_argument('--pho', type=float, default=0.1) # moco

    args = parser.parse_args()

    seeds = list(map(int, args.seeds.split(',')))

    avg_res_x = np.zeros((args.n_prob, 10 if args.problem_name=='VLMOP2' else 6))

    for seed in seeds:
        args.seed_idx = seed
        np.random.seed(args.seed_idx)
        torch.manual_seed(args.seed_idx)
        print('Synthetic discrete')
        print('Running {} on {} with seed {}'.format(args.solver_name, args.problem_name, args.seed_idx) )
        problem = get_problem(problem_name=args.problem_name, n_var=10 if args.problem_name=='VLMOP2' else 6)
        prefs = get_uniform_pref(n_prob=args.n_prob, n_obj = problem.n_obj, clip_eps=1e-2)

        # Actually a bit waste to implement so many solvers. Just import Core solvers.
        if args.solver_name == 'epo':
            core_solver = EPOCore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name == 'gm_mgda':
            core_solver = MGDAUBCore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name == 'gm_crmogm':
            core_solver = CRMOGMCore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name == 'gm_moco':
            core_solver = MocoCore(n_var=problem.n_var, prefs=prefs, eta=args.eta, pho=args.pho)
        elif args.solver_name == 'random':
            core_solver = RandomCore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name == 'pmgda':
            core_solver = PMGDACore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name.startswith('agg'):
            core_solver = AggCore(n_var=problem.n_var, prefs=prefs, solver_name=args.solver_name, mu=args.mu, eta=args.eta)
        elif args.solver_name == 'moosvgd':
            core_solver = MOOSVGDCore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name == 'hvgrad':
            core_solver = HVGradCore(n_obj=problem.n_obj, n_var=problem.n_var, problem_name=problem.problem_name)
        elif args.solver_name == 'pmtl':
            core_solver = PMTLCore(n_obj=problem.n_obj, n_var=problem.n_var, total_epoch=args.epoch, warmup_epoch=args.epoch // 5, prefs=prefs)
        elif args.solver_name == 'ferero':
            core_solver = FEREROCore(n_var=problem.n_var, prefs=prefs)
        elif args.solver_name == 'excessmtl':
            core_solver = ExcessMTLCore(n_var=problem.n_var, prefs=prefs, llr=args.eta)
        else:
            assert False, 'Unknown solver'

        solver = GradBaseSolver(step_size=args.step_size, epoch=args.epoch, tol=args.tol, core_solver=core_solver)
        res = solver.solve(problem=problem, x=synthetic_init(problem, prefs), prefs=prefs )
        # res.keys()
        res['prefs'] = prefs

        folder_name = os.path.join(root_name, 'Output', 'discrete', args.problem_name, args.solver_name,
                                   'seed_{}'.format(args.seed_idx))
        os.makedirs(folder_name, exist_ok=True)
        if problem.n_obj == 2:
            plot_figure_2d(problem=problem)
        elif problem.n_obj == 3:
            plot_figure_2d()

        save_figures(folder_name=folder_name)
        save_pickles(folder_name=folder_name)

        avg_res_x += res['x']

    avg_res_x /= len(seeds)
    res['y'] = problem.evaluate(res['x'])
    folder_name = os.path.join(root_name, 'Output', 'discrete', args.problem_name, args.solver_name,
                               '3seeds')
    os.makedirs(folder_name, exist_ok=True)
    plot_figure_2d(problem=problem)
    save_figures(folder_name=folder_name)

