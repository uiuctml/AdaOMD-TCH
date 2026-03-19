# from libmoonRAW.solver.gradient.methods.base_solver import GradAggSolver
# from libmoonRAW.solver.gradient.methods.mgda_solver import MGDAUBSolver
# from libmoonRAW.solver.gradient.methods.gradhv import GradHVSolver
# from libmoonRAW.solver.gradient.methods.pmtl import PMTLSolver
# from libmoonRAW.solver.gradient.methods.epo_solver import EPOSolver
# from libmoonRAW.solver.gradient.methods.moosvgd import MOOSVGDSolver
# from libmoonRAW.solver.gradient.methods.pmgda_solver import PMGDASolver
# from libmoonRAW.solver.gradient.methods.uniform_solver import UniformSolver
# from libmoonRAW.solver.gradient.methods.core.core_solver_bk import CoreAgg, CoreMGDA, CoreEPO, CoreMOOSVGD, CoreHVGrad
# def get_core_solver(args, pref=None):
#     if args.solver == 'agg':
#         return CoreAgg(pref=pref, agg_mtd=args.agg_mtd)
#     elif args.solver == 'mgda':
#         return CoreMGDA()
#     elif args.solver == 'epo':
#         return CoreEPO(pref=pref)
#     elif args.solver == 'moosvgd':
#         return CoreMOOSVGD(args=args)
#     elif args.solver == 'hvgrad':
#         return CoreHVGrad(args=args)
#     else:
#         assert False, 'not implemented'


from libmoonRAW.solver.gradient.methods.mgda_solver import MGDAUBSolver
from libmoonRAW.solver.gradient.methods.epo_solver import EPOSolver
from libmoonRAW.solver.gradient.methods.random_solver import RandomSolver