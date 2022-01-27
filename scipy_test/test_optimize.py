import os
import time
from scipy.optimize import minimize


def smooth_session_pts(positions_3d, subject, action, cam_idx, output_folder, start, end):
    global LOSS_CONFIGS, x0
    output_path = os.path.join(output_folder,
                               '{}_{}_{}.session_smooth.start_{}.end_{}.sol.pickle'.format(subject, action, cam_idx,
                                                                                           start, end))
    if os.path.exists(output_path):
        sol = load_sol_from_pickle(output_path)
    else:
        import copy
        x0 = copy.deepcopy(positions_3d.flatten())
        s = time.time()
        LOSS_CONFIGS = {}
        LOSS_CONFIGS['BONELENGTH'] = {}
        LOSS_CONFIGS['BONELENGTH']['predefined'] = True
        LOSS_CONFIGS['BONELENGTH']['loss_weight'] = 20

        LOSS_CONFIGS['REGULARIZATION'] = {}
        LOSS_CONFIGS['REGULARIZATION']['loss_weight'] = 1

        LOSS_CONFIGS['TSMOOTH'] = {}
        LOSS_CONFIGS['TSMOOTH']['loss_weight'] = 1

        sol = minimize(ba_gt, x0, options={'maxiter': 100})
        time_cost = time.time() - s
        write_sol_to_pickle(output_path, sol)
        logger.info('written sol to {} using {} s'.format(output_path, time_cost))
    filtered_3d_gt = sol.x
    filtered_3d_gt = filtered_3d_gt.reshape((len(positions_3d), NUM_KPT, 3))
    return filtered_3d_gt

