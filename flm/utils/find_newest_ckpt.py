import os
import glob
import sys


save_folder = sys.argv[1]
exp_name = sys.argv[2]
is_last = True if sys.argv[3] == 'choose_last' else False

# exp_name = '37_cl_causalflm_scratch_lr5e5_nobias_t0002_NEW_GPU32'
target = '{}/{}_seed*_from*/version_*/checkpoints/epoch*-step*.ckpt'.format(
    save_folder, exp_name)
if is_last:
    target = '{}/{}_seed*_from*/version_*/checkpoints/last.ckpt'.format(
        save_folder, exp_name)
out = glob.glob(target)


def get_info(p):
    p = p.rstrip('.ckpt')
    version = float(p.split('/')[-3].split('_')[-1])
    try:
        epoch = float(p.split('/')[-1].split('-')[0].split('_')[1])
    except:
        epoch = None
    try:
        score = float(p.split('/')[-1].split('-')[-1].split('_')[-1])
    except:
        score = None

    if score is None:
        score = -10000.

    return score, epoch, version


out = sorted(out, key=get_info, reverse=True)

if len(out):
    print(out[0])
