import os
from pathlib import Path


output_dir = Path("outputs")
if not output_dir.exists():
    print('output_dir does not exist.'.format(output_dir))
    exit(-1)
eval_folder = output_dir.joinpath('eval')
if not eval_folder.exists():
    eval_folder.mkdir(parents=True, exist_ok=True)

testlist = "./lists/dtu/test.txt"
with open(testlist) as f:
    scans = f.readlines()
    scans = [line.rstrip() for line in scans]

for scan in scans:
    scan_folder = output_dir.joinpath(scan).joinpath('points_mvsnet')
    # print(os.listdir(str(scan_folder)))
    consis_folders = [f for f in os.listdir(str(scan_folder)) if f.startswith('consistencyCheck-')]
    consis_folders.sort()
    # print(consis_folders)
    consis_folder = consis_folders[-1]
    source_ply = scan_folder.joinpath(consis_folder).joinpath('final3d_model.ply')
    scan_idx = int(scan[4:])
    target_ply = eval_folder.joinpath('mvsnet{:03d}_l3.ply'.format(scan_idx))
    # cmd = 'cp ' + str(source_ply) + ' ' + str(target_ply)
    cmd = 'mv ' + str(source_ply) + ' ' + str(target_ply)
    print(cmd)
    os.system(cmd)