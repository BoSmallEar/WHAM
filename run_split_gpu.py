# input a folder and split its subfolders to running on different gpus
import os
import sys
import subprocess
import multiprocessing as mp
from concurrent import futures

GPUS=[0, 1, 2, 3, 4, 5, 6, 7]

def run(file, log_file):
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    worker_id = cur_proc._identity[0] - 1  # 1-indexed processes
    gpu = GPUS[worker_id % len(GPUS)]
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python demo.py  --output_pth experiments/filter_results_0313 --video {file} --save_pkl --visualize"
    )
    # cmd = (
    #     f"CUDA_VISIBLE_DEVICES={gpu} "
    #     f"python scripts/save_action_figure.py {pkl_file}"
    # )

    print(f"LOGGING TO {log_file}")
    cmd = f"{cmd} > {log_file} 2>&1"
    print(cmd)
    subprocess.call(cmd, shell=True)

def main(root_folder):
    file_list = os.listdir(root_folder)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    with futures.ProcessPoolExecutor(max_workers=8) as exe:
        for file in file_list:
            base_name = os.path.splitext(file)[0]
            log_file = f"{log_dir}/{base_name}.log"
            file_full=os.path.join(root_folder, file)
            exe.submit(
                run,
                file_full,
                log_file
            )


if __name__ == "__main__":
    root_folder = sys.argv[1]
    main(root_folder)
