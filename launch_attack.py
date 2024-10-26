import argparse
import os
import torch.multiprocessing as mp

def run_attack(process_idx, num_processes, script, additional_args):
    # 设置每个进程的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(process_idx)
    print(f"Launching process {process_idx} on GPU {process_idx}")

    cmd = [
        'python', script,
        '--process_idx', str(process_idx),
        '--num_processes', str(num_processes),
        '--gpu_device', str(process_idx)
    ]
    if additional_args:
        cmd.extend(additional_args.strip().split())

    # 使用 os.system 来运行命令
    os.system(" ".join(cmd))

def main():
    parser = argparse.ArgumentParser(description='Launch Multiple Processes for One Pixel Attack')
    parser.add_argument('--num_processes', type=int, default=1, help='Total number of processes (and GPUs)')
    parser.add_argument('--script', type=str, default='one_pixel_attack.py', help='Script to run')
    parser.add_argument('--additional_args', type=str, default='', help='Additional arguments to pass to the script')
    args = parser.parse_args()

    # 启动多进程
    processes = []
    for i in range(args.num_processes):
        p = mp.Process(target=run_attack, args=(i, args.num_processes, args.script, args.additional_args))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')  # 使用 'spawn' 方法启动进程以确保多进程的兼容性
    main()
