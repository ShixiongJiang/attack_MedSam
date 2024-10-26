import argparse
import os
import subprocess

def run_attack(process_idx, num_processes, script, additional_args):
    # 设置每个进程的环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(process_idx)
    print(f"Launching process {process_idx} on GPU {process_idx}")

    cmd = [
        'python', script,
        '--process_idx', str(process_idx),
        '--num_processes', str(num_processes),
        '--gpu_device', '0'  # 每个进程只看到一个 GPU，索引为 0
    ]
    if additional_args:
        cmd.extend(additional_args.strip().split())

    # 使用 subprocess.Popen 并传递环境变量
    process = subprocess.Popen(cmd, env=env)
    process.wait()

def main():
    parser = argparse.ArgumentParser(description='Launch Multiple Processes for One Pixel Attack')
    parser.add_argument('--num_processes', type=int, default=1, help='Total number of processes (and GPUs)')
    parser.add_argument('--script', type=str, default='one_pixel_attack.py', help='Script to run')
    parser.add_argument('--additional_args', type=str, default='', help='Additional arguments to pass to the script')
    args = parser.parse_args()

    processes = []
    for i in range(args.num_processes):
        p = subprocess.Popen(
            [
                'python', 'one_pixel_attack.py',
                '--process_idx', str(i),
                '--num_processes', str(args.num_processes),
                '--gpu_device', '0'  # 每个进程内的 GPU 索引为 0
            ] + args.additional_args.strip().split(),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        )
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.wait()

if __name__ == '__main__':
    main()
