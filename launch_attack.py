# launch_attack.py
import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Launch Multiple Processes for One Pixel Attack')
    parser.add_argument('--num_processes', type=int, default=1, help='Total number of processes (and GPUs)')
    parser.add_argument('--script', type=str, default='one_pixel_attack.py', help='Script to run')
    parser.add_argument('--additional_args', type=str, default='', help='Additional arguments to pass to the script')
    args = parser.parse_args()

    num_processes = args.num_processes
    processes = []

    for i in range(num_processes):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)

        cmd = [
            'python', args.script,
            '--process_idx', str(i),
            '--num_processes', str(num_processes),
            '--gpu_device', '0',
        ]
        if args.additional_args:
            cmd.extend(args.additional_args.strip().split())

        print(f"Launching process {i} on GPU {i}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    for p in processes:
        p.wait()

if __name__ == '__main__':
    main()
