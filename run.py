import time
import os
import pynvml
import argparse
from prettytable import PrettyTable
import importlib


"""
Usage:     python run.py file_name interval memory start_id gpus
Example:   python run.py main 900 8 5 2,3
IMPORTANT: def yield_task_parameters(): in file_name.py
           which yields command line parameters in each call until StopIteration
           for example, write function below in file_name.py

def yield_task_parameters():
    for net_white_name in ['ResNet152', 'InceptionV3', 'VGG19', 'DenseNet121']:
        for net_black_name in ['ResNet152', 'InceptionV3', 'VGG19', 'DenseNet121', 'NASNetLarge', 'Xception', 'InceptionResNet']:
            if net_white_name == net_black_name: continue
            yield net_white_name, net_black_name
"""

# get current time str
def get_time(deviation=0): return time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()-deviation))


# decide whether a gpu in index is available (have enough free memory)
def gpu_is_free(memory, index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if meminfo.free / (1024 ** 3) > memory: return True
    return False


# get file's time information
def get_file_time(file):
    def convert_time(t): return time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(t))
    return [convert_time(os.path.getmtime(file))]


# send command constantly
def run_multi_task(file_name, interval, memory, start_id, gpus=[]):
    pynvml.nvmlInit()
    task_generator = importlib.import_module(file_name).yield_task_parameters()
    for i in range(start_id): next(task_generator)
    if gpus == []: gpus = list(range(pynvml.nvmlDeviceGetCount()))
    log_table = {}

    while 1:
        try:
            for i in gpus:
                if not gpu_is_free(memory, int(i)): continue
                parameters = next(task_generator)
                
                # start nohup command, log_file_name = parameter1_paremeter2_parameter3(and so on).log
                cmd = 'nohup python -u ' + file_name + '.py'
                log_file_name = ''
                for p in parameters: 
                    cmd += ' ' + str(p)
                    log_file_name += str(p) + '_'
                log_file_name = log_file_name[:-1] + '.log'
                cmd = cmd + ' ' + str(i) + ' > ' + log_file_name + ' 2>&1 &'
                
                # run command and record
                os.system(cmd)
                log_table[log_file_name] = [log_file_name, cmd, get_time()]
                print(get_time(), '   ', cmd)
                time.sleep(2)
            time.sleep(interval)
        
        # no task remaining
        except StopIteration:
            print(get_time(), '   ', 'Done!\n')
            time.sleep(interval)
            break
    
    # output result
    table = PrettyTable(['Log', 'Command', 'Start', 'Finished', 'Error'])
    table.align["Log"] = "l"
    table.align["Command"] = "l"
    for log_file_name in log_table:
        with open(log_file_name, 'r') as f:
            table.add_row(log_table[log_file_name] + get_file_time(log_file_name) + ['Traceback (most recent call last):' in f.read()])
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file_name', help='py file name (without .py) to import task')
    parser.add_argument('interval', help='search time interval (seconds) for process to acquire full memory it uses')
    parser.add_argument('memory', help='memory requirement (GB)')
    parser.add_argument('start_id', help='the first task id to start, 0 for start from scratch (task 0)')
    parser.add_argument('gpus', help='GPU(s) to be used')
    args, _ = parser.parse_known_args()
    run_multi_task(file_name=args.file_name,
                   interval=float(args.interval), 
                   memory=float(args.memory), 
                   start_id=int(args.start_id),
                   gpus=args.gpus.split(','))
