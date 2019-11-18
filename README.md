# Fully-Explore-GPU
Constantly search GPU with free memory to launch designed process

* Usage:     python run.py file_name interval memory start_id gpus
file_name  py file name (without .py) to import task
interval   search time interval, seconds for process to acquire full memory it uses
memory     memory requirement (GB)
start_id   the first task id to start, 0 for start from scratch (task 0)
gpus       GPU(s) to be used

Example:   python run.py main 900 8 5 2,3
IMPORTANT: def yield_task_parameters(): in file_name.py
           which yields command line parameters in each call until StopIteration
           for example, write function below in file_name.py

def yield_task_parameters():
    for net_white_name in ['ResNet152', 'InceptionV3', 'VGG19', 'DenseNet121']:
        for net_black_name in ['ResNet152', 'InceptionV3', 'VGG19', 'DenseNet121', 'NASNetLarge', 'Xception', 'InceptionResNet']:
            if net_white_name == net_black_name: continue
            yield net_white_name, net_black_name
