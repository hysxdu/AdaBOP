AdaBOP
----------
This is the official code for Achieving Plasticity-Stability Trade-off in Continual Learning Through Adaptive Balanced Orthogonal Projection.

Requirements
----------
torch 1.11.0  
python 3.8.0  
tensorboardX 2.5.1  
scikit-learn 1.1.2  
numpy 1.22.4    

Usage
----------
###### AdaBOP  
sh scripts/split_CIFAR100_10_incremental_task.sh  
sh scripts/split_CIFAR100_20_incremental_task.sh
sh subimagenet.sh   

###### AdaBOP-NS
sh scripts/cifar100_10.sh  
sh scripts/cifar100_20.sh  
sh scripts/tiny.sh  

###### CIFAR-10 & MNIST
python CIFAR10/run_cifar_10.py
