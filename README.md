
### This is forked from the open source code:
https://github.com/shaoxiongji/federated-learning

IST 597 Data Privacy course project.

Utilize noise based differential privacy to defend deep leakage attacks towards federated learning system

## Requirements
python>=3.6  
pytorch>=0.4

## Run


Federated learning with DLG testing is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  




