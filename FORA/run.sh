# python main.py --save_path cifar_level2.pth  --iteration 40000 --gid 1 --layer_id 2  \
#     --lr 1e-4 --dlr 3e-4 --a 0.8 --dataset cifar --batch_size 64 

python main.py --iteration 12000 --layer_id 1 --dataset_num 2500 --dlr 3e-4 --a 0.5

python main.py --iteration 12000 --layer_id 2 --dataset_num 2500 --dlr 3e-4 --a 0.5

python main.py --iteration 12000 --layer_id 2 --dataset_num 5000 --dlr 3e-4 --a 0.5

python main.py --iteration 20000 --layer_id 2 --dataset_num 5000 --dlr 3e-4 --a 0.5

