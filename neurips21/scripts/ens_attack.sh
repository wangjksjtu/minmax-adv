epochs=50

########################## Table 1 ##########################
# MNIST (L_infty)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --eps 0.2 --epochs $epochs --alpha 4 --beta 50 --gamma 3 --models ABCD)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --eps 0.2 --epochs $epochs --alpha 4 --beta 50 --gamma 3 --models ABCD --avg_case True)&

# MNIST (L_2)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --eps 3.0 --epochs $epochs --alpha 10 --beta 100 --gamma 3 --models ABCD --norm 2)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --eps 3.0 --epochs $epochs --alpha 10 --beta 100 --gamma 3 --models ABCD --avg_case True --norm 2)&

# MNIST (L_1)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --eps 20.0 --epochs $epochs --alpha 4 --beta 100 --gamma 5 --models ABCD --norm 1)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --eps 20.0 --epochs $epochs --alpha 4 --beta 100 --gamma 5 --models ABCD --avg_case True --norm 1)&

# MNIST (L_0)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --eps 30.0 --epochs $epochs --alpha 1 --beta 100 --gamma 7 --models ABCD --norm 0)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --eps 30.0 --epochs $epochs --alpha 1 --beta 100 --gamma 7 --models ABCD --avg_case True --norm 0)&

########################## Table 3 ##########################
# CIFAR (L_infty)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --dataset cifar --eps 0.05 --epochs $epochs --alpha 6 --beta 50 --gamma 4 --models ABCD)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --dataset cifar --eps 0.05 --epochs $epochs --alpha 6 --beta 50 --gamma 4 --models ABCD --avg_case True)&

CIFAR (L_2)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --dataset cifar --eps 3.0 --epochs $epochs --alpha 10 --beta 100 --gamma 4 --models ABCD --norm 2)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --dataset cifar --eps 3.0 --epochs $epochs --alpha 10 --beta 100 --gamma 4 --models ABCD --avg_case True --norm 2)&

CIFAR (L_1)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --dataset cifar --eps 30.0 --epochs $epochs --alpha 4 --beta 100 --gamma 4 --models ABCD --norm 1)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --dataset cifar --eps 30.0 --epochs $epochs --alpha 4 --beta 100 --gamma 4 --models ABCD --avg_case True --norm 1)&

CIFAR (L_0)
(export CUDA_VISIBLE_DEVICES=0 && python ens_attack.py --dataset cifar --eps 50.0 --epochs $epochs --alpha 1 --beta 100 --gamma 7 --models ABCD --norm 0)&
(export CUDA_VISIBLE_DEVICES=1 && python ens_attack.py --dataset cifar --eps 50.0 --epochs $epochs --alpha 1 --beta 100 --gamma 7 --models ABCD --avg_case True --norm 0)&