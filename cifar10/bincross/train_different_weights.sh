python cifar10_train.py --train_dir /scratch/tfiez/conformal/cifar10/bincross/equal/ --negative_weight 1.0
echo "1.0 /scratch/tfiez/conformal/cifar10/bincross/weight_10/" >> log.txt
data >> log.txt
python cifar10_train.py --train_dir /scratch/tfiez/conformal/cifar10/bincross/weight_05/ --negative_weight 0.5
echo "0.5 /scratch/tfiez/conformal/cifar10/bincross/weight_05/" >> log.txt
data >> log.txt
python cifar10_train.py --train_dir /scratch/tfiez/conformal/cifar10/bincross/weight_025/ --negative_weight 0.25
echo "0.25 /scratch/tfiez/conformal/cifar10/bincross/weight_025/" >> log.txt
data >> log.txt
python cifar10_train.py --train_dir /scratch/tfiez/conformal/cifar10/bincross/weight_005/ --negative_weight 0.05
echo "0.25 /scratch/tfiez/conformal/cifar10/bincross/weight_005/" >> log.txt
data >> log.txt
python cifar10_train.py --train_dir /scratch/tfiez/conformal/cifar10/bincross/weight_075/ --negative_weight 0.75
echo "0.25 /scratch/tfiez/conformal/cifar10/bincross/weight_075/" >> log.txt
data >> log.txt
python cifar10_train.py --train_dir /scratch/tfiez/conformal/cifar10/bincross/weight_001/ --negative_weight 0.01
echo "0.25 /scratch/tfiez/conformal/cifar10/bincross/weight_001/" >> log.txt
data >> log.txt
