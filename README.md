This is the repository for our paper "Towards Defending Multiple $\ell_p$-norm Bounded Adversarial Perturbations via Gated Batch Normalization".

## Prerequisites
Create anaconda environment.
```
conda create -n gbn python=3.6
conda activate gbn
```
Install requirements.
```
pip install -r requirements.txt
```


## Training


Train an adversarial defensed LeNet5 model with GBN module, and test its accuracy under PGD $\ell_{1}$, $\ell_{2}$, and $\ell_{\infty}$ attack:
```
python train_gbn.py
```

Train a vanilla LeNet5 model without any adversarial attacks:
```
python train_lenet_vanilla.py
```

Train an adversarial defensed model using the average loss of $\ell_{1}$, $\ell_{2}$, and $\ell_{\infty}$ PGD adversarial attack:
```
python train_lenet_AVG.py
```
