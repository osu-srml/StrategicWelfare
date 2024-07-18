# StrategicWelfare

This is the code for our paper ```Non-linear Welfare-Aware Strategic Learning```

The experimental results on 3 datasets can be replicated by directly running 3 notebooks:

- ```Welfare_Logreg_Quadratic```: This notebook produces all results for the Quadratic dataset
- ```Welfare_Logreg_German```: This notebook produces all results for the German Credit dataset
- ```Welfare_Logreg_ACS```: This notebook produces all results for the ACSIncome dataset

All datasets are stored in ```data```, but you need to install ```folktables``` to enable downloading from ACSIncome dataset (refer to [this paper](https://arxiv.org/pdf/2108.04884) for more details). while the pre-trained labeling models are in ```h_models```. You are welcome to check [our paper](https://arxiv.org/pdf/2405.01810) which emphasizes more on theoretical analysis.
