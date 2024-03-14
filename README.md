# Domain Adaptation using Normalizing Flows

This repository includes code for unsupervised domain adaptation for binary classification, under the assumption of covariate shift, on Amazon reviews dataset ``[Ni et. al (2019)] `` using tranport maps learned via the machine learning architecture of normalizing flows. We use masked autoregressive flows ``[Papamakarios et al. (2021)]`` specifically to learn transport maps between source and target domain features.

Let $(X_S, Y_S)$ denote the source domain where $X_S$ are text embeddings and $Y_S$ are binary labels. Denote by $(X_T, Y_T)$ similar quantities for the target domain.  Under our setup, labeled i.i.d. samples are available from source domain but only unlabeled samples are available form the target domain. A classifier learned on the labeled source data is tranfered to the target domain by mapping the source and target features to a latent space, distributed as standard normal distribution, using normalizing flows.
That is, we learn normalizing flows $f_S$ and $f_T$ that map $X_S$ and $X_T$ to normal distribution respectively.

Text embeddings are created by 
1. Finetuning a BERT model on a set of randomly samples 20,000 data points for an empirical risk minimization task
2. Using the penultimate layer of BERT to obtain last hiddebn layer embeddings.

The code for finetuning BERT, creating splits for source and target domain, and obatining embedding can be found in 

```
./create_dataset.ipynb
```

The code for training masked autoregressive flows $f_S$ and $f_T$ and transferring classifier from source to target domain xan be found in

```
./amazon.ipynb
```

