# CBS: Robust Learning with Noisy and Imbalanced Labels
**Abstract:** Learning with noisy labels has gained increasing attention because imperfect labels are ubiquitous in real-world scenarios and substantially hurt the deep model performance.
Recent studies tend to regard low-loss samples as clean ones and discard high-loss ones to alleviate the negative impact of noisy labels. However, real-world datasets contain not only noisy labels bus also class-wise sample imbalance. The imbalance issue is prone to causing failure in the loss-based sample selection since the under-learning of minority classes also leads to high losses. To this end, we propose a method termed Cass-Balance-based sample Selection (CBS) to prevent the minority class samples from being neglected in the training process. We propose Confidence-based Sample Augmentation (CSA) for the chosen clean samples to enhance their reliability in the training process. Finally, we employ Consistency Regularization (CR) between different views of selected noisy samples to maximize data exploitation and boost model performance further. Comprehensive experimental results on synthetic and real-world datasets demonstrate the effectiveness and superiority of our proposed method, especially on class-imbalanced datasets.

# Pipeline

![framework](Feature2_4.png)

### Usage

Here is an example shell script to run CBS on CIFAR-10 :

```python
 python main.py --warmup-epoch 40 --epoch 250 \
        --rho-range 0.6:0.6:100 --batch-size 128 \
        --lr 0.05 --warmup-lr 0.01 --start-expand 200 --noise-type unif \
        --closeset-ratio 0.4 --lr-decay cosine:40,5e-5,240 \
        --opt sgd --dataset cifar10 --imbalance True --imb-factor 0.05 --alpha 0.6 --aph 0.35



