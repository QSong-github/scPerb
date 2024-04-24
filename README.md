# scPerb

Traditional methods for obtaining cellular responses after perturbation are usually labor-intensive and costly, especially when working with multiple different experimental conditions. Therefore, accurate prediction of cellular responses to perturbations is of great importance in computational biology. To address this problem, some methodologies have been previously developed, including graph-based approaches, vector arithmetic, and neural networks. However, these methods either mix the perturbation-related variances with the cell-type-specific patterns or implicitly distinguish them within black-box models. In this work, we introduce a novel framework, scPerb, to explicitly extract the perturbation-related variances and transfer them from unperturbed cells to perturbed cells. scPerb adopts the style transfer strategy by incorporating a style encoder into the architecture of a variational autoencoder. Such style encoder accounts for the differences in the latent representations between unperturbed cells and perturbed cells, which allows scPerb to accurately predict the gene expression data of cells after perturbation. Through comprehensive comparisons with existing methods, scPerb presents improved performance and higher accuracy in predicting cellular responses to perturbations. Specifically, scPerb not only outperforms other methods across multiple datasets, but also achieves superior $R^2$ values of $0.98$, $0.98$, and $0.96$ on three benchmarking datasets.



## Highlights

- scPerb merged the concept of style transfer and VAE, resulting in incredible accuracy.
- scPerb outperforms all the existing models of single-cell perturbation prediction
- scPerb is developed and tailored for single-cell perturbation prediction and provided as a ready-to-use open-source software, demonstrating high accuracy and robust performance over existing methods.

## FAQ

- **How can I install scPerb?**
    You can download scPerb from our GitHub link:

    ```
    git clone https://github.com/tongtongtot/scperb.git
    ```

    scPerb is built based on PyTorch, tested in Ubuntu 18.04, CUDA environment (cuda 11.6).        

    The requirement packages include:
    
    ```
    anndata==0.10.3
    matplotlib==3.7.2
    numpy==1.24.3
    pandas==2.0.3
    scanpy==1.9.6
    scipy==1.11.1
    seaborn==0.12.2
    torch==2.1.0
    torchaudio==2.1.0
    torchvision==0.16.0
    tqdm==4.65.0
    wget==3.2
    ```
    
    or you can also use the following scripts:
    
    ```
    pip install -r requirements.txt
    ```
    
- **I want to try the PBMC demo, can I run scPerb in one command line?**    
    You can use the following commands:

    ```
    python3 scperb.py
    ```

    or please refer to our training tutorial [here](https://github.com/tongtongtot/scperb-tutorial)

- **Do I need a GPU for running scPerb?**   
    scPerb can run on a standard laptop without GPU. For computational efficiency, we recommend you use a GPU. scPerb could detect whether there is an available GPU or not, so do not worry about this.

- **Can I generate my configuration file using the command line? **  
    You can change the default settings in the option.py, while the usage and description are written inside the document.

- **Link?**   
    [pdf](http://yau-awards.com/uploads/file/20231031/20231031150434_30639.pdf)
