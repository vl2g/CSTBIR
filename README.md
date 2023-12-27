# CSTBIR
**Code from paper: Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions**

[project page](https://vl2g.github.io/projects/cstbir/) | [paper](https://vl2g.github.io/projects/cstbir/resources/paper.pdf)

## Requirements
* Use **python >= 3.8.16**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.13.1 CUDA 11.6**

* Other requirements from 'requirements.txt' and 'environment.yaml'

**To setup environment**
```
# create new env cstbir
$ conda create -n cstbir python=3.8.16
```

```
# activate cstbir
$ conda activate cstbir
```

```
# install other dependencies
$ conda env update --file environment.yml --prune
$ pip install -r requirements.txt
```

## Preparing dataset
- Download VG images from [https://homes.cs.washington.edu/~ranjay/visualgenome/index.html](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)

- Download QuickDraw Sketches from [https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset/)

- Download CSTBIR dataset from [Google Drive Link](https://drive.google.com/drive/folders/1UgAZc5rtbO0MQ37WHS4hGQhXlqMPT6Lg?usp=sharing)

Store the downloaded dataset in the `./data/` directory.

## Running the code

### Training parameters
To check and update training, model and dataset parameters see [configs](config.yaml)

### To train the model: 
```
$ CUDA_VISIBLE_DEVICES=XX python run.py
```

## Cite
If you find this code/paper  useful for your research, please consider citing.
```
@InProceedings{cstbir2024aaai,
        author    = {Gatti, Prajwal and Parikh, Kshitij Gopal and Paul, Dhriti Prasanna and Gupta, Manish and Mishra, Anand},
        title     = {Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions},
        booktitle = {AAAI},
        year      = {2024},
}   
```

## Acknowledgements
This work uses https://github.com/openai/CLIP/ for the implementation of the CLIP model.