##

First you need to download data, ask for link and put files in data/ folder.


You might not need the first two exports, but for some reason my mac needs them to load the packages. 

To set up env, you will need pytorch. Google to install that, theta should be ok with out of box
```shell script
conda install pytorch torchvision cpuonly -c pytorch
conda install -c rdkit -c mordred-descriptor mordred
pip install pillow pandas tqdm cairosvg scikit-learn
```

```shell script
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
python main.py -mode image -smiles data/test.csv -model data/image_model.pt -smiles data/train.csv -n 100 -rnaseq data/cellpickle.pkl
```
or
```shell script
python main.py -mode descriptor -smiles data/test.csv -model data/desc_model.pt -smiles data/train.csv -n 100 -rnaseq data/cellpickle.pkl
```