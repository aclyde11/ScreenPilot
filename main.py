import argparse
import io
import pickle
from functools import partial
import cairosvg
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageOps
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision.transforms import ToTensor
from tqdm import tqdm

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Input is a smile. May fail, thats ok.
See MOrdred Documentation https://github.com/mordred-descriptor/mordred
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=['descriptor', 'image'], required=True, help='features to use')
    parser.add_argument('-model', type=str, required=True, help='pytorch model pt')
    parser.add_argument('-rnaseq', type=str, required=True, help='cell line pickle file')
    parser.add_argument('-imputer', type=str, required=False, help='needed for descriptors', default='data/imputer.pkl')
    parser.add_argument('-smiles', type=str, required=True,
                        help='csv with smiles as first row, will remove first row always in case header')
    parser.add_argument('-n', type=int, default=None, help='will limit smiles for testing if given number')
    return parser.parse_args()


def compute_descript(smile, imputer_dict=None):
    smi = Chem.MolFromSmiles(smile)
    calc = Calculator(descriptors, ignore_3D=True)
    res = calc(smi)
    res = np.array(list(res.values())).reshape(1, -1)
    if imputer_dict is not None:
        res = imputer_dict['scaler'].transform(imputer_dict['imputer'].transform(res))
    return res.flatten().astype(np.float32)

def smiles_to_image(mol, molSize=(128, 128), kekulize=True, mol_name='', mol_computed=False):
    if not mol_computed:
        mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100,
                                                   scale=1)))
    image.convert('RGB')
    return ToTensor()(Invert()(image))


class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """

    def invert(self, img):
        r"""Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            inv = ImageOps.invert(rgb)
            r, g, b = inv.split()
            inv = Image.merge('RGBA', (r, g, b, a))
        elif img.mode == 'LA':
            l, a = img.split()
            l = ImageOps.invert(l)
            inv = Image.merge('LA', (l, a))
        else:
            inv = ImageOps.invert(img)
        return inv

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        return self.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Joiner:
    def __init__(self, celllines, mode='image'):
        self.cellines = torch.from_numpy(celllines).to(device)
        self.ncell = celllines.shape[0]
        useing_images = mode == 'image'
        self.mode = self.image_prep if useing_images else self.vec_prep

    def vec_prep(self, v):
        v = v.float().view((1, -1)).repeat([self.ncell, 1])
        return v

    def image_prep(self, v):
        v = v.unsqueeze(0) if len(v.shape) == 3 else v
        v = v.float().repeat([self.ncell, 1, 1, 1])
        return v

    def __call__(self, drug_data):
        drug_data = self.mode(drug_data)
        return {'drug': drug_data, 'cell': self.cellines}


def get_model(fname):
    model = torch.load(fname, map_location='cpu')['inference_model']
    model = model.to(device)
    model.eval()
    return model


def load_cell_data(fname):
    with open(fname, 'rb') as f:
        df = pickle.load(f)
    cell_names = df.iloc[:, 0].tolist()
    cell_data = np.array(df.iloc[:, 1:], dtype=np.float32)
    print(cell_data.shape)
    return cell_names, cell_data


if __name__ == '__main__':
    args = get_args()

    imputer = None
    if args.imputer is not None:
        with open("data/imputer.pkl", 'rb') as f:
            imputer = pickle.load(f)

    print("Setting up some fake models.")
    prop_func = smiles_to_image if args.mode == 'image' else partial(compute_descript, imputer_dict=imputer)
    model = get_model(args.model)
    models_to_test = [model]

    _, celldata = load_cell_data(args.rnaseq)
    joiner = Joiner(celldata, args.mode)

    print("Loading smiles data data")
    try:
        if args.n is not None:
            smiles = pd.read_csv(args.smiles, nrows=args.n).iloc[:, 0].tolist()
        else:
            smiles = pd.read_csv(args.smiles, nrows=args.n).iloc[:, 0].tolist()
    except:
        smiles = pd.read_csv(args.smiles, nrows=200).iloc[:, 0].tolist()
    print("Loaded model, {} smiles, {} cells, and using {}".format(len(smiles), celldata.shape[0], args.mode))

    results_dict = {}
    with torch.no_grad():
        for smile in tqdm(smiles):
            vec = prop_func(smile)

            if vec is None:
                continue

            training_batch = joiner(vec)

            results = []
            for model in models_to_test:
                drugdata, celldata = training_batch['drug'].to(device), training_batch['cell']
                pred = model(celldata, drugdata)
                results.append(pred)

            results_dict[smile] = results

    print("Done. ")
    print(results_dict)
