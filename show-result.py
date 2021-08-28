import argparse
from pathlib import Path

import pandas as pd
import torch

pd.options.display.width = 100
pd.options.display.max_colwidth = 9999

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
parser.add_argument('glob', type=str, help="glob checkpoints in folder")
parser.add_argument('key', help=" primary key for evaluation in ckpt")
parser.add_argument('--higher', action='store_true', help="is higher better?")
parser.add_argument('-l','--list', dest='list', nargs='*', default=list(), help='Other keys list')

args = parser.parse_args()


def result_from_ckpt(args):
    def fileiter(args):
        assert args.folder.is_dir(), f"{args.folder} does not exist!"
        for f in args.folder.glob(args.glob):
            yield f

    datalst = []
    for f in fileiter(args):
        ckpt = torch.load(f, map_location='cpu')
        lst = [str(f), ckpt[args.key]] + [ckpt[i] for i in args.list]
        datalst.append(lst)
    cols = ['file'] + [args.key] + args.list
    df = pd.DataFrame(datalst, columns=cols)
    df = df.sort_values(by=args.key, ascending=not args.higher)
    return df


if __name__ == '__main__':
    print(args)
    df = result_from_ckpt(args)
    print(df)
