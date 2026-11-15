"""I plot the experiment results saved in a CSV file."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True,help='Generated CSV to plot')
    p.add_argument('--x',default='R_Variance',help='X-axis column (e.g. Gap_Length or R_Variance)')
    p.add_argument('--y',default='Identity_Changes',help='Metric to visualize')
    p.add_argument('--group',default='Method',help='Series column')
    p.add_argument('--output',default='outputs/plot.png')
    args=p.parse_args()
    df=pd.read_csv(args.csv)
    for c in (args.x,args.y,args.group):
        if c not in df.columns:raise ValueError(f'Missing {c}; available: {list(df.columns)}')
    fig,ax=plt.subplots(figsize=(8,5))
    for label,sub in df.groupby(args.group):
        sub=sub.sort_values(args.x)
        ax.plot(sub[args.x],sub[args.y],marker='o',label=label)
    ax.set_xlabel(args.x);ax.set_ylabel(args.y)
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=160)
    print('Saved',dest)

if __name__=='__main__': main()
