import argparse
import numpy as np
import os
import pylab
from function_utils import hist_k_occurrence, top_k 

pylab.rcParams.update({'font.size': 22})
trans_dir='exp/bli_500K/'
eps = np.finfo(float).eps
langs=['de', 'es', 'en', 'fr', 'it', 'pt']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, choices=langs,
                        help='source language')
    parser.add_argument('tgt', type=str, choices=langs,
                        help='target language (should be different from src)')
    parser.add_argument('-k', type=int, default=10,
                        help='read top-k')
    parser.add_argument('-n', type=int, default=40,
                        help='number of bins to histogram the counts')
    parser.add_argument('-q', type=float, default=1.0,
                        help='quantile of histogram to cut off')
    parser.add_argument('-N', type=int,
                        help='list N hubs')
    return parser.parse_args()

def read_translations(src, tgt, k=10, method='nn'):
    translations = []
    with open(os.path.join(trans_dir,
                           src + '-' + tgt + '.' + method + '.txt')) as f:
        for line in f:
            ws = line.strip().split()
            translations.append(ws[1:k+1])
    return translations

if __name__ == '__main__':
    args = parse_args()
    colors = ['b', 'g', 'k', 'r']
    for i, mtd in enumerate(['nn', 'isf', 'csls', 'hnn']):
        translations = read_translations(args.src, args.tgt, args.k, mtd)
        bins, freqs = hist_k_occurrence(translations, args.n, args.q)
        pylab.loglog(bins, freqs, colors[i], linewidth=2, label=mtd.upper())

        if mtd == 'nn':
            translations = np.hstack(translations)
            tr, cnt = np.unique(translations, return_counts=True)
            inds, cnt = top_k(cnt, args.N)
            hubs = [tr[ind] for ind in inds]
            occurs = [[c] for c in cnt]
        else:
            translations = np.hstack(translations)
            tr, cnt = np.unique(translations, return_counts=True)
            for ii, w in enumerate(hubs):
                ind = tr==w
                if np.sum(ind) == 1:
                    occurs[ii].append(cnt[tr==w][0])
                elif np.sum(ind) == 0:
                    occurs[ii].append(0)

    print("Hubs\t\t\t\tNN\t\tISF\t\tCSLS\t\tHNN\n")
    for i, w in enumerate(hubs):
        print("{}\t\t\t\t{}\t\t{}\t\t{}\t\t{}".format(w,
              occurs[i][0], occurs[i][1], occurs[i][2], occurs[i][3]))

    pylab.legend()
    pylab.xlabel(r'$N_{'+str(args.k)+'}$')
    pylab.ylabel(r'$p(N_{' + str(args.k) +'})$')
    pylab.tight_layout()
    pylab.show()
    pylab.savefig(
        os.path.join(trans_dir, src + '-' + tgt + '.k_occur.png'))
