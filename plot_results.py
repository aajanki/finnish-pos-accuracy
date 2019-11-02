import os
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    imagedir = 'results/images'
    os.makedirs(imagedir, exist_ok=True)

    results = pd.concat([
        load_results('UD_Finnish_TDT'),
        load_results('ftb1u'),
        load_results('ftb2-news'),
        load_results('ftb2-sofie'),
        load_results('ftb2-wikipedia')
    ])    

    sns.barplot(x='model', y='Lemmatization WER', hue='testset', data=results)
    plt.savefig(os.path.join(imagedir, 'lemma.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS WER', hue='testset', data=results)
    plt.savefig(os.path.join(imagedir, 'pos.png'))
    plt.close()

    sns.relplot(x='Sentences per second', y='Lemmatization WER', hue='model', data=results)
    plt.xscale('log')
    plt.savefig(os.path.join(imagedir, 'lemma_speed.png'))


def load_results(testsetname):
    resultdir = 'results'
    filename = os.path.join(resultdir, f'evaluation_{testsetname}.csv')
    results = pd.read_csv(filename, index_col=0)
    results = results.reset_index().rename(columns={'index': 'model'})
    name = pd.DataFrame([testsetname]*results.shape[0], columns=['testset'])
    return pd.concat([results, name], axis=1)


if __name__ == '__main__':
    main()
