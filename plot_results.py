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
    model_order = [
        'Voikko', 'UDPipe-fi-tdt', 'StanfordNLP', 'Turku-neural-parser', 'FinnPos'
    ]

    sns.barplot(x='model', y='Lemmatization WER', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'lemma_wer.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization WER', hue='testset', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'lemma_wer_by_testset.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization aligned accuracy', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'lemma_acc.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization F1', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'lemma_f1.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS WER', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'pos_wer.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS WER', hue='testset', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'pos_wer_by_testset.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS aligned accuracy', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'pos_acc.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS F1', data=results, order=model_order)
    plt.savefig(os.path.join(imagedir, 'pos_f1.png'))
    plt.close()

    sns.relplot(x='Lemmatization WER', y='Sentences per second', hue='model', data=results)
    plt.yscale('log')
    plt.savefig(os.path.join(imagedir, 'lemma_speed.png'))
    plt.close()

    sns.relplot(x='UPOS WER', y='Sentences per second', hue='model', data=results)
    plt.yscale('log')
    plt.savefig(os.path.join(imagedir, 'pos_speed.png'))
    plt.close()


def load_results(testsetname):
    resultdir = 'results'
    filename = os.path.join(resultdir, f'evaluation_{testsetname}.csv')
    results = pd.read_csv(filename, index_col=0)
    results = results.reset_index().rename(columns={'index': 'model'})
    name = pd.DataFrame([testsetname]*results.shape[0], columns=['testset'])
    return pd.concat([results, name], axis=1)


if __name__ == '__main__':
    main()
