import os
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def main():
    imagedir = 'results/images'
    os.makedirs(imagedir, exist_ok=True)

    full_results = pd.concat([
        load_results('UD_Finnish_TDT'),
        load_results('ftb1u'),
        load_results('ftb2-news'),
        load_results('ftb2-sofie'),
        load_results('ftb2-wikipedia')
    ])
    full_results['model'] = (full_results['model']
                             .replace('Turku-neural-parser', 'Turku parser')
                             .replace('UDPipe-fi-tdt', 'UDPipe'))
    full_results = full_results.rename(columns={
        'Lemmatization WER': 'Lemmatization error rate',
        'UPOS WER': 'Part-of-speech error rate'
    })

    # Leave out the FinnPos evaluation on ftb1u, because ftb1u is the
    # training set for FinnPos
    results = full_results[~((full_results['model'] == 'FinnPos') &
                             (full_results['Dataset'] == 'ftb1u'))]

    model_order = [
        'Voikko', 'UDPipe', 'stanza', 'Turku parser', 'FinnPos', 'spacy-fi', 'trankit'
    ]

    sns.barplot(x='model', y='Lemmatization error rate', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_wer.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization error rate', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_wer_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization aligned accuracy', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_acc.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization F1', data=results, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_f1.png'))
    plt.close()

    sns.barplot(x='model', y='Part-of-speech error rate', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_wer.png'))
    plt.close()

    sns.barplot(x='model', y='Part-of-speech error rate', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_wer_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS aligned accuracy', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_acc.png'))
    plt.close()

    sns.barplot(x='model', y='UPOS F1', data=results, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_f1.png'))
    plt.close()

    results_by_model = results.groupby('model').mean().reset_index()
    results_by_model['Duration per 1000 sentences (s)'] = \
        1000/results_by_model['Sentences per second']

    sns.relplot(x='Lemmatization error rate', y='Duration per 1000 sentences (s)',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Lemmatization error rate'],
                            results_by_model['Duration per 1000 sentences (s)'],
                            results_by_model['model']):
        ha = 'right'
        textrelx = -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.xlim([0, 0.15])
    plt.yscale('log')
    plt.savefig(os.path.join(imagedir, 'lemma_speed.png'))
    plt.close()

    sns.relplot(x='Part-of-speech error rate', y='Duration per 1000 sentences (s)',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Part-of-speech error rate'],
                          results_by_model['Duration per 1000 sentences (s)'],
                          results_by_model['model']):
        ha = 'right'
        textrelx = -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.xlim([0, 0.27])
    plt.yscale('log')
    plt.savefig(os.path.join(imagedir, 'pos_speed.png'))
    plt.close()


def load_results(datasetname):
    resultdir = 'results'
    filename = os.path.join(resultdir, f'evaluation_{datasetname}.csv')
    results = pd.read_csv(filename, index_col=0)
    results = results.reset_index().rename(columns={'index': 'model'})
    name = pd.DataFrame([datasetname]*results.shape[0], columns=['Dataset'])
    return pd.concat([results, name], axis=1)


if __name__ == '__main__':
    main()
