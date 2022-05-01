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
        'UPOS F1': 'Part-of-speech F1',
        'UPOS precision': 'Part-of-speech precision',
        'UPOS recall': 'Part-of-speech recall',
        'UPOS aligned accuracy': 'Part-of-speech aligned accuracy',
    })

    # Leave out the FinnPos evaluation on ftb1u, because ftb1u is the
    # training set for FinnPos
    results = full_results[~((full_results['model'] == 'FinnPos') &
                             (full_results['Dataset'] == 'ftb1u'))]

    model_order = [
        'Voikko', 'UDPipe', 'stanza', 'Turku parser', 'FinnPos', 'spacy-fi',
        'trankit-base', 'trankit-large', 'simplemma', 'uralicnlp'
    ]
    model_order = [x for x in model_order if x in results['model'].unique()]

    sns.barplot(x='model', y='Lemmatization F1', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_f1_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization precision', data=results,
                hue='Dataset', order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_precision_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='Lemmatization recall', data=results,
                hue='Dataset', order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'lemma_recall_by_dataset.png'))
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

    sns.barplot(x='model', y='Part-of-speech F1', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_f1_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='Part-of-speech precision', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_precision_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='Part-of-speech recall', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_recall_by_dataset.png'))
    plt.close()

    sns.barplot(x='model', y='Part-of-speech aligned accuracy', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_acc.png'))
    plt.close()

    sns.barplot(x='model', y='Part-of-speech F1', data=results, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xlabel('')
    plt.savefig(os.path.join(imagedir, 'pos_f1.png'))
    plt.close()

    results_by_model = results.groupby('model').mean().reset_index()

    sns.relplot(x='Lemmatization F1', y='Sentences per second',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Lemmatization F1'],
                          results_by_model['Sentences per second'],
                          results_by_model['model']):
        ha = 'right'
        textrelx = -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.savefig(os.path.join(imagedir, 'lemma_speed.png'))
    plt.close()

    sns.relplot(x='Part-of-speech F1', y='Sentences per second',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Part-of-speech F1'],
                          results_by_model['Sentences per second'],
                          results_by_model['model']):
        ha = 'right'
        textrelx = -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
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
