import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path


def main():
    resultdir = Path('results')
    imagedir = Path('results/images')
    imagedir.mkdir(exist_ok=True)

    full_results = pd.read_csv(resultdir / 'evaluation.csv')
    full_results['Model'] = (
        full_results['Model']
        .replace({
            'Turku-neural-parser': 'Turku parser',
            'UDPipe-fi-tdt': 'UDPipe',
            'spacy-fi_core_news_sm': 'spacy-sm',
            'spacy-fi_core_news_md': 'spacy-md',
            'spacy-fi_core_news_lg': 'spacy-lg',
            'spacy-fi_experimental_web_md': 'spacy-experimental',
        }))
    full_results = full_results.rename(columns={
        'UPOS F1': 'Part-of-speech F1',
        'UPOS precision': 'Part-of-speech precision',
        'UPOS recall': 'Part-of-speech recall',
        'UPOS aligned accuracy': 'Part-of-speech aligned accuracy',
    })

    # Leave out the FinnPos evaluation on ftb1u, because ftb1u is the
    # training set for FinnPos
    results = full_results[~((full_results['Model'] == 'FinnPos') &
                             (full_results['Dataset'] == 'ftb1u'))]

    model_order = sorted(results['Model'].unique())

    sns.barplot(x='Model', y='Lemmatization F1', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_f1_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization precision', data=results,
                hue='Dataset', order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_precision_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization recall', data=results,
                hue='Dataset', order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_recall_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization aligned accuracy', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_acc.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization F1', data=results, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_f1.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech F1', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_f1_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech precision', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_precision_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech recall', hue='Dataset',
                data=full_results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_recall_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech aligned accuracy', data=results, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_acc.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech F1', data=results, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_f1.png')
    plt.close()

    results_by_model = results.groupby('Model').mean().reset_index()

    label_left_lemma = ['FinnPos', 'stanza', 'simplemma', 'spacy-lg']

    sns.relplot(x='Lemmatization F1', y='Tokens per second',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Lemmatization F1'],
                          results_by_model['Tokens per second'],
                          results_by_model['Model']):
        ha = 'left' if text in label_left_lemma else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.savefig(imagedir / 'lemma_f1_speed.png')
    plt.close()

    sns.relplot(x='Lemmatization aligned accuracy', y='Tokens per second',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Lemmatization aligned accuracy'],
                          results_by_model['Tokens per second'],
                          results_by_model['Model']):
        ha = 'left' if text in label_left_lemma else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.savefig(imagedir / 'lemma_acc_speed.png')
    plt.close()

    label_left_pos = ['Voikko', 'stanza', 'spacy-lg']

    sns.relplot(x='Part-of-speech F1', y='Tokens per second',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Part-of-speech F1'],
                          results_by_model['Tokens per second'],
                          results_by_model['Model']):
        ha = 'left' if text in label_left_pos else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.savefig(imagedir / 'pos_f1_speed.png')
    plt.close()

    sns.relplot(x='Part-of-speech aligned accuracy', y='Tokens per second',
                data=results_by_model, s=80)
    for x, y, text in zip(results_by_model['Part-of-speech aligned accuracy'],
                          results_by_model['Tokens per second'],
                          results_by_model['Model']):
        ha = 'left' if text in label_left_pos else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.savefig(imagedir / 'pos_acc_speed.png')
    plt.close()


if __name__ == '__main__':
    main()
