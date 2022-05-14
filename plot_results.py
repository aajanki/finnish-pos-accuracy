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
            'turku-neural-parser': 'Turku pipeline',
            'udpipe-fi-tdt': 'UDPipe',
            'spacy-fi_core_news_sm': 'spacy-sm',
            'spacy-fi_core_news_md': 'spacy-md',
            'spacy-fi_core_news_lg': 'spacy-lg',
            'spacy-fi_experimental_web_md': 'spacy-experimental',
            'finnpos': 'FinnPos',
            'uralicnlp': 'UralicNLP',
            'stanza': 'Stanza',
            'trankit': 'Trankit',
            'voikko': 'Voikko',
            'raudikko': 'Raudikko',
        }))
    full_results = full_results.rename(columns={
        'UPOS F1': 'Part-of-speech F1',
        'UPOS precision': 'Part-of-speech precision',
        'UPOS recall': 'Part-of-speech recall',
        'UPOS aligned accuracy': 'Part-of-speech aligned accuracy',
    })

    # Leave out the FinnPos evaluation on ftb1u, because ftb1u is the
    # training set for FinnPos
    full_results = full_results[~((full_results['Model'] == 'FinnPos') &
                                (full_results['Dataset'] == 'ftb1u'))]

    results_by_testset = full_results[full_results['Dataset'] != 'concatenated']
    results_concatenated = full_results[full_results['Dataset'] == 'concatenated']

    model_order = sorted(results_by_testset['Model'].unique())

    sns.barplot(x='Model', y='Lemmatization F1', hue='Dataset',
                data=results_by_testset, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_f1_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization precision', data=results_by_testset,
                hue='Dataset', order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_precision_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization recall', data=results_by_testset,
                hue='Dataset', order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_recall_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization aligned accuracy', data=results_concatenated, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_acc.png')
    plt.close()

    sns.barplot(x='Model', y='Lemmatization F1', data=results_concatenated, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'lemma_f1.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech F1', hue='Dataset',
                data=results_by_testset, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_f1_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech precision', hue='Dataset',
                data=results_by_testset, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_precision_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech recall', hue='Dataset',
                data=results_by_testset, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_recall_by_dataset.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech aligned accuracy', data=results_concatenated, order=model_order)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_acc.png')
    plt.close()

    sns.barplot(x='Model', y='Part-of-speech F1', data=results_concatenated, order=model_order)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(imagedir / 'pos_f1.png')
    plt.close()

    label_left_lemma = ['Stanza', 'simplemma', 'spacy-lg', 'Raudikko']

    sns.relplot(x='Lemmatization F1', y='Tokens per second',
                data=results_concatenated, s=80)
    for x, y, text in zip(results_concatenated['Lemmatization F1'],
                          results_concatenated['Tokens per second'],
                          results_concatenated['Model']):
        ha = 'left' if text in label_left_lemma else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(imagedir / 'lemma_f1_speed.png')
    plt.close()

    label_left_lemma = ['FinnPos', 'simplemma', 'spacy-lg', 'Raudikko']

    sns.relplot(x='Lemmatization aligned accuracy', y='Tokens per second',
                data=results_concatenated, s=80)
    for x, y, text in zip(results_concatenated['Lemmatization aligned accuracy'],
                          results_concatenated['Tokens per second'],
                          results_concatenated['Model']):
        ha = 'left' if text in label_left_lemma else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(imagedir / 'lemma_acc_speed.png')
    plt.close()

    label_left_pos = ['Voikko', 'Raudikko', 'Stanza', 'spacy-lg']

    sns.relplot(x='Part-of-speech F1', y='Tokens per second',
                data=results_concatenated, s=80)
    for x, y, text in zip(results_concatenated['Part-of-speech F1'],
                          results_concatenated['Tokens per second'],
                          results_concatenated['Model']):
        ha = 'left' if text in label_left_pos else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(imagedir / 'pos_f1_speed.png')
    plt.close()

    sns.relplot(x='Part-of-speech aligned accuracy', y='Tokens per second',
                data=results_concatenated, s=80)
    for x, y, text in zip(results_concatenated['Part-of-speech aligned accuracy'],
                          results_concatenated['Tokens per second'],
                          results_concatenated['Model']):
        ha = 'left' if text in label_left_pos else 'right'
        textrelx = 7 if ha == 'left' else -5
        plt.annotate(text, (x, y), xytext=(textrelx, 0), textcoords='offset points',
                     horizontalalignment=ha, verticalalignment='center')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(imagedir / 'pos_acc_speed.png')
    plt.close()

    # For comparing with the results published by Stanza, Trankit and Spacy
    print('Results on UD_Finnish_TDT:')
    ud_results = results_by_testset[results_by_testset['Dataset'] == 'UD_Finnish_TDT']
    print(ud_results[['Model', 'Dataset', 'Part-of-speech F1', 'Lemmatization F1']]
          .sort_values('Model')
          .to_string(index=False))


if __name__ == '__main__':
    main()
