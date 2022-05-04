import logging
import conll18_ud_eval
import typer
from pathlib import Path
from datasets import gold_path

FORM = 1
LEMMA = 2


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    resultdir = Path('results')
    errordir = resultdir / 'errors'
    errordir.mkdir(parents=True, exist_ok=True)
    predictionsdir = resultdir / 'predictions'
    model_result_dirs = predictionsdir.glob('*')

    for model_result_dir in model_result_dirs:
        model_name = model_result_dir.stem
        testset_files = model_result_dir.glob('*.conllu')
        for predictions_file in testset_files:
            testset_name = predictions_file.stem

            gold_ud = conll18_ud_eval.load_conllu_file(gold_path(testset_name))
            system_ud = conll18_ud_eval.load_conllu_file(predictions_file)

            output_file = errordir / f'lemma_errors_{model_name}_{testset_name}.txt'
            print(f'Writing errors in {output_file}')
            with open(output_file, 'w') as f:
                for system_sentence, gold_sentence in zip(system_ud.sentences, gold_ud.sentences):
                    system_words = [
                        w for w in system_ud.words
                        if w.span.start >= system_sentence.start and w.span.end <= system_sentence.end
                    ]
                    system_lemmas = [
                        x.columns[LEMMA].lower() for x in system_words if not x.is_multiword
                    ]

                    gold_words = [
                        w for w in gold_ud.words
                        if w.span.start >= gold_sentence.start and w.span.end <= gold_sentence.end
                    ]
                    gold_lemmas = [
                        x.columns[LEMMA].lower() for x in gold_words if not x.is_multiword
                    ]
                    gold_forms = [
                        x.columns[FORM].lower() for x in gold_words if not x.is_multiword
                    ]

                    if system_lemmas != gold_lemmas:
                        f.write(' '.join(gold_forms) + '\n')
                        f.write(' '.join(gold_lemmas) + '\n')
                        f.write(' '.join(system_lemmas) + '\n')
                        f.write('\n')


if __name__ == '__main__':
    typer.run(main)
