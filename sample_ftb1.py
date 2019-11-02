import hashlib
import re


def main():
    inputfile = 'data/test/ftb1/ftb1u.tsv'
    outputfile = 'data/test/ftb1/ftb1u_sample.tsv'
    with open(inputfile) as f:
        sentences = list(split_sentences(f.readlines()))

    selected = []
    for sentence in sentences:
        digest = hashlib.sha256(sentence.encode('utf-8')).hexdigest()
        if digest.startswith('0'):
            selected.append(sentence)

    print(f'Sampled {len(selected)} out of {len(sentences)} sentences')

    with open(outputfile, 'w') as f:
        f.write('\n'.join(selected))

    print(f'Wrote samples to {outputfile}')


def split_sentences(lines):
    start_re = re.compile(r'^#\d+$')
    
    sentence = []
    for line in lines:
        if start_re.match(line):
            if sentence:
                yield ''.join(sentence)
            sentence = []

        if line.strip():
            sentence.append(line)

    if sentence:
        yield ''.join(sentence)


if __name__ == '__main__':
    main()
