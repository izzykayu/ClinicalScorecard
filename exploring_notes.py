import pandas as pd

import glob
import unicodedata
import re
import csv
import numpy as np
import pprint
import os

import string

all_letters = string.ascii_letters + " .,;':"
n_letters = len(all_letters)
print(all_letters)
entity_dict = {"MEDICATION": "Drug",
               "FREQUENCY": "Frequency",
               "ROUTE_OR_MODE": "Route",
               "DOSAGE":"Dosage",
               "STRENGTH": "Strength",
              "FORM": "Form",
              "DURATION": "Duration"}

entities_re = re.compile('(%s)' % '|'.join(entity_dict.keys()))
def replace_entities_n2c2(s):
    def replace(match):
        return entity_dict[match.group(0)]
    return entities_re.sub(replace, s)


def find_files(path):
    """
    returns a list of files in a path
    """
    return glob.glob(path)

def flatten_json(y):
    """
    :param y: the json you want to flatten
    :return: flattened json
    """

    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def flatten_list(l):
    """
    input is list [[ ]]
    returns flattened list
    """
    return [item for sublist in l for item in sublist]

def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
        )


def normalizeStringAndDigitsLower(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.:!?])", r" \1 ", s)
    s = re.sub(r"[\|]+", r" ", s)
    s = re.sub('\n', '<NEWLINE>', s)
    s = re.sub('\t', ' ', s)
    s = re.sub('\d+', '#', s)
    s = re.sub(' +', ' ', s)
    return s


def build_vocab(vocab_min, infile, vocab_filename):
    """
        INPUTS:
            vocab_min: how many documents a word must appear in to be kept
            infile: (training) data file to build vocabulary from
            vocab_filename: name for the file to output
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header
        next(reader)

        # 0. read in data
        print("reading in data...")
        # holds number of terms in each document
        note_numwords = []
        # indices where notes start
        note_inds = [0]
        # indices of discovered words
        indices = []
        # holds a bunch of ones
        data = []
        # keep track of discovered words
        vocab = {}
        # build lookup table for terms
        num2term = {}
        # preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            text = row[2]
            numwords = 0
            for term in text.split():
                # put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            # record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            # go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        # clip trailing zeros
        note_occur = note_occur[note_occur > 0]

        # turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word, ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        # 1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        # also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))

        # 2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        # inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        # drop those rows
        C = C[inds, :]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        print("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")


def letterToIndex(letter):
    """
    find index for all letters
    """
    return all_letters.find(letter)


# def letterToTensor(letter):
#     """
#     Just for demonstration, turn a letter into a <1 x n_letters> Tensor
#     """
#     tensor = torch.zeros(1, n_letters)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor
#
#
# def lineToTensor(line):
#     """# Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors"""
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][letterToIndex(letter)] = 1
#     return tensor


all_CNER_words = {}
all_CNER_labels = []


# Read a file and split into lines
def read_norm_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line.split('||') for line in lines]

def make_dictionary_annotations_and_text(name):
    text_file_path = str(name) + ".txt"
    text_norm_path = str(name) + ".norm"
    note_text = open(os.path.join('/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_note', text_file_path)).read()
    list_norm_annotations = read_norm_lines(os.path.join('/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_norm', text_norm_path))
    annotation_dictionary = {'id': [], 'concept': [],
                             'beginCharacterOffset': [],
                             'endCharacterOffset': [],
                             'text': [],
                             'disjointed_concept': [],
                             'additionalBeginCharacterOffset': [],
                             'additionalEndCharacterOffset': [],
                             'additionalText': []}
    for list_norm in list_norm_annotations:
        annotation_dictionary['id'].append(list_norm[0])
        annotation_dictionary['concept'].append(list_norm[1])
        annotation_dictionary['beginCharacterOffset'].append(list_norm[2])
        annotation_dictionary['endCharacterOffset'].append(list_norm[3])
        annotated_text = note_text[int(list_norm[2]): int(list_norm[3])]
        annotation_dictionary['text'].append(annotated_text)
        if len(list_norm) < 5:
            annotation_dictionary['disjointed_concept'].append(None)
            annotation_dictionary['additionalBeginCharacterOffset'].append(None)
            annotation_dictionary['additionalEndCharacterOffset'].append(None)
            annotation_dictionary['additionalText'].append(None)
        else:
            annotation_dictionary['disjointed_concept'].append('Y')
            annotation_dictionary['additionalBeginCharacterOffset'].append(list_norm[4])
            annotation_dictionary['additionalEndCharacterOffset'].append(list_norm[5])
            annotation_dictionary['additionalText'].append(note_text[int(list_norm[4]): int(list_norm[5])])

    return pd.DataFrame(annotation_dictionary)[['concept',
                             'beginCharacterOffset',
                             'endCharacterOffset',
                             'text',
                             'disjointed_concept',
                             'additionalBeginCharacterOffset',
                             'additionalEndCharacterOffset',
                             'additionalText']] #.to_dict('records')# list_norm_annotations


# pprint.pprint(make_dictionary_annotations_and_text('0038').to_dict('split'))

example_list_norm_annot = read_norm_lines(filename='/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_norm/0214.norm')

def count_CUI(list_norm_annotations):
    """
    this function takes a list of lists (with the norm annotations) and returns stats
    :param list_norm_annotations: '
    :return: counts of CUI, counts of CUI_less, counts of disjointed_CUI
    """
    counter = 0
    disjointed_cui_counter = 0
    cui_less_counter = 0
    for line in list_norm_annotations:
        counter += 1
        if line[1]  == 'CUI-less':
            cui_less_counter += 1
        if len(line) > 4:
            disjointed_cui_counter += 1

    return counter,cui_less_counter, disjointed_cui_counter




def return_eda_stats(file_list_path):
    """

    :param file_list_path: pathname of file containing a list of core-names of .txt/text.norm files (for example:
    :return: a pandas data-frame with counts from each file
    """
    file_list_text = open(file_list_path, encoding='utf-8').read().strip().split('\n')
    path_name_main_list = [str(name_in_list) for name_in_list in file_list_text]
    eda_stats_dictionary = {'file_name_main': [],
                            'note_lines_count': [],
                            'CUI_count': [],
                            'CUI_less_count': [],
                            'disjointed_CUI_count': []
                            }
    for path_name_main in path_name_main_list:
        eda_stats_dictionary['file_name_main'].append(path_name_main)
        text_file_path = str(path_name_main) + ".txt"
        text_norm_path = str(path_name_main) + ".norm"
        note_text_lines = open(os.path.join('/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_note',
                                      text_file_path),  encoding='utf-8').read().strip().split('\n')
        eda_stats_dictionary['note_lines_count'].append(str(len(note_text_lines)))
        list_norm_annotations = read_norm_lines(os.path.join('/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_norm', text_norm_path))
        eda_stats_dictionary['CUI_count'].append(count_CUI(list_norm_annotations)[0])
        eda_stats_dictionary['CUI_less_count'].append(count_CUI(list_norm_annotations)[1])
        eda_stats_dictionary['disjointed_CUI_count'].append(count_CUI(list_norm_annotations)[2])

    return pd.DataFrame(eda_stats_dictionary)



def calculate_average(*args):
    '''
    Function that accept variable length arguments
    :param args: numbers
    :return: average of numbers passed through
    '''
    num = len(args)
    if num == 0:
        return 0
    sum_of_numbers = 0
    for elem in args:
        sum_of_numbers += elem
    return sum_of_numbers / num

eda_stats_train = return_eda_stats('/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_file_list.txt')
eda_stats_train.to_csv('eda_stats_train.csv', index=False)



#
# def amazon_convert_n2c2(amazon_output_list, outfile_path):
#     counter = 0
#     with open(outfile_path,'w') as f:
#         for x in flatten_list(example_list):
#             counter += 1
#             new_line =  "T" + str(counter) + "\t" + replace_entities_n2c2(x['Category']) + "\t" + str(x['BeginOffset']) + "\t" +  str(x['EndOffset']) + "\t" + x['Text']
#             print(new_line)
#             f.write(new_line + os.linesep)
#             if x.get('Attributes') != None:
#                 attribute_list = x.get('Attributes')
#                 for a in attribute_list:
#                     counter += 1
#                     a_line = "T" + str(counter) + "\t" + replace_entities_n2c2(a['Type']) + "\t" + str(a['BeginOffset']) + "\t" +  str(a['EndOffset']) +"\t" + re.sub("\n", " ",
#                     a['Text'])
#                     print(a_line)
#                     f.write(a_line + os.linesep)
#         print(counter)

# pprint.pprint(read_norm_lines('/Users/isabelmetzger/PycharmProjects/ClinicalScorecard/data/train/train_norm/0070.norm'))