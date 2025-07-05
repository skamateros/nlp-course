import csv
import os

class CategorizedCorpus:
    """
    A corpus loader that iterates over categorized text files.

    This class loads files from a given directory, supporting two main formats:
    1. Directory-based categories: If the directory contains subdirectories ending with 'pos' or 'neg',
        it yields the first line of each file in those subdirectories, labeled with the corresponding category.
    2. Tab-separated files: If a file does not match the above pattern, it is assumed to be a tab-separated file
        where each line contains a text and its category label.

    Args:
         directory (str): Path to the directory or file containing the categorized corpus.

    Attributes:
         files (list): List of file paths to be processed.

    Yields:
         tuple: A tuple (text, label), where 'text' is the content and 'label' is the category ('pos', 'neg', or as found in the file).
    """

    def __init__(self, directory: str):
        self.files = []
        try:
            for file in os.listdir(directory):
                if file.startswith('.'): continue
                self.files.append(os.path.join(directory, file))
        except NotADirectoryError:
            self.files.append(directory)

    def __iter__(self):
        for file in self.files:
            if file.endswith('pos'):
                for subfile in os.listdir(file):
                    with open(os.path.join(file, subfile), 'r') as fhandle:
                        text = fhandle.readlines()[0]
                        yield (text, 'pos')
            elif file.endswith('neg'):
                for subfile in os.listdir(file):
                    with open(os.path.join(file, subfile), 'r') as fhandle:
                        text = fhandle.readlines()[0]
                        yield (text, 'neg')
            else:
                with open(file, 'r') as fhandle:
                    reader = csv.reader(fhandle, delimiter='\t')
                    for line in reader:
                        yield (line[0], line[1])
