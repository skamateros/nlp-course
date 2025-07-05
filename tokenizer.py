from nltk import regexp_tokenize

def tokenize(text: str) -> list[str]:
    # Taken from Jurafsky and Martin (2024, Section 2.5.1)
    pattern = r'''(?x) # set flag to allow verbose regexps
    (?:[A-Z]\.)+ # abbreviations, e.g. U.S.A.
    | \w+(?:-\w+)* # words with optional internal hyphens
    | [\$£€]?\d+(?:\.\d+)?%? # currency, percentages, e.g. $12.40, 82%
    | \.\.\. # ellipsis
    | [][.,;"'’?():_‘-] # these are separate tokens; includes ], [
    '''
    return regexp_tokenize(text, pattern)
