'''
------------------------------------------------------
@Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
@Roll-No: 2016048
@Date: Thursday August 22nd 2019 7:20:56 pm
------------------------------------------------------
'''

import codecs
import string
from nltk.tokenize import sent_tokenize, word_tokenize


def read_file(filepath, header=0):
    '''
    @params: filepath, header size in no of characters
    @return: returns header text of the file and also the remaining text.
    '''
    fp = codecs.open(filepath,"r",encoding='utf-8', errors='ignore')
    text = fp.read()
    return text[:header], text[header:]

def sentence_tokenizer(text):
    tokens = sent_tokenize(text)
    return tokens


if __name__ == "__main__":
    filepath = "./20_newsgroups/alt.atheism/49960"
    header, text = read_file(filepath, header=1019)
    
    sentences = sentence_tokenizer(text)
    print("Sentences count:", len(sentences))




