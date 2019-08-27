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
import re

# print(string.punctuation)

def read_file(filepath, header=0):
    # fp = codecs.open(filepath,"r",encoding='utf-8', errors='ignore')
    # text = fp.read()
    # return text[:header], text[header:]
    file = open(filepath, 'r')
    file.seek(0)
    text = file.read()
    return text[:header], text[header:]

def sentence_tokenizer(text):
    tokens = sent_tokenize(text)
    return tokens

def word_tokenizer(text):
    words = word_tokenize(text)
    return words

def SentPreprocessing(text):
    text = text.strip()
    text = text.lower()
    # text = re.sub(r'[^\w]', ' ', text)
    text = re.sub(r'\d+.\s+', '', text)
    for i in string.punctuation:
        if (i not in ['.', '!', '?', "@"]):
            text = text.replace(i, ' ')
    text = re.sub(r'\d+', '', text)
    return text

def WordPreProcessing(text):
    text = re.sub(r'[^\w]', ' ', text)
    return text

if __name__ == "__main__":
    filepath = "./20_newsgroups/alt.atheism/49960"
    filepath = "./20_newsgroups/comp.graphics/37261"
    header, text = read_file(filepath, header=687)
    header = SentPreprocessing(header)
    text = SentPreprocessing(text)
    # print(header[:100])
    # print(text[:100])
    while(True):
        print("\n")
        print("1. No of sentences and No of words.")
        print("2. No of words starting with consonant and the No of words starting with vowel.")
        print("3. List of all the email ids in the file.")
        print("4. Given a word and a file as input, return the number of sentences starting with that word.")
        print("5. Given a word and a file as input, return the number of sentences ending with that word.")
        print("6. Given a word and a file as input, return the count of that word.")
        Option = int(input("Enter Input: "))

        if (Option == 1):
            sentences = sentence_tokenizer(text)
            for i in range(len(sentences)):
                print(sentences[i], i)
            print("Sentences count:", len(sentences))
            text = WordPreProcessing(text)
            words = word_tokenizer(text)
            print("Words count:", len(words))
        elif (Option == 2):
            words = word_tokenizer(text)
            vowels = ('a', 'e', 'i', 'o', 'u')
            consonants = ('b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z')
            consonantCount = 0
            vowelCount = 0
            for w in words:
                if (w.lower().startswith(consonants)):
                    consonantCount += 1
                elif (w.lower().startswith(vowels)):
                    vowelCount += 1
            print("Words starting with Consonants:", consonantCount)
            print("Words starting with Vowels:", vowelCount)
        elif (Option == 3):
            Emails = re.findall(r'[\w\.-]+@[\w\.-]+', header)
            Emails += re.findall(r'[\w\.-]+@[\w\.-]+', text)
            print("Email Count:", len(Emails))
            for email in Emails:
                print(email)
        elif (Option == 4):
            NewFilePath = str(input("Enter valid file path: "))
            Header = int(input("Enter header lenght in lines: "))
            FindWord = str(input("Enter word: ")).lower()
            header, text = read_file(NewFilePath, header=Header)
            text = SentPreprocessing(text)
            sentences = sentence_tokenizer(text)
            count = 0
            for i in sentences:
                sent = WordPreProcessing(i)
                words = word_tokenizer(sent)
                if (words[0] == FindWord):
                    count += 1
            print("Number of sentences starting with that word:", count)

        elif (Option == 5):
            NewFilePath = str(input("Enter valid file path: "))
            Header = int(input("Enter header lenght in lines: "))
            FindWord = str(input("Enter word: ")).lower()
            header, text = read_file(NewFilePath, header=Header)
            text = SentPreprocessing(text)
            sentences = sentence_tokenizer(text)
            count = 0
            for i in sentences:
                sent = WordPreProcessing(i)
                words = word_tokenizer(sent)
                if (words[len(words)-1] == FindWord):
                    count += 1
            print("Number of sentences ending with that word:", count)

        elif (Option == 6):
            NewFilePath = str(input("Enter valid file path: "))
            Header = int(input("Enter header lenght in lines: "))
            FindWord = str(input("Enter word: ")).lower()
            header, text = read_file(NewFilePath, header=Header)
            text = SentPreprocessing(text)
            text = WordPreProcessing(text)
            words = word_tokenizer(text)
            count = 0
            for w in words:
                if (w == FindWord):
                    count += 1
            print("Number of sentences contains that word:", count)
            


