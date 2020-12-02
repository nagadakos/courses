import sys
import os
from pathlib import Path
from os.path import isdir, join, isfile
from os import listdir
import fnmatch
import re
from dateutil.parser import parse
import spacy
import en_core_web_sm
import unidecode
import string
from word2number import w2n
from pycontractions import Contractions
import gensim.downloader as api
# Multithreading libs
import queue
import threading # fun fact: Python dfoes not actually allow threads to run on multiple cores due to Global Interpret Lock (GIL)
from multiprocessing import Process, Queue # use multiprocessing instead
import time
import pandas as pd  # import pandas to read xlsx files
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path)

# ----------------------------------------------------------------------------------------------------


def get_files_from_path(targetPath, expression, excludePattern = 'dumz'):

    # Find all folders that are not named Solution.
    d = [f for f in listdir(targetPath) if (isdir(join(targetPath, f)) and "Solution" not in f)]
    # Find all file in target directory that match expression
    f = [f for f in listdir(targetPath) if (isfile(join(targetPath, f)) and fnmatch.fnmatch(f,expression) and excludePattern not in f)]
    # initialize a list with as many empty entries as the found folders.
    l = [[] for i in range(len(d))]
    # Create a dictionary to store the folders and whatever files they have
    contents = dict(zip(d,l))
    contents['files'] = f

    # Pupulate the dictionary with files that match the expression, for each folder.
    # This will consider all subdirectories of target directory and populate them with
    # files that match the expression.
    for folder, files in contents.items():
        stuff = sorted(Path(join(targetPath, folder)).glob(expression))
        for s in stuff:
            files.append(os.path.split(s)[1] )
        # print(folder, files)
    for files in contents['files']:
        stuff = sorted(Path(join(targetPath, files)).glob(expression))
    # print(contents)
    return contents

# ---------------------------------------------------------------

def clear_relics(files):
    """ Description: This function will clear all relics from a text. Relics are considered to be any characters
                     that are not: in the Aa-Zz range, numbers, [],(), panctuation, space, new line, apostrophes, quotations.
    """

    if not isinstance(files, list):
        files = [files]

    apoPattern = '&apos;'
    rndPattern = '&#160;'
    spcPattern = r'[^a-zA-Z0-9\?\!\;\.\,\-\:\/\'\[\]\(\)\n\t ]'
    tabPattern = r'\t'
    rLine = ''
    for f in files:
        f2 = f.split('.')[0]+'_corrected.txt' # in name is patientx.txt -> patientx_corrected.txt
        with open(f, 'r') as fr, open(f2,'w+') as fw:
            for cnt, line in enumerate(fr):
                rLine = re.sub(apoPattern, '\'', line) # substitube weirt pattern for ' with actual '
                rLine = re.sub(rndPattern, '', rLine) # substitube weirt pattern for ' with actual '
                rLine = re.sub(spcPattern, '', rLine) # substitube weirt pattern for ' with actual '
                rLine = re.sub(tabPattern, ' ', rLine) # substitube tabs with a single space
                print(line,rLine)
                fw.write(rLine)
                

# ---------------------------------------------------------------
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    # try: 
        # parse(string, fuzzy=fuzzy)
        # return True

    # except ValueError:
        # return False
    
    datePattern = r'[0-9]+\/[0-9]+\/[0-9]{2,4}'
    if re.search(datePattern, string):
        #print(string,'is Date')
        return True
    else:
        return False
# ---------------------------------------------------------------
def is_time(string):
    """ Description: Simple function to parse if a token is time. Some tokenizers
                     parse xx:xx as one token and then the code might have time discerning
                     what kind of word this is.
    """
    timePattern = r'\b[0-9]{1,2}[\:?\.?]{1}[0-9]{1,2}\b'
    if re.search(timePattern, string):
        #print(string,'is Time')
        return True
    else:
        return False
# ---------------------------------------------------------------
def remove_accented_chars(text):
    """remove accented characters from text, e.g. cafe"""
    text = unidecode.unidecode(text)
    return text
# ---------------------------------------------------------------
def expand_contractions(text):
    # Choose model accordingly for contractions function
    model = api.load("glove-twitter-25")
    # model = api.load("glove-twitter-100")
    # model = api.load("word2vec-google-news-300")
    cont = Contractions(kv_model=model)
    cont.load_models()
    """expand shortened words, e.g. don't to do not"""
    text = list(cont.expand_texts([text], precise=True))[0]
    return text
# ---------------------------------------------------------------
def remove_number_words(text):
    doc = nlp(text)
    tokens = [w2n.word_to_num(token.text) if token.pos_ == 'NUM' else token for token in doc]
# ---------------------------------------------------------------
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
# ---------------------------------------------------------------
# ---------------------------------------------------------------
def normalize_texts(texts, inTextLabel = '', outLabel = 'normalized', 
                    steps = {'rmvStopWords':True, 'rmvAccents' : True, 'expandContractions' : True, 'rmvNumberWords' : True,'rmvCustom':False,
                             'removeDate':False,'rmvSpecialChars':True,'rmvPunctuation':True, 'rmvNumbers':False, 'lemmatization':True,
                              'rmvSmileys':True, 'rmvURL':True}):
    """ Description: This function normilizes text to the basic NLP standard. By default it
                     it will perform all the preprocessing steps, providing succeeding steps
                     will standardized input. It can save the resulting text to disk, to eliminate
                     future computational waste. Resulting text might not be very "humanesque"...
    """
    # TODO: Do the steps one by one.
    # nlp = spacy.load('en_core_web_md')
    # load core nlp object to do tokenization,lemmatization etc
    nlp = en_core_web_sm.load()
    # Deselectors
    deselect_punct = []#['.']
    customList = ["'s", '%', '&']
    smiley = [';(', ':)', ':(']
    select_stop_wrods = ['to']
    # Sanitize input to iterate over given files. Works for just one file too!
    if not isinstance(texts, list):
        texts = [texts]
    # IF we need to remove stop words, there might be a few words that we don't want
    # to remove, as they might actually be informative. This is handled here.
    if steps['rmvStopWords']:
        # exclude words from spacy stopwords list
        deselect_stop_words = ['no', 'not']
        for w in deselect_stop_words:
            nlp.vocab[w].is_stop = False
    if steps['rmvPunctuation']:
        for p in deselect_punct:
            nlp.vocab[p].is_punct = False
            
    if steps['expandContractions']:
        # Choose model accordingly for contractions function
        model = api.load("glove-twitter-25")
        # model = api.load("glove-twitter-100")
        # model = api.load("word2vec-google-news-300")
        cont = Contractions(kv_model=model)
        cont.load_models()
    for i, t in enumerate(texts):
        if os.path.isfile(t):
            f2 = t.rsplit('_', 2)[0]+'_' +outLabel +'.txt' # in name is patientx.txt -> patientx_corrected.txt
            fr, fw = open(t, 'r', encoding='utf8'), open(f2,'w+', encoding='utf8')
        else:
            f2 = inTextLabel+'_' +outLabel +'.txt' # in name is patientx.txt -> patientx_corrected.txt
            fr, fw = t, open(f2, 'w+', encoding='utf8') 
        text = fr.read()
        # Regular text normlization
        if steps['rmvAccents']:
            text = remove_accented_chars(text)
        if steps['expandContractions']:
            """expand shortened words, e.g. don't to do not"""
            text = list(cont.expand_texts([text], precise=True))[0]
            #text = expand_contractions(text)

        # Tokenize and normilize tokens
        doc = nlp(text) #tokenise text

        clean_text = []
        docLen = len(doc)
        for i,token in enumerate(doc):
            flag = True
            isEntity = 0
            edit = token.text.lower()
            edit = edit.lower()
            #if i < 100:
                #print('Initial edit',edit)
            # Check if token is date...
            tokenIsDate, tokenIsTime = 0,0
            if is_date(token.text):
                tokenIsDate = 1 
                edit = token.text
            if is_time(token.text):
                tokenIsTime = 1
                edit = token.text
            
            if steps['removeDate'] == True and tokenIsDate:
                flag = False
            
            # Remove URLS
            if (steps['rmvURL'] == True) and (re.match(r'^https?:\/\/.*[\r\n]*', edit) is not None):
                flag = False
            # Remove twitters <e>
            if (re.match(r'<', edit) is not None) or (re.match(r'>', edit) is not None) or (re.match(r'e', edit) is not None):
                #print(edit)
                flag = False
            # Remove tweet replies
            if '@' in edit:
                flag = False
            # Remove </e artifacts
            elif '</e' in edit:
                edit = edit.split('<')[0]
                #print('</e found', edit)
                isEntity = 1
            elif '</a' in edit:
                edit = edit.split('<')[0]
                isEntity = 1
                
            if tokenIsDate == 0 and flag == True and isEntity == 0:
                # remove stop words
                if steps['rmvStopWords'] == True and token.is_stop and token.pos_ != 'NUM': 
                    flag = False
                # remove punctuations
                elif steps['rmvPunctuation'] == True and token.pos_ == 'PUNCT' and flag == True: 
                    if token.text not in deselect_punct:
                        flag = False
                # remove special characters
                elif steps['rmvSpecialChars'] == True and token.pos_ == 'SYM' and flag == True and tokenIsDate == 0: 
                    flag = False
                # remove special characters
                elif steps['rmvSmileys'] == True and edit in smiley: 
                    flag = False
                # remove numbers
                elif steps['rmvNumbers'] == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
                and flag == True:
                    flag = False
                # convert number words to numeric numbers
                elif steps['convNumberWords'] == True and token.pos_ == 'NUM' and flag == True and tokenIsTime == 0:
                    try:
                        edit = w2n.word_to_num(token)
                    except:
                        pass
                        #print(token,t,  "Dont know how to convert this into a number!")
                # convert tokens to base form
                elif steps['lemmatization'] == True and token.lemma_ != "-PRON-" and flag == True:
                    edit = token.lemma_.lower()
                    #if i < 100:
                        #print("In lemma: ", edit)
                elif steps['rmvCustom'] == True and token.text in customList:
                    flag = False
                # remove <e> stuff from tweeter msgs
                elif (edit == '<') or (edit == ">") or (edit == 'e'):
                    flag = False
                

            # append tokens edited and not removed to list 
            # print('edit is', str(edit), flag)
            if edit != "" and flag == True:
                #if i < 100:
                    #print("Before list write", edit)
                if i+1 < docLen:
                    if (doc[i+1].text in string.punctuation and steps['rmvPunctuation'] is False) or (doc[i+1].text in deselect_punct):
                        clean_text.append(str(edit))        
                    elif edit == '#':
                        clean_text.append(str(edit))        
                    else:
                        clean_text.append(str(edit) + " ")        
            # Dump clean text to file
        # print(clean_text)
        fw.writelines(clean_text)
        # Close files
        if os.path.isfile(t):
            fr.close()
        fw.close()
        
    return True

# Round 2 specific functions
def get_target_cols(df, target, saveFile=None):
    dfNew = df[target]
    if saveFile is not None:
        writer = ExcelWriter(saveFile)
        dfNew.to_excel(writer,'Sheet1',index=False)
        writer.save()
    return dfNew

# -------------------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------------------
def process_data(file = 'Data/training-Obama-Romney-tweets.xlsx', obamaCentric = False, sheet = 0):
    
    rootFile = file.split('.')[0]
    mod = 'Romney_to_Obama_' if (sheet == 1 or sheet == 'Romney') else ''
    mod = 'Romney_to_Obama_' if obamaCentric else ''
    dataFile = '_'.join((rootFile, mod+'filtered_data.txt'))
    labelFile = '_'.join((rootFile, mod+'filtered_labels.txt'))
    
    print(dataFile)
    if os.path.isfile(dataFile) and os.path.isfile(labelFile):
        with open(labelFile) as f:
            labels = f.readlines()
    else:
        df = pd.read_excel(file, sheet_name = sheet)
        # Extract only the tweets and labels; dates and the rest are irrelevant
        # Label Legend: 1: positive, -1: negative, 0: neutral, 2: mixed
        data = df['Anootated tweet'][1:]
        labels2 = df['Unnamed: 4'][1:].values.tolist()
        labels2 = [str(l) for l in labels2]
        for i, l in enumerate(labels2):
            try:
                if type(data.iloc[i]) != float:
                    labels2[i] = int(l)
                else:
                    labels2[i] = 5 # for that annoyiing nan data case
            except:
                labels2[i] = 5
            #if (l != 'nan') & (l != 'irrelevant') & (l !='irrevelant') & (type(data.iloc[i]) != float) &(l != '!!!!'):
            #    print(l)
            #    labels2[i] = int(l)
            #elif (l == 'irrelevant') or (l =='irrevelant') or (l == 'nan') or (type(data.iloc[i]) == float) or (l == '!!!!'):
            #    labels2[i] = 5

        labels = np.asarray(labels2)
        
        # Diseregard all tweets of mixed type, that is class 2
        idxs = np.where((labels != 2) & (labels != 5))[0]
        data = data.iloc[idxs].values.tolist()
        
        # Flip labels so positive refers to Obama always
        if obamaCentric:
            labels = labels[idxs] * -1
        else:
            labels = labels[idxs]
        labels = list(labels)
        
        # Clear tweeter relics
        for i, l in enumerate(data):
            data[i] = re.sub(r'(<e>|</e>|<a>|</a>)', '', l)
        
        with open(dataFile, 'w', encoding='utf-8') as f:
            for i, d in enumerate(data):
                # For a multi lined tweet just keep the first line.   
                f.write("{} \n".format(d.split('\n')[0]))
        with open('_'.join((rootFile, mod+'filtered_data_idxs.txt')), 'w', encoding='utf-8') as f:
            for i, d in enumerate(data):
                f.write("{}\n".format(idxs[i]))
        with open(labelFile, 'w', encoding='utf-8') as f:
            for i, l in enumerate(labels):
                f.write("{} \n".format(l))
    
    # removing Accents and expanding contractions takes time!
    steps = {'rmvStopWords':True, 'rmvAccents' : True, 'expandContractions' : False, 'convNumberWords' : False, 'rmvCustom':True
             ,'removeDate':False,'rmvSpecialChars': True, 'rmvPunctuation':True, 'rmvNumbers':True, 'lemmatization':True,
             'rmvSmileys':True, 'rmvURL':True}
    # ---|
    
    # Normalize texts
    outLabel = mod + 'corrected2_normalized_no_stop_words' 
    normalize_texts('_'.join((rootFile, mod+'filtered_data.txt')), outLabel= outLabel,steps = steps)
    with open('_'.join((rootFile, outLabel, 'labels.txt')), 'w', encoding='utf-8') as f:
            for i, l in enumerate(labels):
                f.write("{} \n".format(str(l).strip('\n')))
                #np.savetxt(f, l)
   
# -------------------------------------------------------------------

if __name__ == '__main__':
    process_data()
