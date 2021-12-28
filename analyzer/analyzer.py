import pandas as pd
from spacy import displacy
import spacy
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import wordnet as wn

nltk.download('vader_lexicon')

class AnalyzeEngine():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()


    def clean_text(self, text, doLower=True):
        if doLower:
            text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        if doLower:
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub('[\W+]', ' ', text)
        return text.strip()

    def get_sentiment(self, text):
        return (self.sentiment_analyzer.polarity_scores(text)['compound'])

    def analyze(self, texts,categories=[]):
        result = []
        for ss in texts:
            phrases=[]
            if type(ss) != str:
                continue
            doc = self.nlp(ss.strip())
            for sent in doc.sents:
                processed_index = -1
                subjective = {}
                for token in sent:
                    if token.dep_ in ['dobj', 'nsubj', 'csubj'] and token.pos_ == 'ADJ':
                        subjective[token.text] = token.dep_ + "=>" + token.head.text
                for token in sent:
                    if (token.i <= processed_index):
                        #                            print(token.text, 'already processed')
                        continue
                    #                        print(token.text)
                    if token.pos_ == 'PUNCT':
                        continue
                    filter_chunk = ['NOUN', 'PROPN']
                    filtered_deps = ['npadvmod', 'aux', 'nummod']
                    A = None
                    M = None
                    l = 99999999
                    r = 0
                    extra = ''
                    rule = 0

                    ##
                    ## tenth aux ex sound is good
                    ##
                    if (A == None and M == None) and token.pos_ in ['NOUN', 'PRON', 'PROPN',
                                                                    'DET'] and token.dep_ == 'nsubj' and token.head.pos_ == 'AUX' and token.head.lemma_ == 'be':
                        l = 99999999
                        r = 0
                        A = token

                        l = A.i
                        r = A.i
                        for child in token.head.children:
                            if child.pos_ == 'ADJ':
                                M = child
                                l = min(l, M.i)
                                r = max(r, M.i)
                                filter_chunk.append('ADJ')
                                rule = 10

                    ##
                    ## 11th Noun (nsubj)=> VERB Ex: Sometime It Connects , Sometime it dosen't. or battery last long
                    ##
                    if (A == None and M == None) and token.pos_ in ['NOUN', 'PRON', 'PROPN',
                                                                    'DET'] and token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                        l = 99999999
                        r = 0
                        if (len(token.head.lemma_) > 2 and token.head.dep_ != 'xcomp' and (
                                token.pos_ != 'PRON' or token.text.lower() in ['it', 'that', 'those', 'this',
                                                                               'product'])):
                            A = token.head
                            M = token
                            l = min(A.i, M.i)
                            r = max(A.i, M.i)
                            for child in A.children:
                                if child.pos_ == 'ADV' and child.dep_ == 'advmod':
                                    filter_chunk.append('ADV')
                                    extra = child.lemma_
                                if child.dep_ == 'dep':
                                    extra = ''
                                    if 'ADV' in filter_chunk:
                                        filter_chunk.remove('ADV')
                                    break
                            filter_chunk.append('VERB')
                            # if 'NOUN' in filter_chunk:
                            #    filter_chunk.append('NOUN')
                            # if 'PROPN' in filter_chunk:
                            #    filter_chunk.append('PROPN')
                            rule = 11
                    ##
                    ## 12th Adjective or Adverb Root Ex : Quiet or Very Quiet
                    ##
                    if (A == None and M == None) and token.pos_ in ['ADV', 'ADJ'] and token.dep_ in ['ROOT',
                                                                                                     'conj']:
                        l = 99999999
                        r = 0
                        M = token
                        A = token
                        for child in A.children:
                            if child.pos_ in ["AUX", 'VERB'] and child.dep_ not in ['ccomp']:
                                M = child
                                filter_chunk.append(child.pos_)
                        filter_chunk.append(token.pos_)
                        l = min(A.i, M.i)
                        r = max(A.i, M.i)
                        rule = 12
                    ##
                    ## 13th Adjective Caluses motor burnt out
                    ##
                    if (A == None and M == None) and token.pos_ in ['VERB', 'AUX'] and token.dep_ == 'acl':
                        l = 99999999
                        r = 0
                        M = token
                        A = token.head
                        filter_chunk.append(token.pos_)
                        l = min(A.i, M.i)
                        r = max(A.i, M.i)
                        rule = 13

                        ##
                    ## 15th Noun (npadvmod)=>VERB Ex The last one I bought of these lasted three years (lasted three years)
                    ##
                    if (A == None and M == None) and token.pos_ in ['NOUN',
                                                                    'PROPN'] and token.dep_ == 'npadvmod' and token.head.pos_ in [
                        'VERB', 'AUX']:
                        l = 99999999
                        r = 0
                        M = token.head
                        A = token
                        filter_chunk.append(M.pos_)
                        if 'npadvmod' in filtered_deps:
                            filtered_deps.remove('npadvmod')
                        l = min(A.i, M.i)
                        r = max(A.i, M.i)
                        rule = 15

                    ##
                    ## 16th Noun (npadvmod)=>VERB Ex The lad three years)
                    ##
                    if (A == None and M == None) and token.pos_ in ['NOUN',
                                                                    'PROPN'] and token.dep_ == 'pobj' and token.head.text.lower() in [
                        'of', 'with'] and token.head.pos_ in ['ADP'] and token.head.dep_ in 'prep':
                        l = 99999999
                        r = 0
                        M = token.head.head
                        A = token
                        if token.head.text.lower() != 'of':
                            M = A
                        filter_chunk.append(M.pos_)
                        l = min(A.i, M.i)
                        r = max(A.i, M.i)
                        rule = 16

                    ##
                    ## 18th ADV (advmod)=>VERB ex: runs silently.
                    ##
                    if (
                            A == None and M == None) and not token.is_stop and token.dep_ == 'advmod' and token.head.pos_ in [
                        'VERB']:
                        l = 99999999
                        r = 0
                        M = token
                        A = token.head
                        filter_chunk.append(A.pos_)
                        filter_chunk.append(M.pos_)
                        l = min(M.i, A.i)
                        r = max(M.i, A.i)
                        while (M.dep_ == 'compound'):
                            M = M.head
                            l = min(M.i, l)
                            r = max(M.i, r)
                        rule = 18
                        ##
                    ## 19th ADV (advmod)=>VERB ex: runs silently.
                    ##
                    if (A == None and M == None) and token.pos_ in ['NOUN',
                                                                    'PROPN'] and token.dep_ == 'nmod' and token.head.pos_ in [
                        'NOUN', 'PROPN']:
                        l = 99999999
                        r = 0
                        M = token
                        A = token.head
                        l = min(M.i, A.i)
                        r = max(M.i, A.i)
                        rule = 19
                        ##
                    ## eighth verb (dobj)=> NOUN Example (lost connection)
                    ##
                    if (A == None and M == None) and token.pos_ == "NOUN" and token.dep_ in [
                        'dobj'] and token.head.pos_ in ['AUX', 'VERB']:
                        # print('token:',token.text,'=>',token.pos_)
                        # print('head:',token.head.text,'=>',token.head.pos_)
                        l = 99999999
                        r = 0
                        M = token.head
                        A = token

                        filter_chunk.append('VERB')
                        # for child in M.children:
                        #    if(child.dep_=='nsubj'):
                        #        A =child
                        l = min(M.i, A.i)
                        r = max(M.i, A.i)
                        rule = 8

                        ##
                    ## nineth compund nouns ** sentence should be subjective **
                    ##
                    if (A == None and M == None) and token.dep_ == 'compound':
                        l = 99999999
                        r = 0
                        M = token.head
                        A = token
                        l = min(M.i, A.i)
                        r = max(M.i, A.i)
                        while (M.dep_ == 'compound'):
                            M = M.head
                            l = min(M.i, l)
                            r = max(M.i, r)

                        while (A.dep_ == 'compound'):
                            A = A.head
                            l = min(A.i, l)
                            r = max(A.i, r)
                        rule = 9

                    ##
                    ## First Rule (amod) => NOUN or VERB
                    ##
                    if (A == None and M == None) and token.dep_ == "amod" and token.head.pos_ in ['NOUN', 'VERB',
                                                                                                  'ADJ', 'ADV']:
                        l = 99999999
                        r = 0
                        M = token
                        A = token.head
                        rule = 1
                        l = min(A.i, M.i)
                        r = max(A.i, M.i)
                        filter_chunk.append(A.pos_)
                        # filter_chunk.append(M.pos_)
                    # print(token,'=>','first rule',doc[l:r+1])
                    ##
                    ## 14th Adjective (nsubj) =>NOUN
                    ##
                    if (A == None and M == None) and token.pos_ in ['NOUN', 'PROPN',
                                                                    'DET'] and token.dep_ == 'nsubj':
                        l = 99999999
                        r = 0
                        M = token
                        A = token.head
                        filter_chunk.append(M.pos_)
                        l = min(A.i, M.i)
                        r = max(A.i, M.i)
                        rule = 14

                    if (A != None and M != None):
                        #  print(subtree_span.text, '|', subtree_span.root.text,'[',l,'-',r,']')
                        aspects = [A]
                        for child in A.children:
                            if child.dep_ == 'conj' and child.text.lower() in ['and', 'or']:
                                aspects.append(child)
                        ## Find compound root
                        root = A.lemma_
                        if (A.head and A.dep_ == "compound"):
                            root = doc[min(A.i, A.head.i):max(A.i, A.head.i) + 1].lemma_
                            r = max(r, A.head.i)
                            l = min(l, A.head.i)
                        left_root = ''
                        for child in A.children:
                            #  print('child',child.dep_)
                            if child.dep_ == 'compound':
                                left_root = left_root + " " + child.lemma_
                                r = max(r, child.i)
                                l = min(l, child.i)
                        root = (left_root.strip()) + " " + (root.strip())
                        #                           print('root', root)
                        for aspect in aspects:
                            subtree_span = doc[l:r + 1]
                            #                               print('subtree', subtree_span.text)
                            processed_index = max(processed_index, r)
                            # extract chunk
                            filtered = ''
                            res = ""
                            lemma = ""
                            shouldReplaceConj = True
                            for t in subtree_span:
                                if t.lemma_ == aspect.lemma_:
                                    shouldReplaceConj = False
                            for t in subtree_span:
                                # print(t,'=>',t.pos_,'|',t.dep_,'|',t.is_stop)
                                res += t.text + ' '
                                if t.pos_ in filter_chunk or t.dep_ == 'compound' or (
                                        t.dep_ == 'conj' and t.head.pos_ in filter_chunk):
                                    if t.dep_ in filtered_deps:
                                        continue
                                    if not t.is_stop:
                                        filtered += t.text + ' '
                                        lemma += t.lemma_ + " "
                                # else :
                                #     print('stop',t.text)
                            if shouldReplaceConj:
                                res = res.replace(A.lemma_, aspect.lemma_, 1)
                                filtered = filtered.replace(A.text, aspect.text, 1)
                                lemma = lemma.replace(A.lemma_, aspect.lemma_, 1)
                            res += " " + extra
                            filtered += " " + extra
                            lemma += " " + extra

                            # sentiment_chunk
                            scr = r
                            scl = l
                            if aspect.head:
                                scr = max(r, aspect.head.right_edge.i)
                                scl = min(l, aspect.head.left_edge.i)
                            if M.head:
                                scr = max(r, M.head.right_edge.i)
                                scl = min(l, M.head.left_edge.i)
                            sc = doc[scl:scr + 1].text

                            #                         print('root :', root, '=>', rule, ' -- ', res, '(', lemma, ')', subjective)
                            # print("res:",res)
                            p=self.clean_text(res)
                            phrases.append({
                                'sent': sent.text.strip(),
                                'phrase': p,
                                'category':self.get_category(p,categories),
                                'root': self.clean_text(root),
                                'sentiment': self.get_sentiment(sc),
                            })
                    else:
                        #                  print('no aspect :', sent, '|', token.text, token.pos_, token.dep_)
                        pass
                result.append(phrases)
        return result

    def find_lemmas(self,q):
        embeds = {}
        for t in self.nlp(q):
            if t.is_stop or t.pos_ == 'PUNCT':
                continue
            # if t.pos_ in ['ADJ'] and t.head:
            #    continue;
            text = t.lemma_
            embeds[text] = [text]
            ls = []
            if True:
                try:
                    synsets = wn.synsets(text)
                    for s in synsets:
                        embeds[text] += s.lemma_names()
                except:
                    pass
            embeds[text] = [self.nlp(x.replace('_', ' ')) for x in list(set(embeds[text]))]

        return embeds

    def get_category(self,phrase,categories):
        if not categories:
            return "None"
        p = self.nlp(phrase)
        cs = [self.find_lemmas(c) for c in categories]
        res= "None"
        max=-2

        for cc in cs:
            for key,lemmas in cc.items():
                for c in lemmas:
                    r = p.similarity(c)
                    if r>max:
                        max=r
                        res=key
        return res if max>0.2 else "None"

