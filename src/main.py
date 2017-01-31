#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import argparse as ap
import textwrap

import pickle as pkl
import gzip

import keras
from keras.models import load_model
import numpy as np

description = '''\
Ceci est un programme qui cherche à identifier la relation binaire
reliant deux noms dans le contexte d'une phrase.

--------------------------------------------------------------------------

Le résultat peut être l'une de ces relations binaires,
ainsi que leur symétries :
    - Cause-Effet(e1,e2) : relation de cause à effet entre deux mots
    - Outil-Agent(e1,e2) : agent utilisant un outil, un instrument
    - Produit-Fabricant(e1,e2) : produit conçu par un fabricant
    - Contenant-Contenu(e1,e2) : contenant contenant un contenu (oui)
    - Entité-Origine(e1,e2) : entité provenant d'une source, géographique
ou physique
    - Entité-Destination(e1,e2) : entité se dirigeant vers une destination
    - Composant-Catégorie(e1,e2) : composant d'une catégorie plus globale
    - Membre-Collection(e1,e2) : membre d'un groupe d'objet de même sorte
    - Message-Sujet(e1,e2) : message dont le thème est <e2>
    - Autre(e1,e2) (cette relation est symétrique)

--------------------------------------------------------------------------
'''

parser = ap.ArgumentParser(formatter_class=ap.RawDescriptionHelpFormatter,
description=textwrap.dedent(description),
epilog= "Matthieu RÉ, Yaohui WANG")
parser.add_argument("sentence",
                    help="La phrase dans laquelle on veut trouver une relation")
parser.add_argument("-e1","--first_term", type=int,
                    help="La position du premier terme")
parser.add_argument("-e2","--second_term", type=int,
                    help="La position de second terme")
parser.add_argument("-v","--verbose", action="store_true",
                    help="Affiche certaines informations supplémentaires")
args = vars(parser.parse_args())
pos1 = args["first_term"]
pos2 = args["second_term"]
sentence = args["sentence"]


################################################################################
######################### Preprocessing de la phrase ###########################
################################################################################

embeddingsPath = 'corpus/deps.words'

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in np.arange(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)

labelsMapping = {'Other':0,
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2,
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4,
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6,
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}
word2Idx = {}
words = {}
folder = 'files/'
files = [folder+'train.txt', folder+'test.txt']

# Create a dictionnary of words of the sentence
for fileIdx in np.arange(len(files)):
    file = files[fileIdx]
    for line in open(file):
        splits = line.strip().split('\t')

        sentence_ = splits[3]
        tokens = sentence_.split(" ")
        for token in tokens:
            words[token.lower()] = True

for token in sentence.split(" "):
    words[token.lower()] = True

# Create the word to index dictionary using 'deps.words'
for line in open(embeddingsPath):
    split = line.strip().split(" ")
    word = split[0]

    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word

        word2Idx["UNKNOWN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        word2Idx[split[0]] = len(word2Idx)


def getWordIdx(token, word2Idx):
    """Returns from the word2Idex table the word index for a given token"""
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]

    return word2Idx["UNKNOWN"]

def createMatricesSentence(sentence, pos1, pos2, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []

    tokens = sentence.split(" ")

    tokenIds = np.zeros(maxSentenceLen)
    positionValues1 = np.zeros(maxSentenceLen)
    positionValues2 = np.zeros(maxSentenceLen)

    for idx in range(0, min(maxSentenceLen, len(tokens))):
        tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)

        distance1 = idx - int(pos1)
        distance2 = idx - int(pos2)

        if distance1 in distanceMapping:
            positionValues1[idx] = distanceMapping[distance1]
        elif distance1 <= minDistance:
            positionValues1[idx] = distanceMapping['LowerMin']
        else:
            positionValues1[idx] = distanceMapping['GreaterMax']

        if distance2 in distanceMapping:
            positionValues2[idx] = distanceMapping[distance2]
        elif distance2 <= minDistance:
            positionValues2[idx] = distanceMapping['LowerMin']
        else:
            positionValues2[idx] = distanceMapping['GreaterMax']

    tokenMatrix.append(tokenIds)
    positionMatrix1.append(positionValues1)
    positionMatrix2.append(positionValues2)

    return [np.array(tokenMatrix, dtype='int32'),
           np.array(positionMatrix1, dtype='int32'),
           np.array(positionMatrix2, dtype='int32')]


################################################################################
########################## Prédiction de la relation ###########################
################################################################################

stcP, pos1P, pos2P = createMatricesSentence(sentence, pos1, pos2, word2Idx, 97)

if (args["verbose"]):
    print(stcP)
    print(pos1P)
    print(pos2P)

model = load_model('models/model1.h5')
pred = model.predict_classes([stcP,pos1P,pos2P],
                             verbose=args["verbose"])[0]

if (args["verbose"]):
    print("Valeur de la relation : "+ str(pred))

relation = [r for r in labelsMapping.keys() if (labelsMapping[r]==pred)][0]
relation = relation.split("(")[0]

if (pred%2==0):
    relation += "("+sentence.split(" ")[pos2]+", "+sentence.split(" ")[pos1]+")"
else:
    relation += "("+sentence.split(" ")[pos1]+", "+sentence.split(" ")[pos2]+")"

print("Relation trouvée : "+relation)

del model
import gc; gc.collect()
