### Projet d'extraction de relations

Ce projet constitue l'une des deux composantes de notre travail dans l'UE WIA.

### Objectif

Le but de cette application est de permettre la détection d'une des 19 relations binaires proposées, sur la donnée d'une phrase et de deux entités.
Par exemple :

~~~
$ python src/main.py "The tree is coming from the forest ." -e1 1 -e2 6
Relation trouvée : Entity-Origin(tree, forest)
~~~

### Utilisation

Le code est testé pour :
- Tensorflow 0.12.1
- Keras 1.2.1
- Python 3.5

La version de Tensorflow n'est pas la dernière, et quelques bugs (de simples _warnings_) apparaissent régulièrement lors du run du programme. Ils ne sont cependant pas corrigé pour l'instant dans les versions les plus récentes, mais pas de quoi s'inquiéter.

Pour les instructions, lancez :

~~~
# Pour créer les fichiers d'entraînement et de test
python src/CreateTrainTestFiles.py

# Pour créer les embeddings et adapter les données à l'entrainement du réseau de neurones
python src/preprocessing.py

# Pour entrainer le réseau de neurones
python src/CNN.py

# Pour obtenir l'aide du programme
python src/main.py -h
~~~

### Origines

Le réseau de neurones se base sur l'article "Relation Classification via Convolutional Deep Neural Network" de MM. Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou et Jun Zha, publié à l'occasion du COLING de 2014.

### Authors

Matthieu RÉ, Yaohui WANG
