# VocalPortrait

### TODO
Stavo pensando che piusttosto che richiamare preprocessing in train.py potesse essere meglio fare un file di preprocessing a parte per ottenere gia il dataset pronto per il train del VAE. Sarebbe quindi da rivedere la funzione che per velocita' e per stanchezza ho fatto scrivere a GPT di split e preprocess e richiamarla direttamente nello stesso file. Se volete fare prove il dataset che ho utilizzato e' https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

Inoltre, per quanto riguarda la loss del VAE magari ve ne parlo meglio domani ma ho avuto problemi non avendo mai fatto una cosa del genere ma praticamente cio' che bisogna fare e' fare lo slice dei layers e valutare man mano la generazione dell'immagine rispetto a quella vera usando una CNN a parte. Anche quella sarebbe da rivedere, ho seguito le indicazioni che danno nel paper ma non ne sono convinto https://paperswithcode.com/paper/deep-feature-consistent-variational

Infine, non sono sicurissimo dello script di train del VAE, anche questo mi sembra un po troppo complicato, non me lo ricordavo cosi difficile il train dei modelli ma probabilmente essendo di un altro tipo ha un altro tipo di script per il train

In audioe c'e' una prima implementazione di quello che dovrebbe fare l'encoder delle tracce audio che risultano dal modello che dovrebbe aver fatto andrea oggi, cosi da mappare appunto sti vettori nel latent space
