import numpy as np
#'''
#valid/test:
test_accompaniment=np.load('./data/test_accompaniment.npy')
valid_accompaniment = test_accompaniment[:len(test_accompaniment)/2]
test_accompaniment = test_accompaniment[len(test_accompaniment)/2:]
np.save('./data/test_accompaniment.npy',test_accompaniment)
np.save('./data/valid_accompaniment.npy',valid_accompaniment)
del test_accompaniment,valid_accompaniment
'''
test_vocals= np.load('./data/test_vocals.npy')
valid_vocals = test_vocals[:len(test_vocals)/2]
test_vocals = test_vocals[len(test_vocals)/2:]
np.save('./data/test_vocals.npy',test_vocals)
np.save('./data/valid_vocals.npy',valid_vocals)
del test_vocals,valid_vocals

test_mixture=np.load('./data/test_mixture.npy')
valid_mixture = test_mixture[:len(test_mixture)/2]
test_mixture = test_mixture[len(test_mixture)/2:]
np.save('./data/test_mixture.npy',test_mixture)
np.save('./data/valid_mixture.npy',valid_mixture)
del test_mixture,valid_mixture

'''
