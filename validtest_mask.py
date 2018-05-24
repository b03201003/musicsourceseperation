import numpy as np
#'''
#valid/test:
test_maskX=np.load('../CDAE/maskdata/test_maskX.npy')
valid_maskX = test_maskX[:len(test_maskX)/2]
test_maskX = test_maskX[len(test_maskX)/2:]
np.save('../CDAE/maskdata/test_maskX.npy',test_maskX)
np.save('../CDAE/maskdata/valid_maskX.npy',valid_maskX)
del test_maskX,valid_maskX

test_maskY=np.load('../CDAE/maskdata/test_maskY.npy')
valid_maskY = test_maskY[:len(test_maskY)/2]
test_maskY = test_maskY[len(test_maskY)/2:]
np.save('../CDAE/maskdata/test_maskY.npy',test_maskY)
np.save('../CDAE/maskdata/valid_maskY.npy',valid_maskY)
#del test_maskY,valid_maskY

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
