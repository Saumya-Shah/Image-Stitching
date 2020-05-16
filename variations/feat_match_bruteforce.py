import cv2
import numpy as np

def feat_match(descs1, descs2):
    match = []
    dMatch = []
    descs1 = descs1.T
    descs2 = descs2.T
    for desc in descs1:
        diff = descs2-desc
        norms = np.linalg.norm(diff,axis=1)
        idxs = norms.argsort()
        if norms[idxs[0]] < 0.8*norms[idxs[1]]:
            dMatch.append(cv2.DMatch(len(match),idxs[0],norms[idxs[0]]))
            match.append(idxs[0])
        else:
            match.append(-1)
    return match,dMatch
