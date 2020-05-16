'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
import numpy as np
import sys
def ransac_est_homography(x, y, X, Y, threshold):
    N = x.size
    A = np.zeros([2 * N, 9])
    ranIter = 1000
    ux = x.reshape(-1,N)
    uy = y.reshape(-1,N)
    uz = np.ones(N).reshape(-1,N)
    uPtsFull = np.vstack([ux,uy,uz])
    vx = X.reshape(-1,N)
    vy = Y.reshape(-1,N)
    vz = np.ones(N).reshape(-1,N)
    vPtsFull = np.vstack([vx,vy,vz])
    i = 0
    maxInlierCount = 0
    minDistanceSum = sys.maxsize
    # populates A with points
    while i < N:
        a = np.array([x[i], y[i], 1]).reshape(-1, 3)
        c = np.array([[X[i]], [Y[i]]])
        d = - c * a
        A[2 * i, 0 : 3], A[2 * i + 1, 3 : 6]= a, a
        A[2 * i : 2 * i + 2, 6 : ] = d
        i += 1
    ptindices = []
    inlier_ind = []
    for i in range(ranIter):
        At = np.zeros([8,9])
        indexList = list(range(N))
        backup = []
        for j in range(4):
            rnd = np.random.randint(0,len(indexList))
            ind = indexList.pop(rnd)
            backup.append(ind)
            At[2*j,:]=A[2*ind,:]
            At[2*j +1,:] = A[2*ind+1,:]
        [u,d,v] = np.linalg.svd(At,full_matrices=True)
        Ht = v[-1,:]/v[-1,-1]
        H = Ht.reshape(3,3)
        vPtsPred = np.matmul(H,uPtsFull)
        vPtsPred = vPtsPred/vPtsPred[-1,:]
        distances = np.linalg.norm(vPtsFull-vPtsPred,axis=0)
        inliers = 1*np.less_equal(distances,threshold)
        inlierCount = np.sum(1*inliers)
        distanceSum = np.sum(distances)
        if (maxInlierCount==inlierCount and distanceSum < minDistanceSum) or maxInlierCount<inlierCount:
            maxInlierCount = inlierCount
            minDistanceSum = distanceSum
            ptindices = np.argwhere(inliers==1)
            inlier_ind = inliers
    ptindices = ptindices[:,0]
    j=0
    Anew = np.zeros([2*ptindices.shape[0],9])
    print(ptindices.shape)
    for i in range(ptindices.shape[0]):
        Anew[2*j,:] = A[2*ptindices[i],:]
        Anew[2*j+1,:] = A[2*ptindices[i]+1,:]
        j=j+1
    U, s, V = np.linalg.svd(Anew, full_matrices=True)
    h = V[-1, :]/V[-1,-1]
    print("Max Inlier Count: " + str(maxInlierCount))
    H = h.reshape(3, 3)
    return H, inlier_ind
