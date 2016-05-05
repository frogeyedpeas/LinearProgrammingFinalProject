
import numpy as np





def DiagonalNewton(d0,d1,Q,lamb,tolerance):
    while(linalg.norm(d1 -d0) > tolerance):
        d0 = d1
        D = np.diag(d1)
        y = linalg.matrix_power((linalg.matrix_power(D,-2)+Q),-1)*(linalg.matrix_power(np.diag(D,-1) )-Q*d1)
        lamb = (np.transpose(y) * linalg.matrix_power(D,-2)*y  + np.transpose(y)*Q*y)[0] #computing lambda
        #lamb < 1, a = 1 lamb >=1 1, a = 1/2
        if lamb > 1:
            d1 = d0 + float(1)/(1 + lamb)*y

        else:
            d1 = d0 + y



            
            
    

def HomogenousProgram(Q,eps,gamma): #matrix Q, item e

    def PhaseI(Q,eps,gamma):
        #verify that input is good

        if eps <= 0:
            print "epsilon <= 0, algorithm won't terminate"
            return None

        #verify if Q is positive semidefinite, correct dimensions
        try:
            y = linalg.cholesky(Q)
        except linalg.LinAlgError:
            print "Matrix is either not positive semidefinite, or nonsquare"
            return None


        #Step 0:
        
        params = Q.shape
        #need to create unit vector
        e = ones(params[0])
        u = e - Q*e
        d = e
        t = 1

        rstar = (params[0]**0.5 - gamma)/(params[0]**0.5-gamma**2)
        CGam = linalg.norm(u)**2*(2*params[0] + gamma*(1 + params[0]**0.5))/gamma**2
        tstar = eps/CGam

        #Step 1:

        






