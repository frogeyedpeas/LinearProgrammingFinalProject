from numpy import *
import numpy as np


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

        





x = array([1,2,3,4])
y = diag(x)
print y


#need to create unit vector


