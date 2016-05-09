
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D





def pointToLineSegement(point, start,end):
    direction = start - end
    normalized = direction/(np.norm(direction))
    phase = start - normalized #starting from this point, moving one unit at a time we pass through the line
    maxbound = norm(direction) #so from K = 1 to K = maxbound we are on the line segment, phase + K*normalized 
    # Line = normalized + 
    
    possible = np.dot(phase - point,)




def DiagonalNewton(d0,d1,Q,tolerance):

    if tolerance <= 0:
            print "tolerance <= 0, algorithm won't terminate"
            return None

    #verify if Q is positive semidefinite, correct dimensions
    try:
        y = linalg.cholesky(Q)
    except linalg.LinAlgError:
        print "Matrix is either not positive semidefinite, or nonsquare"
        return None

    e = np.ones(d1.size)
    w = linalg.norm(d1 -d0)
    D = np.diag(d1)
    w = linalg.norm(np.dot(np.dot(np.dot(D,Q),D),e)-e)
    iterationcounter = 0
    while(w > tolerance):
        d0 = d1
        D = np.diag(d1)


        try:
       
            y = np.dot(linalg.matrix_power((linalg.matrix_power(D,-2)+Q),-1), np.diag( linalg.matrix_power(D,-1 ))-np.dot(Q,d1))

        except:

            return None
      
          
       # print  np.dot(np.dot(np.transpose(y) ,linalg.matrix_power(D,-2)),y)
        
        lamb = np.dot(
                    np.dot(np.transpose(y) ,linalg.matrix_power(D,-2)),
                            y) + np.dot(np.dot(np.transpose(y),Q),y) #computing lambda
        #lamb < 1, a = 1 lamb >=1 1, a = 1/2
        
        #print lamb
        if lamb > 1:
            d1 = d0 + float(1)/(1 + lamb**(0.5))*y

        else:
            d1 = d0 + y
        w = linalg.norm(np.dot(np.dot(np.dot(D,Q),D),e)-e)

        iterationcounter += 1

    return (w, d1,iterationcounter)


d0 = np.array([1,1])
Q = np.array([[1,0],[0,2]])
d1 = np.array([2,9])


i = 0
while i < 5:
    j = 0
    while j < 5:
        if j != 0 and i !=0:
            #rendering plots now:
            k = 0
            X = []
            Y = []
            Z = []
            while k < 10:
                h = 0
                X.append(k)
                Y.append(k)
                Z.append([])
                while h < 10:
                
                    teal = DiagonalNewton(np.array([0,0]),np.array([h,k]), np.diag(np.array([i,j])),.00000001)
                    if teal != None:
                        Z[k].append(teal[2])
                    else:
                        Z[k].append(0)
                    h+=1
    
                k+=1
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            xx,yy = np.meshgrid(X,Y)
            ax.plot_wireframe(xx, yy, Z)
            plt.savefig("Image: "+str(i)+ ","+str(j), cmap=plt.cm.jet, rstride=.01, cstride=.01, linewidth=1,alpha=0)
            
        j+=1
    i+=1




                    











