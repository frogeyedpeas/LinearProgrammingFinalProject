
import math
import numpy

def dist(seg,point): # segment is a pair of points
    

    return dist,x,y #distance followed by x,y






def Triangle(point,hullboundary,pivot,e): #given pivot, find point in hull so line cost is minimized
    if distance(pivot,point) < e:
        return True,pivot

    for p in hullboundary:
        tmp = 0.5*(point+pivot) #average the point and pivot
