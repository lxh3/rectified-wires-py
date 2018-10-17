# Note - calcsig, calcoutgrad, calcsigvel and incbias have all been moved outside the class 
# to make it possible to precompile them easily with Numba

import sys
import numpy as np
import os.path

from numba import jit

@jit
def calcsig(depth, size, node, val, insize, inlist, inedge, bias, sig, EPS, w, c):
    """ 
    Calculates the values on each node, the value on each edge,
    true, zero, mult

    Inputs:
    * depth - number of layers in network 
    * size - list of the number of nodes in each layer
    * node - list of the indices of the nodes in each layer
    * val - value on each of the nodes
    * insize - number of nodes coming into each node
    * inlist - indices of the nodes coming into each node
    * inedge - indices of the edges coming into each node
    * bias - bias on each edge of the network
    * sig - value on each edge of the network
    * EPS - value below which values are deemed to be effectively zero
    * w - weights on each node of the network
    * c - the correct class for the given data vector

    Outputs:
    * val - values on each node
    * sig - values on each edge
    * true - 1 if the output node corresponding to the correct class is smallest, 0 otherwise
    * zero - 1 if output node corresponding to the correct class is effectively zero, 0 otherwise
    * mult - number of out nodes which have effectively the same value as the output
             node corresponding to the correct class
    """

    for d in xrange(1,depth+1):
        for k in xrange(size[d]):

            outnode = node[d][k]
            val[outnode] = 0.

            for i in xrange(insize[outnode]):
                innode = inlist[outnode][i]
                edge = inedge[outnode][i]

                s = val[innode]-bias[edge]
                sig[edge] = s if s>EPS else 0.

                val[outnode] += sig[edge]

            val[outnode] *= w[outnode]

    outclass = val[node[depth][c]]

    true = 1
    zero = 1 if outclass < EPS else 0
    mult = 1

    for k in xrange(size[depth]):

        if k==c:
            continue

        outnode = node[depth][k]

        if abs(val[outnode]-outclass)<EPS:
            mult+=1

        elif val[outnode]<outclass:
            true = 0
            return val, sig, true, zero, mult

    return val, sig, true, zero, mult

@jit
def calcoutgrad(depth, size, node, outgrad, outsize, sig, outedge, outlist, w):
    """ 
    Calculates the gradient on each node of the network.
    This gradient is used in the training update.

    Inputs:
    * depth - number of layers in network 
    * size - list of the number of nodes in each layer
    * node - list of the indices of the nodes in each layer
    * outgrad - gradient for each node of the network used in update 
    * outsize - number of nodes coming out from each node
    * sig - value on each edge of the network
    * outedge - indices of the edges going out of each node
    * outlist - indeices of the nodes coming out of each node
    * w - weights on each node of the network

    Outputs:
    * outgrad - updated gradient for each node of the network
    """

    for d in xrange(depth-1,0,-1):
        for k in xrange(size[d]):

            innode = node[d][k]

            outgrad[innode]=0.

            for i in xrange(outsize[innode]):
                if sig[outedge[innode][i]]>0.:

                    outgrad[innode]+=outgrad[outlist[innode][i]]

            outgrad[innode]*=w[innode]

    return outgrad

@jit
def calcsigvel(depth, size, node, outsize, outlist, outedge, sigvel, sig, insize, inedge, w, outgrad):
    """
    Calculates the velocity on each edge of the network.
    This velocity is used in the training update.
 
    Inputs:
    * depth - number of layers in network 
    * size - list of the number of nodes in each layer
    * node - list of the indices of the nodes in each layer
    * outsize - number of nodes coming out from each node
    * outlist - indeices of the nodes coming out of each node
    * outedge - indices of the edges going out of each node
    * sigvel - velocity on each edge of the network used for update
    * sig - value on each edge of the network
    * insize - number of nodes coming into each node
    * inedge - indices of the edges coming into each node
    * w - weights on each node of the network
    * outgrad - gradient for each node of the network used in update 

    Outputs:
    * sigvel - updated velocity on each edge
    """

    for d in xrange(depth):
        for k in xrange(size[d]):

            innode = node[d][k]

            for i in xrange(outsize[innode]):

                outnode = outlist[innode][i]

                edgeout = outedge[innode][i]

                sigvel[edgeout]=0.

                if sig[edgeout]==0.:
                    continue

                if d>0:
                    for j in xrange(insize[innode]):

                        edgein = inedge[innode][j]

                        if sig[edgein]>0.:

                            sigvel[edgeout]+=sigvel[edgein]

                    sigvel[edgeout]*=w[innode]

                sigvel[edgeout]+=outgrad[outnode]

    return sigvel

@jit()
def incbias(rmin, sig, depth, size, node, insize, inedge, bias, outgrad):
    """
    Uses the computed outgrad and sigvel (through rmin) 
    to increase the bias on each edge.
 
    Inputs:
    * rmin - step size for the update
    * sig - value on each edge of the network
    * depth - number of layers in network 
    * size - list of the number of nodes in each layer
    * node - list of the indices of the nodes in each layer
    * insize - number of nodes coming into each node
    * inedge - indices of the edges coming into each node
    * bias - bias on each edge of the network
    * outgrad - gradient for each node of the network used in update 

    Outputs:
    * bias - updated bias on each edge
    """

    for d in xrange(1,depth+1):
        for k in xrange(size[d]):

            outnode = node[d][k]

            for i in xrange(insize[outnode]):

                edge = inedge[outnode][i]

                if sig[edge]>0.:
                    bias[edge]+=rmin*outgrad[outnode]
    return bias

class SDA(object):
    """ 
    Sets up a class to keep the training via the 
    Sequential Deactivation Algorithm self contained.

    Inputs:
    * net - network generated by expander.py and loaded into a class via rectified_wires_net.py
    * vecsize - size of the data vectors (input nodes)
    * classnum - number of classes (output nodes)
    * symsize - 1 if analog values in input vectors, >1 otherwise
    * amax - maximum value if analog, None otherwise
    * pos - conversion of symbolic data into a numerical vector
    * trainf - training file to read from
    * testf - test file to read from
    * trainstart - position of the first line of data in trainf
    * teststart - position of the first line of data in testf
    * verbose - flag to increase verbosity

    Outputs:
    * class containing all of the information necessary to apply the 
      Sequential Deactivation Algorithm to train a network
    """

    def __init__(self, net, vecsize, classnum, symsize, amax, pos, trainf, testf, trainstart, teststart, verbose):

        self.nodenum = net.nodenum
        self.edgenum = net.edgenum
        self.size = net.size
        self.depth = net.depth
        self.node = net.node
        self.insize = net.insize
        self.inlist = net.inlist
        self.inedge = net.inedge
        self.outsize = net.outsize
        self.outlist = net.outlist
        self.outedge = net.outedge

        self.vecsize = vecsize
        self.classnum = classnum
        self.symsize = symsize
        self.amax = amax
        self.pos = pos
        self.trainf = trainf
        self.testf = testf
        self.trainstart = trainstart
        self.teststart = teststart
        self.verbose = verbose
        
        self.w = net.w.copy()
        self.val = np.zeros(self.nodenum)
        self.outgrad = np.zeros(self.nodenum)
        self.bias = net.bias
        self.bias0 = np.zeros(self.edgenum)
        self.sig = np.zeros(self.edgenum)
        self.sigvel = np.zeros(self.edgenum)
        self.invec = np.zeros(vecsize)


        self.EPS = 1e-12
        self.D = 20
        self.ZEROSTOP = .1
        self.itercount = 0
        self.falsecount = 0

        self.avemax = 0.
        self.c = 0

        self.tolerance = 1e-12
        self.stuck = False


    def get_rmin(self):

	#Calculate the step size for an update

        r_array = self.sig/self.sigvel
        r_array = r_array[~np.isnan(r_array)]

        return np.min(r_array)


    def readdata(self, fileptr, start, test):

        # Read line from file to set the input vector and class

        line = fileptr.readline().strip().split()

        if line == []:
            if test:
                return 0 
            else:
                fileptr.seek(start)
                line = fileptr.readline().strip().split()

        if self.symsize == 1:

            for j in xrange(self.vecsize/2):
                self.invec[2*j]=float(line[j])/self.amax
                self.invec[2*j+1] = 1.- self.invec[2*j]
        else:

            for i in xrange(self.vecsize):
                self.invec[i]=0.
            for j in xrange(self.vecsize/self.symsize):
                sym = ord(line[0][j])
                self.invec[self.symsize*j+np.int(self.pos[sym])]=1.

        self.c = int(line[-1])

        return 1
     

    def setinput(self):

        # Set the value of the input nodes to be the input vector

        for k in xrange(self.vecsize):
            self.val[self.node[0][k]] = self.invec[k]

        return


    def setoutgrad(self):
        
        # Initialize the out gradient on each node

        for k in xrange(self.size[self.depth]):
            self.outgrad[self.node[self.depth][k]]=0.

        self.outgrad[self.node[self.depth][self.c]]=1.

        return


    def initbias(self):

        # Initialize bias to zero before training

        self.bias = np.zeros(self.edgenum)
        
        return


    def setweights(self):

        # Determine the appropriate weights for each node of the network

        self.w = np.empty(self.nodenum)

        for d in xrange(1,self.depth):
            for k in xrange(self.size[d]):
                n = self.node[d][k]
                self.w[n] = np.sqrt(1./(self.insize[n]*self.outsize[n]))

        for k in xrange(self.size[self.depth]):
            n = self.node[self.depth][k]
            self.w[n] = 1.

        return


    def reducebias(self, iteration):

        # Reduce the bias on an edge if it is larger than the value on the input node

        for d in xrange(1, self.depth+1):
            for k in xrange(self.size[d]):

                outnode = self.node[d][k]

                for i in xrange(self.insize[outnode]):
                    edge = self.inedge[outnode][i]
                    innode = self.inlist[outnode][i]

                    if self.bias[edge]>self.val[innode]:
                        self.bias[edge] = self.val[innode] if self.val[innode]>self.bias0[edge] else self.bias0[edge]
        return
    

    def learn(self):

        # Performs one update cycle

        for e in xrange(self.edgenum):
            self.bias0[e] = self.bias[e];

        self.setinput()
        self.setoutgrad()
        
        iteration = 0

        done = 0
        while not done:

            self.val, self.sig, true, zero, mult = calcsig(self.depth, self.size, self.node, self.val, self.insize, self.inlist, \
                                                           self.inedge, self.bias, self.sig, self.EPS, self.w, self.c)

            if zero or (true and mult==1):
                done = 1

            else:

                self.outgrad = calcoutgrad(self.depth, self.size, self.node, self.outgrad, self.outsize, self.sig, self.outedge, self.outlist, self.w)

                self.sigvel = calcsigvel(self.depth, self.size, self.node, self.outsize, self.outlist, self.outedge, \
                                         self.sigvel, self.sig, self.insize, self.inedge, self.w, self.outgrad)

                rmin = self.get_rmin()
                self.bias = incbias(rmin, self.sig, self.depth, self.size, self.node, self.insize, self.inedge, self.bias, self.outgrad)

                iteration+=1

        if iteration!=0:

            self.reducebias(iteration)

        return iteration


    def train(self, trainbatch, overallt):

        # Calls learn to perform an update trainbatch times each time it is called

        aveiter=0.
        false=0

        t=0
        while t<trainbatch:

            self.readdata(self.trainf,self.trainstart,False)

            t+=1

            iteration = self.learn()
            #if t%1000==0:
            #    print overallt+t

            self.itercount+=iteration

            aveiter+=iteration
            false+=1 if iteration>0 else 0
            
        self.falsecount+=false

        return aveiter/false if false else 0.


    def test(self, batch):

        # Tests on testbatch number of test data points after each iteration of train
       
        avetrue = 0.
        avezero = 0.
        aveact = np.zeros(self.D)

        for d in xrange(self.depth):
            aveact[d] = 0.

        self.testf.seek(self.teststart)

        t = 0
        
        while t<batch:

            flag = self.readdata(self.testf,self.teststart,True)
            if flag==1:
                
                t+=1

                self.setinput()
                
                self.val, self.sig, true, zero, mult = calcsig(self.depth, self.size, self.node, self.val, self.insize, self.inlist, self.inedge, self.bias, self.sig, self.EPS, self.w, self.c)

                avetrue+=1./mult if true==1 else 0.

                if zero and mult>=0:
                    avezero+=1.

                for d in xrange(self.depth):
                
                    actcount = 0
                    edgecount = 0

                    for k in xrange(self.size[d]):

                        innode = self.node[d][k]

                        if self.val[innode]<self.EPS:
                            continue

                        for i in xrange(self.outsize[innode]):
                            actcount+=1 if self.sig[self.outedge[innode][i]]==0 else 0 
                            edgecount+=1

                    aveact[d]+=float(actcount)/edgecount
              

            else:

                break

        avetrue/=t
        avezero/=t

        for d in xrange(self.depth):
            aveact[d]/=t

        return avetrue, avezero, aveact
