import sys
import numpy as np

class RectifiedWiresNet(object):
    """ 
    Sets up a network from file as a class. 
    This class gets passed to the training 
    algorithm in sda.py

    Inputs:
    * netfile - file to read network data from
    * classnum - number of classes (output nodes)
    * vecsize - size of the data vectors (input nodes)

    Outputs:
    * class containing all of the information about the
      network necessary to train/test on the data
    """

    def __init__(self, netfile=' 2_10', classnum=2, vecsize=32):
        
        self.node = []
        self.nodenum = 0

        try:
            with open(netfile) as nf:

                # Define depth and size of each layer

                self.hidden = int(nf.readline())
                self.depth = self.hidden + 1

                self.size = [int(i) for i in nf.readline().split()]
                if self.size[0]!=vecsize or self.size[self.depth]!=classnum:
                    exit('net must have '+str(vecsize)+' input nodes and '+str(classnum)+' output nodes\n')

                # Populate the node array by node index

                for d in xrange(self.depth+1):
                    self.node.append([i+self.nodenum for i in xrange(self.size[d])])
                    self.nodenum += self.size[d]

                # Construct arrays where, for each node, array[node] lists the input size, output size,
                # input nodes, output nodes, input edge indices, and output edge indices respectively
                # In the same loop, initialize bias values on the edges

                self.insize = [0 for n in xrange(self.nodenum)]
                self.outsize = [0 for n in xrange(self.nodenum)]
                self.inlist = [[] for n in xrange(self.nodenum)]
                self.outlist = [[] for n in xrange(self.nodenum)]
                self.inedge = [[] for n in xrange(self.nodenum)]
                self.outedge = [[] for n in xrange(self.nodenum)]

                self.edgenum = int(nf.readline())
                nf.readline()
                self.bias = np.zeros(self.edgenum)

                for e in xrange(self.edgenum):
                    line = nf.readline().split()
                    n1 = int(line[0])
                    n2 = int(line[1])
                    self.bias[e] = float(line[2])
                    self.insize[n2]+=1
                    self.outsize[n1]+=1
                    self.inlist[n2].append(n1)
                    self.outlist[n1].append(n2)
                    self.inedge[n2].append(e)
                    self.outedge[n1].append(e)

                # Initialize weight values

                self.w = np.zeros(self.nodenum)
                for d in xrange(1,self.depth):
                    for k in xrange(self.size[d]):
                        n = self.node[d][k]
                        self.w[n] = np.sqrt(1./(self.insize[n]*self.outsize[n]))
                for k in xrange(self.size[self.depth]):
                    n = self.node[self.depth][k]
                    self.w[n] = 1.

        except IOError:
            exit('netfile not found')


