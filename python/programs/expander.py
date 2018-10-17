import sys
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('insize', type=int, help='number of input nodes')
    parser.add_argument('outsize', type=int, help='number of classes')
    parser.add_argument('hidlayers', type=int, help='number of hidden layers')
    parser.add_argument('growth', type=int, help='layer-to-layer growth factor (integer)')
    parser.add_argument('netfile', type=str, nargs='*', default=['netfile'], help='destination of networkinput file')
    args = parser.parse_args()

    print '\nConstructing sparse expander network with the following characteristics:'
    for arg in vars(args):
    	print arg, getattr(args, arg)

    outdeg = 2*args.growth
    
    try:
	if args.hidlayers>20:
	    raise ValueError('hiddenlayers may not exceed 20')
    except ValueError:
	exit('hiddenlayers may not exceed 20')

    depth = args.hidlayers+1
    
    size = [0]*21
    size[0] = args.insize
    size[depth] = args.outsize
    edgenum = 0

    for d in xrange(depth-1):
	size[d+1] = (outdeg/2)*size[d]
        edgenum += 2*size[d+1]

    edgenum += args.outsize*size[depth-1]

    if args.netfile==['netfile']:
        print str(edgenum)+' edges'

    notused=np.empty(size[depth-1])

    txtfile = open(args.netfile[0],'w')
    txtfile.write(str(args.hidlayers)+'\n')
    
    for d in xrange(depth+1):
	txtfile.write(str(size[d])+' ')

    txtfile.write('\n'+str(edgenum)+'\n\n')

    instart=0
    outstart=size[0]

    count=0

    for d in xrange(depth-1):
	for innum in xrange(2):
	    for j in xrange(size[d+1]):
		if count==0:
		    for i in xrange(size[d]):
			notused[i]=i
			count=size[d]
				
	    	i=np.random.randint(0,count)
	    	txtfile.write(str(int(instart+notused[i]))+' '+str(int(outstart+j))+' 0. \n')
	    	
		count-=1		
		notused[i]=notused[count]
	
	instart+=size[d]
        outstart+=size[d+1]
  
    for j in xrange(size[depth]):
	for i in xrange(size[depth-1]):
		txtfile.write(str(int(instart+i))+' '+str(int(outstart+j))+' 0. \n')

    txtfile.close()

    print('Sparse expander network successfully constructed.\n')

if __name__ == '__main__':

    main()
   
