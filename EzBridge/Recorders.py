import openseespy.opensees as ops
import os
import numpy as np

def user(obj):    
    
    # ops.recorder('Node', '-file',  'disp1.txt', '-time', '-node', obj.D1Nodes[1], '-dof', 1,2, 'disp')
    # eleNodes = ops.eleNodes(obj.EleIDsPY[0][5])
    # ops.recorder('Element', '-file',  'eleForce_py.txt', '-time', '-ele', obj.EleIDsPY[0][5], 'forces')
    # ops.recorder('Element', '-file',  'eleDisp_py.txt', '-time', '-ele', obj.EleIDsPY[0][5], 'deformation')
    
    ops.recorder('Element', '-file',  'eleForce_Gap1.txt', '-time', '-ele', obj.EleIDsGap[0][3], 'localForce')
    ops.recorder('Element', '-file',  'eleDisp_Gap1.txt', '-time', '-ele', obj.EleIDsGap[0][3], 'deformation')

    ops.recorder('Element', '-file',  'eleForce_AB1R.txt', '-time', '-ele', obj.EleIDsAB1[0], 'forces')
    ops.recorder('Element', '-file',  'eleDisp_AB1R.txt', '-time', '-ele', obj.EleIDsAB1[0], 'deformation')
    ops.recorder('Element', '-file',  'eleForce_AB2R.txt', '-time', '-ele', obj.EleIDsAB2[0], 'forces')
    ops.recorder('Element', '-file',  'eleDisp_AB2R.txt', '-time', '-ele', obj.EleIDsAB2[0], 'deformation')

    
    ops.record()


def animation(animation_dir):
    """
    This function saves the nodes and elments for an active model, in a 
    standardized format. The OpenSees model must be active in order for the 
    function to work.
    """
   
    # Get nodes and elements
    nodeList = ops.getNodeTags()
    eleList = ops.getEleTags()
    dofList = [1, 2, 3]
    # Consider making these optional arguements
    nodeName = 'Nodes'
    eleName = 'Elements'
    delim = ' '
    fmt = '%.10e'
    ftype = '.out'
    
    dispFile = os.path.join(animation_dir,'NodeDisp_All.out')
    ops.recorder('Node', '-file', dispFile, '-time', '-closeOnWrite',
                 '–node', *nodeList, '-dof', *dofList, 'disp')
    
    # Check Number of dimensions and intialize variables
    ndm = len(ops.nodeCoord(nodeList[0]))
    Nnodes = len(nodeList)
    nodes = np.zeros([Nnodes, ndm + 1])
    
    # Get Node list
    for ii, node in enumerate(nodeList):
        nodes[ii,0] = node
        nodes[ii,1:] = ops.nodeCoord(nodeList[ii])           
    
    Nele = len(eleList)
    elements = [None]*Nele
    
    # Generate the element list by looping through all elements
    for ii, ele in enumerate(eleList):
        tempNodes = ops.eleNodes(ele)
        tempNnodes = len(tempNodes)
        tempEle = np.zeros(tempNnodes + 1)
        tempEle[0] = int(ele)
        tempEle[1:] = tempNodes
        elements[ii] = tempEle       

    # Sort through the element arrays
    eleNode = np.array([ele for ele in elements])

    # SaveNodes
    nodeFile = os.path.join(animation_dir, nodeName + ftype)
    np.savetxt(nodeFile, nodes, delimiter = delim, fmt = fmt)
    
    # Save element arrays
    eleFile = os.path.join(animation_dir, eleName + ftype)
    np.savetxt(eleFile, eleNode, delimiter = delim, fmt = fmt)