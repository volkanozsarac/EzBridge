import sys
import os
import matplotlib
from math import asin, sqrt, acos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from numpy.matlib import repmat
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
import openseespy.opensees as ops
from scipy import interpolate

def plot_sec(obj):
    num_sec = len(set(obj.model['Bent']['Sections']))
    for i in range(num_sec):
        D = obj.model['Bent']['D'][i]  # Section diameter
        cc = obj.model['Bent']['cover'][i]  # Section diameter
        numBars = obj.model['Bent']['numBars'][i]  # Section diameter
        dl = obj.model['Bent']['dl'][i]  # Section diameter
        barArea = np.pi * dl ** 2 / 4
        secTag = str(i + 1)
        yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT = obj._circ_fiber_config(D)
        rc = ro - cc
        barRplotfactor = 1
        # filename = os.path.join('Outputs','FiberSections','section'+str(SecTag)+".png")
        plt.figure()
        plot_patchcirc(yC, zC, nfCoverT, nfCoverR, rc, ro, startAng, endAng, 'lightgrey')  # unconfined concrete
        plot_patchcirc(yC, zC, nfCoreT, nfCoreR, ri, rc, startAng, endAng, 'grey')  # confined concrete
        plot_layercirc(numBars, barArea, yC, zC, rc, startAng, endAng, barRplotfactor)
        plt.title('Pier Section no: %s' % str(secTag))
        plt.xlim([-ro, ro])
        plt.ylim([-ro, ro])
        plt.xlabel('z-coord (m)')
        plt.ylabel('y-coord (m)')
        plt.show()
        # plt.savefig(filename,bbox_inches='tight')

    if obj.model['Bent_Foundation']['Type'] in ['Pile-Shaft', 'Group Pile'] and obj.model['Bent_Foundation']['EleType'] in [1, 2]:
        num_sec = len(set(obj.model['Bent_Foundation']['Sections']))
        for i in range(num_sec):
            D = obj.model['Bent_Foundation']['D'][i]  # Section diameter
            cc = obj.model['Bent_Foundation']['cover'][i]  # Section diameter
            numBars = obj.model['Bent_Foundation']['numBars'][i]  # Section diameter
            dl = obj.model['Bent_Foundation']['dl'][i]  # Section diameter
            barArea = np.pi * dl ** 2 / 4
            secTag = str(i + 1)
            yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT = obj._circ_fiber_config(D)
            rc = ro - cc
            barRplotfactor = 1
            # filename = os.path.join('Outputs','FiberSections','section'+str(SecTag)+".png")
            plt.figure()
            plot_patchcirc(yC, zC, nfCoverT, nfCoverR, rc, ro, startAng, endAng, 'lightgrey')  # unconfined concrete
            plot_patchcirc(yC, zC, nfCoreT, nfCoreR, ri, rc, startAng, endAng, 'grey')  # confined concrete
            plot_layercirc(numBars, barArea, yC, zC, rc, startAng, endAng, barRplotfactor)
            plt.title('Pile Section no: %s' % str(secTag))
            plt.xlim([-ro, ro])
            plt.ylim([-ro, ro])
            plt.xlabel('z-coord (m)')
            plt.ylabel('y-coord (m)')
            plt.show()
            # plt.savefig(filename,bbox_inches='tight')

def plot_patchcirc(yC, zC, nfp, nft, intR, extR, startAng, endAng, mcolor):
    # yC, zC: y & z-coordinates of the center of the circle
    # nft: number of radial divisions in the core (number of "rings")
    # nfp: number of theta divisions in the core (number of "wedges")
    # intR:	internal radius
    # extR:	external radius
    # startAng:	starting angle
    # endAng:	ending angle
    # mcolor: color to use for fiber fill

    Rvals = np.linspace(intR, extR, nft + 1);
    Rvals.shape = (1, len(Rvals))
    x = repmat(zC, 1, nft + 1)
    y = repmat(yC, 1, nft + 1)
    N = 2 * nfp + 1
    theta = np.linspace(startAng * np.pi / 180, endAng * np.pi / 180, N)
    theta.shape = (len(theta), 1)
    tx = repmat(np.cos(theta), 1, max(x.shape))
    ty = repmat(np.sin(theta), 1, max(x.shape))
    newx = repmat(x, N, 1) + repmat(Rvals, N, 1) * tx
    newy = repmat(y, N, 1) + repmat(Rvals, N, 1) * ty
    plt.plot(newx, newy, c='k', lw=1)
    plt.fill(newx, newy, mcolor)

    thetavals = np.linspace(startAng * np.pi / 180, endAng * np.pi / 180, nfp + 1)
    zInt = zC + np.cos(thetavals) * intR
    zExt = zC + np.cos(thetavals) * extR
    yInt = yC + np.sin(thetavals) * intR
    yExt = yC + np.sin(thetavals) * extR
    plt.plot([zInt, zExt], [yInt, yExt], c='k', lw=1)


def plot_layercirc(nbars, barA, yC, zC, midR, startAng, endAng, barRplotfactor=1):
    # yC, zC: y & z-coordinates of the center of the circle
    # nbars: Number of longitudinal bars
    # barA: Area of longitudinal bars
    # midR:	radius of circular arc
    # startAng:	starting angle
    # endAng:	ending angle
    # barRplotfactor: magnification factor to apply on rebar plots

    N = 12
    mcolor = 'k'
    theta = np.linspace(startAng, endAng, nbars) * np.pi / 180
    x = midR * np.cos(theta)
    y = midR * np.sin(theta)
    Radius = (barA / np.pi) ** 0.5
    R = Radius * barRplotfactor
    theta = np.linspace(0, 2 * np.pi, N)
    theta.shape = (len(theta), 1)
    tx = repmat(np.cos(theta), 1, max(x.shape))
    ty = repmat(np.sin(theta), 1, max(x.shape))
    newx = repmat(x, N, 1) + repmat(R, N, 1) * tx
    newy = repmat(y, N, 1) + repmat(R, N, 1) * ty
    plt.fill(newx, newy, mcolor)


def plot_model(obj, show_node_tags='no', show_element_tags='no', show_node='no'):
    #  ------------------------------------------------------------------------------------------------------------
    #  MODEL PLOTTING
    #  ------------------------------------------------------------------------------------------------------------
    nodeList = ops.getNodeTags()
    eleList = ops.getEleTags()

    beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': '-'}  # beam elements
    bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # bearing elements
    soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': '-'}  # spring elements
    rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # rigid links
    LinkSlab_style = {'color': 'blue', 'linewidth': 1, 'linestyle': '-'}  # link slab
    node_style = {'s': 3, 'color': 'black', 'marker': 'o', 'facecolor': 'black'}
    node_text_style = {'fontsize': 6, 'fontweight': 'regular', 'color': 'green'}
    ele_text_style = {'fontsize': 6, 'fontweight': 'bold', 'color': 'darkred'}

    x = []
    y = []
    z = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # ELEMENTS
    for element in eleList:
        if str(element)[-2:] == obj.BearingTag[-2:]:
            ele_style = bearing_style
        elif str(element)[-2:] == obj.LinkTag[-2:]:
            ele_style = LinkSlab_style
        elif str(element)[-2:] == obj.RigidTag[-2:]:
            ele_style = rigid_style
        elif str(element)[-2:] == obj.SpringEleTag[-2:]:
            ele_style = soil_style
        elif str(element)[-2:] == obj.AbutTag[-2:]:
            ele_style = soil_style
        else:
            ele_style = beam_style

        Nodes = ops.eleNodes(element)
        # 3D beam-column elements
        iNode = ops.nodeCoord(Nodes[0])
        jNode = ops.nodeCoord(Nodes[1])

        plt.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)

        if show_element_tags == 'yes':
            ax.text((iNode[0] + jNode[0]) / 2, (iNode[1] + jNode[1]) * 1.02 / 2,
                    (iNode[2] + jNode[2]) * 1.02 / 2, str(element), **ele_text_style)  # label elements

    # RIGID LINKS
    if obj.const_opt != 1:
        for i in range(len(obj.RigidLinkNodes)):
            iNode = ops.nodeCoord(obj.RigidLinkNodes[i][0])
            jNode = ops.nodeCoord(obj.RigidLinkNodes[i][1])
            plt.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **rigid_style)

    # NODES
    x = []
    y = []
    z = []
    for node in nodeList:
        x.append(ops.nodeCoord(node)[0])  # list of x coordinates to define plot view area
        y.append(ops.nodeCoord(node)[1])
        z.append(ops.nodeCoord(node)[2])
        if show_node_tags == 'yes':
            ax.text(ops.nodeCoord(node)[0] * 1.02, ops.nodeCoord(node)[1] * 1.02, ops.nodeCoord(node)[2] * 1.02,
                    str(node), **node_text_style)  # label nodes

    if show_node == 'yes':
        ax.scatter(x, y, z, **node_style)

    nodeMins = np.array([min(x), min(y), min(z)])
    nodeMaxs = np.array([max(x), max(y), max(z)])
    xViewCenter = (nodeMins[0] + nodeMaxs[0]) / 2
    yViewCenter = (nodeMins[1] + nodeMaxs[1]) / 2
    zViewCenter = (nodeMins[2] + nodeMaxs[2]) / 2
    view_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    ax.set_xlim(xViewCenter - (view_range / 4), xViewCenter + (view_range / 4))
    ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
    ax.set_zlim(zViewCenter - (view_range / 3), zViewCenter + (view_range / 3))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text2D(0.10, 0.95, "Undeformed shape", transform=ax.transAxes, fontweight="bold")

    plt.axis('on')
    plt.show()


def plot_deformedshape(obj, ax=None, scale=5):
    # scale: scale factor to be applied
    # ax: the axis handler to plot the deformed shape
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    eleList = ops.getEleTags()
    dofList = [1, 2, 3]
    Disp_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
    beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': ':'}  # beam elements
    bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': ':'}  # bearing elements
    soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
    rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': ':'}  # rigid links
    LinkSlab_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # link slab

    x = [];
    y = [];
    z = []
    for element in eleList:
        if str(element)[-2:] == obj.BearingTag[-2:]:
            ele_style = bearing_style
        elif str(element)[-2:] == obj.LinkTag[-2:]:
            ele_style = LinkSlab_style
        elif str(element)[-2:] == obj.RigidTag[-2:]:
            ele_style = rigid_style
        elif str(element)[-2:] == obj.SpringEleTag[-2:]:
            ele_style = soil_style
        elif str(element)[-2:] == obj.AbutTag[-2:]:
            ele_style = soil_style
        else:
            ele_style = beam_style
        Disp_style['color'] = ele_style['color']

        Nodes = ops.eleNodes(element)
        iNode = ops.nodeCoord(Nodes[0])
        jNode = ops.nodeCoord(Nodes[1])

        iNode_Disp = [];
        jNode_Disp = []
        for dof in dofList:
            iNode_Disp.append(ops.nodeDisp(Nodes[0], dof))
            jNode_Disp.append(ops.nodeDisp(Nodes[1], dof))
        # Add original and deformed shape to get final node coordinates
        iNode_final = [iNode[0] + scale * iNode_Disp[0], iNode[1] +
                       scale * iNode_Disp[1], iNode[2] + scale * iNode_Disp[2]]
        jNode_final = [jNode[0] + scale * jNode_Disp[0], jNode[1] +
                       scale * jNode_Disp[1], jNode[2] + scale * jNode_Disp[2]]

        x.append(iNode[0]);
        x.append(jNode[0])  # list of x coordinates to define plot view area
        y.append(iNode[1]);
        y.append(jNode[1])  # list of y coordinates to define plot view area
        z.append(iNode[2]);
        z.append(iNode[2])  # list of z coordinates to define plot view area

        ax.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)
        ax.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]), (iNode_final[2], jNode_final[2]),
                marker='', **Disp_style)

    # RIGID LINKS if constraints are used
    if obj.const_opt != 1:
        Eig_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
        for i in range(len(obj.RigidLinkNodes)):
            iNode = ops.nodeCoord(obj.RigidLinkNodes[i][0])
            jNode = ops.nodeCoord(obj.RigidLinkNodes[i][1])
            iNode_Disp = [];
            jNode_Disp = []
            for dof in dofList:
                iNode_Disp.append(ops.nodeDisp(obj.RigidLinkNodes[i][0], dof))
                jNode_Disp.append(ops.nodeDisp(obj.RigidLinkNodes[i][1], dof))

            # Add original and mode shape to get final node coordinates
            iNode_final = [iNode[0] + scale * iNode_Disp[0], iNode[1] + scale * iNode_Disp[1],
                           iNode[2] + scale * iNode_Disp[2]]
            jNode_final = [jNode[0] + scale * jNode_Disp[0], jNode[1] + scale * jNode_Disp[1],
                           jNode[2] + scale * jNode_Disp[2]]
            plt.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **rigid_style)
            plt.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                     (iNode_final[2], jNode_final[2]),
                     marker='', **Eig_style)

    nodeMins = np.array([min(x), min(y), min(z)])
    nodeMaxs = np.array([max(x), max(y), max(z)])
    xViewCenter = (nodeMins[0] + nodeMaxs[0]) / 2
    yViewCenter = (nodeMins[1] + nodeMaxs[1]) / 2
    zViewCenter = (nodeMins[2] + nodeMaxs[2]) / 2
    view_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    ax.set_xlim(xViewCenter - (view_range / 4), xViewCenter + (view_range / 4))
    ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
    ax.set_zlim(zViewCenter - (view_range / 3), zViewCenter + (view_range / 3))
    ax.text2D(0.10, 0.95, "Deformed shape", transform=ax.transAxes, fontweight="bold")
    ax.text2D(0.10, 0.90, "Scale Factor: " + str(scale), transform=ax.transAxes, fontweight="bold")
    ax.axis('off')


def plot_modeshape(obj, modeNumber=1, scale=200):
    #  ------------------------------------------------------------------------------------------------------------
    #  MODE SHAPE PLOTTING
    #  ------------------------------------------------------------------------------------------------------------
    # Perform eigenvalue analysis
    listSolvers = ['-genBandArpack', '-fullGenLapack', '-symmBandLapack']

    ok = 1
    for s in listSolvers:
        ops.wipeAnalysis()
        try:
            eigenValues = ops.eigen(s, modeNumber + 1)
            catchOK = 0
            ok = 0
        except:
            catchOK = 1

        if catchOK == 0:
            for i in range(modeNumber + 1):
                if eigenValues[i] < 0:
                    ok = 1
            if ok == 0:
                break
    if ok != 0:
        print("Error on eigenvalue something is wrong...")
        sys.exit()
    else:
        Lambda = np.asarray(eigenValues)
        Omega = Lambda ** 0.5
        Tn = 2 * np.pi / Omega

    eleList = ops.getEleTags()
    beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': ':'}  # beam elements
    bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': ':'}  # bearing elements
    soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
    rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': ':'}  # rigid links
    LinkSlab_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # link slab
    Eig_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements

    x = []
    y = []
    z = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for element in eleList:
        if str(element)[-2:] == obj.BearingTag[-2:]:
            ele_style = bearing_style
        elif str(element)[-2:] == obj.LinkTag[-2:]:
            ele_style = LinkSlab_style
        elif str(element)[-2:] == obj.RigidTag[-2:]:
            ele_style = rigid_style
        elif str(element)[-2:] == obj.SpringEleTag[-2:]:
            ele_style = soil_style
        elif str(element)[-2:] == obj.AbutTag[-2:]:
            ele_style = soil_style
        else:
            ele_style = beam_style
        Eig_style['color'] = ele_style['color']

        Nodes = ops.eleNodes(element)
        iNode = ops.nodeCoord(Nodes[0])
        jNode = ops.nodeCoord(Nodes[1])
        iNode_Eig = ops.nodeEigenvector(Nodes[0], modeNumber)
        jNode_Eig = ops.nodeEigenvector(Nodes[1], modeNumber)

        # Add original and mode shape to get final node coordinates
        iNode_final = [iNode[0] + scale * iNode_Eig[0], iNode[1] + scale * iNode_Eig[1],
                       iNode[2] + scale * iNode_Eig[2]]
        jNode_final = [jNode[0] + scale * jNode_Eig[0], jNode[1] + scale * jNode_Eig[1],
                       jNode[2] + scale * jNode_Eig[2]]

        x.append(iNode_final[0])  # list of x coordinates to define plot view area
        y.append(iNode_final[1])  # list of y coordinates to define plot view area
        z.append(iNode_final[2])  # list of z coordinates to define plot view area

        plt.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)
        plt.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                 (iNode_final[2], jNode_final[2]),
                 marker='', **Eig_style)

    # RIGID LINKS if constraints are used
    if obj.const_opt != 1:
        Eig_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
        for i in range(len(obj.RigidLinkNodes)):
            iNode = ops.nodeCoord(obj.RigidLinkNodes[i][0])
            jNode = ops.nodeCoord(obj.RigidLinkNodes[i][1])
            iNode_Eig = ops.nodeEigenvector(obj.RigidLinkNodes[i][0], modeNumber)
            jNode_Eig = ops.nodeEigenvector(obj.RigidLinkNodes[i][1], modeNumber)

            # Add original and mode shape to get final node coordinates
            iNode_final = [iNode[0] + scale * iNode_Eig[0], iNode[1] + scale * iNode_Eig[1],
                           iNode[2] + scale * iNode_Eig[2]]
            jNode_final = [jNode[0] + scale * jNode_Eig[0], jNode[1] + scale * jNode_Eig[1],
                           jNode[2] + scale * jNode_Eig[2]]
            plt.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **rigid_style)
            plt.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                     (iNode_final[2], jNode_final[2]),
                     marker='', **Eig_style)

    nodeMins = np.array([min(x), min(y), min(z)])
    nodeMaxs = np.array([max(x), max(y), max(z)])
    xViewCenter = (nodeMins[0] + nodeMaxs[0]) / 2
    yViewCenter = (nodeMins[1] + nodeMaxs[1]) / 2
    zViewCenter = (nodeMins[2] + nodeMaxs[2]) / 2
    view_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    ax.set_xlim(xViewCenter - (view_range / 4), xViewCenter + (view_range / 4))
    ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
    ax.set_zlim(zViewCenter - (view_range / 3), zViewCenter + (view_range / 3))
    ax.text2D(0.10, 0.95, "Mode " + str(modeNumber), transform=ax.transAxes, fontweight="bold")
    ax.text2D(0.10, 0.90, "T = " + str("%.3f" % Tn[modeNumber-1]) + " s", transform=ax.transAxes, fontweight="bold")
    plt.axis('off')
    plt.show()
    ops.wipeAnalysis()


def animate_nspa(obj, loading_scheme, LoadFactor,
                 DispCtrlNode, ctrlNode):
    nodeFile = os.path.join(obj.animation_dir, 'Nodes.out')
    eleFile = os.path.join(obj.animation_dir, 'Elements.out')
    dispFile = os.path.join(obj.animation_dir, 'NodeDisp_All.out')
    Movie = obj.Movie
    scale = obj.scale
    fps = obj.fps
    FrameStep = obj.FrameStep
    nodes = np.loadtxt(nodeFile)
    elements = np.loadtxt(eleFile)
    nodeDispArray = np.loadtxt(dispFile)
    nodeDispArray[:, 1:] = scale * nodeDispArray[:, 1:]
    timeArray = np.arange(0, len(nodeDispArray[:, 0]))
    Frames = np.arange(0, len(nodeDispArray[:, 0]), FrameStep)
    framesTime = timeArray[Frames]
    FrameInterval = 1000 / fps
    aniFrames = len(Frames)
    dynLines1 = []  # handler for pushover curve
    dynLines2 = []  # handler for displaced shaped - elements
    dynLines3 = []  # handler for displaced shaped - rigid links

    beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': ':'}  # beam elements
    bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': ':'}  # bearing elements
    rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': ':'}  # rigid links
    LinkSlab_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # link slab
    soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
    Disp_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
    node_style = {'s': 12, 'color': 'red', 'marker': 'o', 'facecolor': 'blue'}
    node_text_style = {'fontsize': 10, 'fontweight': 'regular', 'color': 'green'}

    fig = plt.figure(figsize=(18, 8))
    plt.suptitle('%s pushover with control node: %d' % (loading_scheme, ctrlNode), fontweight="bold", y=0.92)
    plt.tight_layout()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], projection='3d')

    # plot the pushover curve
    temp = ax1.plot(DispCtrlNode[:0], LoadFactor[:0], lw=2, color='red')
    dynLines1 = temp[0]
    ax1.grid(True)
    ax1.set_xlabel('$u_{ctrl}$ [m]')
    ax1.set_ylabel('$V_{base}$ [kN]')
    ax1.set_xlim([0, 1.2 * max(DispCtrlNode)])
    ax1.set_ylim([0, 1.2 * max(LoadFactor)])

    x = [];
    y = [];
    z = [];
    ctrlplot = 0
    for i in range(len(elements[:, 0])):
        element = int(elements[i, 0])
        if str(element)[-2:] == obj.BearingTag[-2:]:
            ele_style = bearing_style
        elif str(element)[-2:] == obj.LinkTag[-2:]:
            ele_style = LinkSlab_style
        elif str(element)[-2:] == obj.RigidTag[-2:]:
            ele_style = rigid_style
        elif str(element)[-2:] == obj.SpringEleTag[-2:]:
            ele_style = soil_style
        elif str(element)[-2:] == obj.AbutTag[-2:]:
            ele_style = soil_style
        else:
            ele_style = beam_style
        Disp_style['color'] = ele_style['color']

        idx_i = np.where(nodes[:, 0] == elements[i, 1:][0])[0][0]
        idx_j = np.where(nodes[:, 0] == elements[i, 1:][1])[0][0]

        # get the nodal coordinates of element in undeformed shape
        iNode = nodes[idx_i, 1:]
        jNode = nodes[idx_j, 1:]

        # get the nodal coordinates of element in deformed shape        
        iNode_Disp = nodeDispArray[0, idx_i * 3 + 1: idx_i * 3 + 4]
        jNode_Disp = nodeDispArray[0, idx_j * 3 + 1: idx_j * 3 + 4]
        # Add original and deformed shape to get final node coordinates
        iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
        jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]

        x.append(iNode[0]);
        x.append(jNode[0])  # list of x coordinates to define plot view area
        y.append(iNode[1]);
        y.append(jNode[1])  # list of y coordinates to define plot view area
        z.append(iNode[2]);
        z.append(jNode[2])  # list of z coordinates to define plot view area

        ax2.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)
        temp = ax2.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                        (iNode_final[2], jNode_final[2]),
                        marker='', **Disp_style)
        dynLines2.append(temp[0])

        if ctrlNode == nodes[idx_i, 0] and ctrlplot == 0:
            ctrlplot = 1
            ax2.scatter(iNode[0], iNode[1], iNode[2], **node_style)
            dynNode = ax2.scatter(iNode_final[0], iNode_final[1], iNode_final[2], **node_style)
            ax2.text(iNode[0] * 1.02, iNode[1] * 1.02, iNode[2] * 1.02, str(ctrlNode), **node_text_style)  # label nodes

        if ctrlNode == nodes[idx_j, 0] and ctrlplot == 0:
            ctrlplot = 1
            ax2.scatter(jNode[0], jNode[1], jNode[2], **node_style)
            dynNode = ax2.scatter(jNode_final[0], jNode_final[1], jNode_final[2], **node_style)
            ax2.text(jNode[0] * 1.02, jNode[1] * 1.02, jNode[2] * 1.02, str(ctrlNode),
                     **node_text_style)  # label nodes

    # RIGID LINKS if constraints are used
    if obj.const_opt != 1:
        Disp_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
        for i in range(len(obj.RigidLinkNodes)):    
            idx_i = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][0])[0][0]
            idx_j = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][1])[0][0]
            # get the nodal coordinates of element in undeformed shape
            iNode = nodes[idx_i, 1:]
            jNode = nodes[idx_j, 1:]
            # get the nodal coordinates of element in deformed shape        
            iNode_Disp = nodeDispArray[0, idx_i * 3 + 1: idx_i * 3 + 4]
            jNode_Disp = nodeDispArray[0, idx_j * 3 + 1: idx_j * 3 + 4]
            # Add original and deformed shape to get final node coordinates
            iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
            jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]

            ax2.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **rigid_style)
            temp = ax2.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                            (iNode_final[2], jNode_final[2]),
                            marker='', **Disp_style)
            dynLines3.append(temp[0])

    nodeMins = np.array([min(x), min(y), min(z)])
    nodeMaxs = np.array([max(x), max(y), max(z)])
    xViewCenter = (nodeMins[0] + nodeMaxs[0]) / 2
    yViewCenter = (nodeMins[1] + nodeMaxs[1]) / 2
    zViewCenter = (nodeMins[2] + nodeMaxs[2]) / 2
    view_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    xmin = xViewCenter - (view_range / 4 * 1.3);
    xmax = xViewCenter + (view_range / 4) * 1.1
    ymin = yViewCenter - (view_range / 6);
    ymax = yViewCenter + (view_range / 6)
    zmin = zViewCenter - (view_range / 10);
    zmax = zViewCenter + (view_range / 10)
    ax2.grid(True)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_zlim(zmin, zmax)
    ax2.text2D(0.05, 0.1, "Deformed shape", transform=ax2.transAxes, fontweight="bold")
    ax2.text2D(0.05, 0.05, "Scale Factor: " + str(scale), transform=ax2.transAxes, fontweight="bold")
    ax2.view_init(elev=60, azim=120)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    # Slider Location and size relative to plot
    axSlider = plt.axes([0.25, .02, 0.50, 0.02])
    plotSlider = Slider(axSlider, 'Step No', Frames[0], Frames[-1], valinit=Frames[0], valfmt='%0.0f')

    # Animation controls
    global is_paused
    is_paused = False  # True if user has taken control of the animation

    def on_click(event):
        # Check where the click happened
        (xm, ym), (xM, yM) = plotSlider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # Toggle on off based on clicking
            global is_paused
            if is_paused == True:
                is_paused = False
            elif is_paused == False:
                is_paused = True

    def animate_slider(Time):

        global is_paused
        is_paused = True
        now = framesTime[(np.abs(framesTime - plotSlider.val)).argmin()]
        tStep = (np.abs(timeArray - now)).argmin()
        dynLines1.set_data(DispCtrlNode[:tStep], LoadFactor[:tStep])

        ctrlplot = 0
        # update element locations
        for i in range(len(elements[:, 0])):
            idx_i = np.where(nodes[:, 0] == elements[i, 1:][0])[0][0]
            idx_j = np.where(nodes[:, 0] == elements[i, 1:][1])[0][0]

            # get the nodal coordinates of element in undeformed shape
            iNode = nodes[idx_i, 1:]
            jNode = nodes[idx_j, 1:]

            # get the nodal coordinates of element in deformed shape        
            iNode_Disp = nodeDispArray[tStep, idx_i * 3 + 1: idx_i * 3 + 4]
            jNode_Disp = nodeDispArray[tStep, idx_j * 3 + 1: idx_j * 3 + 4]
            # Add original and deformed shape to get final node coordinates
            iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
            jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]
            coords_x = [iNode_final[0], jNode_final[0]]
            coords_y = [iNode_final[1], jNode_final[1]]
            coords_z = [iNode_final[2], jNode_final[2]]
            dynLines2[i].set_data_3d(coords_x, coords_y, coords_z)

            if ctrlNode == nodes[idx_i, 0] and ctrlplot == 0:
                ctrlplot = 1

                dynNode._offsets3d = (
                    np.array([iNode_final[0]]), np.array([iNode_final[1]]), np.array([iNode_final[2]]))

            if ctrlNode == nodes[idx_j, 0] and ctrlplot == 0:
                ctrlplot = 1
                dynNode._offsets3d = (
                    np.array([jNode_final[0]]), np.array([jNode_final[1]]), np.array([jNode_final[2]]))

        if obj.const_opt != 1:
            for i in range(len(obj.RigidLinkNodes)):       
                idx_i = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][0])[0][0]
                idx_j = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][1])[0][0]
                # get the nodal coordinates of element in undeformed shape
                iNode = nodes[idx_i, 1:]
                jNode = nodes[idx_j, 1:]
                # get the nodal coordinates of element in deformed shape        
                iNode_Disp = nodeDispArray[tStep, idx_i * 3 + 1: idx_i * 3 + 4]
                jNode_Disp = nodeDispArray[tStep, idx_j * 3 + 1: idx_j * 3 + 4]
                # Add original and deformed shape to get final node coordinates
                iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
                jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]
                coords_x = [iNode_final[0], jNode_final[0]]
                coords_y = [iNode_final[1], jNode_final[1]]
                coords_z = [iNode_final[2], jNode_final[2]]
                dynLines3[i].set_data_3d(coords_x, coords_y, coords_z)

        fig.canvas.draw_idle()

        return dynLines1, dynLines2, dynLines3, dynNode

    def update_plot(ii):
        # If the control is manual, we don't change the plot    
        global is_paused
        if is_paused:
            return dynLines1, dynLines2, dynLines3, dynNode

        # Find the close timeStep and plot that
        CurrentTime = plotSlider.val
        CurrentFrame = (np.abs(framesTime - CurrentTime)).argmin()

        CurrentFrame += 1
        if CurrentFrame >= len(Frames):
            CurrentFrame = Frames[0]

        # Update the slider
        plotSlider.set_val(framesTime[CurrentFrame])

        is_paused = False  # the above line called update_slider, so we need to reset this
        return dynLines1, dynLines2, dynLines3, dynNode

    plotSlider.on_changed(animate_slider)

    # assign click control
    fig.canvas.mpl_connect('button_press_event', on_click)

    if Movie == 1:
        print('Saving the animation...')
        Movfile = os.path.join(obj.animation_dir, 'NSPA.mp4')
        ani = animation.FuncAnimation(fig, update_plot, aniFrames, interval=FrameInterval)
        ani.save(Movfile, writer='ffmpeg')
        print('Animation is saved!')


def animate_nrha(obj):
    nodeFile = os.path.join(obj.animation_dir, 'Nodes.out')
    eleFile = os.path.join(obj.animation_dir, 'Elements.out')
    dispFile = os.path.join(obj.animation_dir, 'NodeDisp_All.out')
    Movie = obj.Movie
    scale = obj.scale
    fps = obj.fps
    FrameStep = obj.FrameStep
    nodes = np.loadtxt(nodeFile)
    elements = np.loadtxt(eleFile)
    nodeDispArray = np.loadtxt(dispFile)
    nodeDispArray[:, 1:] = scale * nodeDispArray[:, 1:]
    timeArray = nodeDispArray[:, 0]
    Frames = np.arange(0, len(nodeDispArray[:, 0]), FrameStep)
    framesTime = timeArray[Frames]
    FrameInterval = 1000 / fps
    aniFrames = len(Frames)

    beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': ':'}  # beam elements
    bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': ':'}  # bearing elements
    rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': ':'}  # rigid links
    LinkSlab_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # link slab
    Disp_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
    soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
    fig = plt.figure(figsize=(14.4, 8.1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    dynLines = []
    dynLines2 = []
    x = []
    y = []
    z = []
    for i in range(len(elements[:, 0])):
        element = int(elements[i, 0])
        if str(element)[-2:] == obj.BearingTag[-2:]:
            ele_style = bearing_style
        elif str(element)[-2:] == obj.LinkTag[-2:]:
            ele_style = LinkSlab_style
        elif str(element)[-2:] == obj.RigidTag[-2:]:
            ele_style = rigid_style
        elif str(element)[-2:] == obj.SpringEleTag[-2:]:
            ele_style = soil_style
        elif str(element)[-2:] == obj.AbutTag[-2:]:
            ele_style = soil_style
        else:
            ele_style = beam_style
        Disp_style['color'] = ele_style['color']

        idx_i = np.where(nodes[:, 0] == elements[i, 1:][0])[0][0]
        idx_j = np.where(nodes[:, 0] == elements[i, 1:][1])[0][0]

        # get the nodal coordinates of element in undeformed shape
        iNode = nodes[idx_i, 1:]
        jNode = nodes[idx_j, 1:]

        # get the nodal coordinates of element in deformed shape        
        iNode_Disp = nodeDispArray[0, idx_i * 3 + 1: idx_i * 3 + 4]
        jNode_Disp = nodeDispArray[0, idx_j * 3 + 1: idx_j * 3 + 4]
        # Add original and deformed shape to get final node coordinates
        iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
        jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]

        x.append(iNode[0])
        x.append(jNode[0])  # list of x coordinates to define plot view area
        y.append(iNode[1])
        y.append(jNode[1])  # list of y coordinates to define plot view area
        z.append(iNode[2])
        z.append(jNode[2])  # list of z coordinates to define plot view area

        ax.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)
        temp = ax.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                       (iNode_final[2], jNode_final[2]),
                       marker='', **Disp_style)
        dynLines.append(temp[0])

    # RIGID LINKS if constraints are used
    if obj.const_opt != 1:
        Disp_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
        for i in range(len(obj.RigidLinkNodes)):
            idx_i = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][0])[0][0]
            idx_j = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][1])[0][0]
            # get the nodal coordinates of element in undeformed shape
            iNode = nodes[idx_i, 1:]
            jNode = nodes[idx_j, 1:]
            # get the nodal coordinates of element in deformed shape
            iNode_Disp = nodeDispArray[0, idx_i * 3 + 1: idx_i * 3 + 4]
            jNode_Disp = nodeDispArray[0, idx_j * 3 + 1: idx_j * 3 + 4]
            # Add original and deformed shape to get final node coordinates
            iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
            jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]

            ax.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **rigid_style)
            temp = ax.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                           (iNode_final[2], jNode_final[2]),
                           marker='', **Disp_style)
            dynLines2.append(temp[0])

    nodeMins = np.array([min(x), min(y), min(z)])
    nodeMaxs = np.array([max(x), max(y), max(z)])
    xViewCenter = (nodeMins[0] + nodeMaxs[0]) / 2
    yViewCenter = (nodeMins[1] + nodeMaxs[1]) / 2
    zViewCenter = (nodeMins[2] + nodeMaxs[2]) / 2
    view_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    plt.grid(True)
    xmin = xViewCenter - (view_range / 4 * 1.3)
    xmax = xViewCenter + (view_range / 4) * 1.1
    ymin = yViewCenter - (view_range / 5)
    ymax = yViewCenter + (view_range / 5)
    zmin = zViewCenter - (view_range / 10)
    zmax = zViewCenter + (view_range / 10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.text2D(0.10, 0.95, "Deformed shape", transform=ax.transAxes, fontweight="bold")
    ax.text2D(0.10, 0.90, "Scale Factor: " + str(scale), transform=ax.transAxes, fontweight="bold")
    ax.view_init(elev=60, azim=120)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Slider Location and size relative to plot
    axSlider = plt.axes([0.25, .03, 0.50, 0.02])
    plotSlider = Slider(axSlider, 'Time [sec]', framesTime[0], framesTime[-1], valinit=framesTime[0])

    # Animation controls
    global is_paused
    is_paused = False  # True if user has taken control of the animation

    def on_click(event):
        # Check where the click happened
        (xm, ym), (xM, yM) = plotSlider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # Toggle on off based on clicking
            global is_paused
            if is_paused == True:
                is_paused = False
            elif is_paused == False:
                is_paused = True

    def animate_slider(Time):

        global is_paused
        is_paused = True
        now = framesTime[(np.abs(framesTime - plotSlider.val)).argmin()]
        tStep = (np.abs(timeArray - now)).argmin()

        # update element locations
        for i in range(len(elements[:, 0])):
            idx_i = np.where(nodes[:, 0] == elements[i, 1:][0])[0][0]
            idx_j = np.where(nodes[:, 0] == elements[i, 1:][1])[0][0]

            # get the nodal coordinates of element in undeformed shape
            iNode = nodes[idx_i, 1:]
            jNode = nodes[idx_j, 1:]

            # get the nodal coordinates of element in deformed shape        
            iNode_Disp = nodeDispArray[tStep, idx_i * 3 + 1: idx_i * 3 + 4]
            jNode_Disp = nodeDispArray[tStep, idx_j * 3 + 1: idx_j * 3 + 4]
            # Add original and deformed shape to get final node coordinates
            iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
            jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]
            coords_x = [iNode_final[0], jNode_final[0]]
            coords_y = [iNode_final[1], jNode_final[1]]
            coords_z = [iNode_final[2], jNode_final[2]]
            dynLines[i].set_data_3d(coords_x, coords_y, coords_z)

        if obj.const_opt != 1:
            for i in range(len(obj.RigidLinkNodes)):
                idx_i = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][0])[0][0]
                idx_j = np.where(nodes[:, 0] == obj.RigidLinkNodes[i][1])[0][0]
                # get the nodal coordinates of element in undeformed shape
                iNode = nodes[idx_i, 1:]
                jNode = nodes[idx_j, 1:]
                # get the nodal coordinates of element in deformed shape
                iNode_Disp = nodeDispArray[tStep, idx_i * 3 + 1: idx_i * 3 + 4]
                jNode_Disp = nodeDispArray[tStep, idx_j * 3 + 1: idx_j * 3 + 4]
                # Add original and deformed shape to get final node coordinates
                iNode_final = [iNode[0] + iNode_Disp[0], iNode[1] + iNode_Disp[1], iNode[2] + iNode_Disp[2]]
                jNode_final = [jNode[0] + jNode_Disp[0], jNode[1] + jNode_Disp[1], jNode[2] + jNode_Disp[2]]
                coords_x = [iNode_final[0], jNode_final[0]]
                coords_y = [iNode_final[1], jNode_final[1]]
                coords_z = [iNode_final[2], jNode_final[2]]
                dynLines2[i].set_data_3d(coords_x, coords_y, coords_z)

        fig.canvas.draw_idle()

        return dynLines, dynLines2

    def update_plot(ii):
        # If the control is manual, we don't change the plot    
        global is_paused
        if is_paused:
            return dynLines, dynLines2

        # Find the close timeStep and plot that
        CurrentTime = plotSlider.val
        CurrentFrame = (np.abs(framesTime - CurrentTime)).argmin()

        CurrentFrame += 1
        if CurrentFrame >= len(Frames):
            CurrentFrame = Frames[0]

        # Update the slider
        plotSlider.set_val(framesTime[CurrentFrame])

        is_paused = False  # the above line called update_slider, so we need to reset this
        return dynLines, dynLines2

    plotSlider.on_changed(animate_slider)

    # assign click control
    fig.canvas.mpl_connect('button_press_event', on_click)

    if Movie == 1:
        print('Saving the animation...')
        Movfile = os.path.join(obj.animation_dir, 'NRHA.mp4')
        ani = animation.FuncAnimation(fig, update_plot, aniFrames, interval=FrameInterval)
        ani.save(Movfile, writer='ffmpeg')
        print('Animation is saved!')


def bilin_pushover(out_dir, LoadFactor, DispCtrlNode, M_star, Bilin_approach, loading_scheme, ctrlNode):
    # idealized elasto-perfectly plastic force-displacement relationship using equal energy approach
    if Bilin_approach == 'EC':  # Eurocode 8 Approach
        idx_star = LoadFactor.index(max(LoadFactor))
        Fy_star = LoadFactor[idx_star]
        Fu_star = Fy_star
        Du_star = DispCtrlNode[idx_star]
        E_star = np.trapz(LoadFactor[:idx_star], DispCtrlNode[:idx_star])
        Dy_star = 2 * (Du_star - E_star / Fy_star)

    elif Bilin_approach == 'ASCE':  # ASCE 7-16 approach
        ninterp = 1e4
        idx_star = LoadFactor.index(max(LoadFactor))
        Fu_star = LoadFactor[idx_star]
        Du_star = DispCtrlNode[idx_star]
        E_star = np.trapz(LoadFactor[:idx_star], DispCtrlNode[:idx_star])
        tolcheck = 1e-2  # 1% error is accepted
        Fy_range = np.arange(0.3 * Fu_star, 0.95 * Fu_star, (0.65 * Fu_star) / ninterp)
        Dy_range = interpolate.interp1d(LoadFactor, DispCtrlNode)(Fy_range)
        for Fy_trial in Fy_range:
            Dy_trial = ((Fu_star + Fy_trial) * Du_star - 2 * E_star) / Fu_star
            if Dy_trial > 0:
                D_60 = 0.6 * Dy_trial
                F_60 = 0.6 * Fy_trial
                F_diff = abs(Fy_range - F_60)
                F_diff = F_diff.tolist()
                D_diff = abs(Dy_range - D_60)
                D_diff = D_diff.tolist()
                idx_check = F_diff.index(min(F_diff))
                tol1 = F_diff[idx_check] / F_60
                tol2 = D_diff[idx_check] / D_60
                if tol1 <= tolcheck and tol2 <= tolcheck:
                    Fy_star = Fy_trial
                    Dy_star = Dy_trial
                    break

    elif Bilin_approach == 'NTC':  # NTC-2018 approach
        ninterp = 1e4
        idx_min = LoadFactor.index(max(LoadFactor))
        diff_F = abs(np.asarray(LoadFactor[idx_min:]) - max(LoadFactor) * 0.85)
        diff_F = diff_F.tolist()
        idx_max = idx_min + diff_F.index(min(diff_F))
        tolcheck = 1e-2  # 1% error is accepted
        Du_range = np.arange(DispCtrlNode[idx_min], DispCtrlNode[idx_max],
                             (DispCtrlNode[idx_max] - DispCtrlNode[idx_min]) / ninterp)
        Fu_range = interpolate.interp1d(DispCtrlNode, LoadFactor)(Du_range)
        Dy_range = np.arange(0, DispCtrlNode[idx_min], DispCtrlNode[idx_min] / ninterp)
        Fy_range = interpolate.interp1d(DispCtrlNode, LoadFactor)(Dy_range)

        for idx in range(len(Fu_range)):
            Fu_trial = Fu_range[idx]
            Du_trial = Du_range[idx]
            Fy_trial = Fu_trial
            diff_D = abs(np.asarray(DispCtrlNode) - Du_trial)
            diff_D = diff_D.tolist()
            idx_trial = diff_D.index(min(diff_D))
            E_trial = np.trapz(LoadFactor[:idx_trial], DispCtrlNode[:idx_trial])
            Dy_trial = ((Fu_trial + Fy_trial) * Du_trial - 2 * E_trial) / Fu_trial
            if Dy_trial > 0:
                D_60 = 0.6 * Dy_trial
                F_60 = 0.6 * Fy_trial
                F_diff = abs(Fy_range - F_60)
                F_diff = F_diff.tolist()
                D_diff = abs(Dy_range - D_60)
                D_diff = D_diff.tolist()
                idx_check = F_diff.index(min(F_diff))
                tol1 = F_diff[idx_check] / F_60
                tol2 = D_diff[idx_check] / D_60
                if tol1 <= tolcheck and tol2 <= tolcheck:
                    Fu_star = Fu_trial
                    Fy_star = Fy_trial
                    Du_star = Du_trial
                    Dy_star = Dy_trial
                    break

    T_star = 2 * np.pi * (M_star * Dy_star / Fy_star) ** 0.5
    fepp = [0, Fy_star, Fu_star]
    depp = [0, Dy_star, Du_star]

    plt.figure()
    plt.tight_layout()
    plt.plot(DispCtrlNode, LoadFactor, label='Actual Curve')
    plt.plot(depp, fepp, label='Idealized Curve')
    plt.legend(frameon=False, loc='lower right')
    plt.grid(True)
    plt.xlabel('$u_{ctrl}$ [m]')
    plt.ylabel('$V_{base}$ [kN]')
    plt.title('%s pushover with control node: %d' % (loading_scheme, ctrlNode))
    ax = plt.gca()
    ax.text(0.75, 0.15, '$T^{*}$ = ' + "{:.2f}".format(float(T_star)) + ' sec', transform=ax.transAxes, fontsize=10,
            style='italic')
    fname = os.path.join(out_dir, 'PushOver Curve.png')
    plt.savefig(fname, bbox_inches='tight')
