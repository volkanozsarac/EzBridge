"""
This module is used to model and analyze: Multi-Span RC bridges having piers with circular sections
It makes use of OpenSees (Open System for Earthquake Engineering Simulation) framework
Output units: kN, m, sec

Author: Volkan Ozsarac, Earthquake Engineering PhD Candidate
Affiliation: University School for Advanced Studies IUSS Pavia
e-mail: volkanozsarac@iusspavia.it
"""

#  ----------------------------------------------------------------------------
#  Import Python Libraries
#  ----------------------------------------------------------------------------
import pickle
import openseespy.opensees as ops
from scipy.interpolate import interp1d
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from numpy.matlib import repmat
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from .utils import program_info, get_distance, create_dir, read_nga_record, get_current_time, get_run_time, get_units


class _builder:

    def __init__(self):
        """
        --------------------------
        OBJECT INITIALIZATION
        --------------------------
        """

        # Define units
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, degrees
        global gamma_rc, gamma_c, gamma_s, Ubig, Usmall, g

        # Define unit conversion
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()

        # Define constants
        Usmall = 1e-9  # very small or negligible value
        Ubig = 1e8  # very big value
        g = 9.81 * m / sec ** 2  # gravitational acceleration constant
        degrees = np.pi / 180.0  # angle unit, degrees
        gamma_rc = 25 * kN / m ** 3  # unit weight of reinforced concrete
        gamma_c = (2400 * kg / m ** 3) * g  # unit weight of normal-weight concrete
        gamma_s = (7850 * kg / m ** 3) * g  # unit weight of light-weight concrete

        # Node Tags
        self.STag = '001'  # Identifier for span nodes
        self.JTag = '002'  # Identifier for joint nodes
        self.PTag = '003'  # Identifier for pier nodes
        self.BTag = '004'  # Identifier for bearing node
        self.CTag = '005'  # Identifier for cap node
        self.ATag = '006'  # Identified for abutment nodes

        # Element Tags
        self.DeckTag = '001'  # Identifier for deck elements
        self.PierTag = '002'  # Identifier for pier elements
        self.AbutTag = '003'  # Identified for abutment springs
        self.JointTag = '004'  # Identifier for joint elements
        self.BearingTag = '005'  # Identifier for bearing elements
        self.BentCapTag = '006'  # Identifier for bent cap elements
        self.ShearKeyTag = '007'  # Identifier for shear key elements
        self.SpringEleTag = '008'  # Identifier for zero length soil springs
        self.PileEleTag = '009'  # Identifier for pile elements (single pile foundation)
        self.RigidTag = '099'  # Rigid like elements end with this tag

    def _build(self):
        """
        --------------------------
        MODEL BUILDER
        --------------------------
        """
        # Model initialization
        ops.wipe()  # Remove any existing model
        ops.wipeAnalysis()  # Remove any analysis object
        ops.model('basic', '-ndm', 3, '-ndf', 6)  # Define the model builder, ndm=#dimension, ndf=#dofs

        # Define material counters
        self.EndMatTag = 2  # Last used Material tag
        self.EndSecTag = 1  # Last used Section tag
        self.EndIntTag = 1  # Last used Integration tag
        self.EndTransfTag = 1  # Last used Transformation tag
        self.EndPtag = 0  # Last used load pattern tag
        self.EndTsTag = 0  # Last used load time-series tag
        # Define special materials, geometric transformations, sections and integrations
        self.BigMat = 1  # Material tag for elastic material with very stiffness
        self.ZeroMat = 2  # Material tag for elastic material with no stiffness
        self.BigSec = 1  # Section tag for section with high elastic stiffness
        self.BigInt = 1  # Beam-column integration tag for section with high elastic stiffness
        self.RigidTransfTag = 1  # Geometric transformation tag for rigid-like elements
        ops.uniaxialMaterial('Elastic', self.BigMat, Ubig)  # Generate elastic material with big stiffness
        ops.uniaxialMaterial('Elastic', self.ZeroMat, 0)  # Generate elastic material with no stiffness
        ops.section('Elastic', self.BigSec, Ubig, 1, 1, 1, Ubig, 1)
        ops.beamIntegration('Legendre', self.BigInt, self.BigSec, 2)
        ops.geomTransf('Linear', self.RigidTransfTag, 1, 0, 0)

        # Initialize some lists
        self.RigidLinkNodes = []  # List of node pairs to rigidly connect e.g. [[1,2],[1,3],[1,4]]
        self.fixed = []  # List of nodes to fully restrain
        self.equalDOFs = []  # List of node pairs and DOFs to apply equalDOFs e.g. [[1,2,1],[1,3,1],[1,4,1]]
        self.PointLoads = []  # List of point loads
        self.DistributedLoads = []  # List of distributed element loads

        # Mass type to use in elements
        if self.model['General']['Mass'] == 'Consistent':
            self.mass_type = '-cMass'
        elif self.model['General']['Mass'] == 'Lumped':
            self.mass_type = '-lMass'

        # Start building the model
        self._nodes()  # Define nodes
        self._decks()  # Define deck elements
        self._joints()  # Define link elements
        self._bearings()  # Define bearing elements or pier to deck connection
        self._bentcaps()  # Define bent cap elements
        self._bents()  # Define bent elements
        self._abutments()  # Define abutment elements
        self._shearkeys()  # Define shear key elements
        self._foundations()  # Define foundation elements
        self._constraints()  # Define constraints

    def _nodes(self):
        """
        -----------------------------
        MODELLING OF STRUCTURAL NODES
        -----------------------------

        Notes:
            - Super-elevation is not considered in deck elements, it is always 0.

        """

        # ELEMENT DISCRETIZATION
        deck_ele_max_length = self.model['General']['Deck Element Discretization']  # Number of deck elements
        pier_ele_max_length = self.model['General']['Pier Element Discretization']  # Pier of deck elements

        # JOINT INFORMATION
        # TODO: instead of coordinates provide span lengths, radius of curvature, and skew angles at joints
        # TODO: retrieve DXs, DYs, DZs, skew angles based on these.
        xs = np.array(self.model['General']['Joint Xs']).astype('float')  # Joint coordinates, x
        ys = np.array(self.model['General']['Joint Ys']).astype('float')  # Joint coordinates, y
        zs = np.array(self.model['General']['Joint Zs']).astype('float')  # Joint coordinates, z

        #  ----------------------------------------------------------------------------
        #  1-) CREATE SPAN NODES
        #  ----------------------------------------------------------------------------
        self.num_spans = len(xs) - 1  # Number of spans
        self.S1Nodes = []  # Start nodes for spans
        self.S2Nodes = []  # End nodes for spans
        self.S1dv = []  # vertical distance from deck centroid S1Nodes[i] to deck bottom
        self.S2dv = []  # vertical distance from deck centroid S2Nodes[i] to deck bottom
        self.D1Nodes = {i: [] for i in range(1, self.num_spans + 1)}  # Start nodes for deck elements
        self.D2Nodes = {i: [] for i in range(1, self.num_spans + 1)}  # End nodes for deck elements
        self.skew = []  # Skew angles

        count = 0  # node counter
        for i in range(self.num_spans):
            # Save the vertical distances from span end nodes to bearing top nodes
            tag = self.model['General']['Decks'][i]
            dv = self.model[tag]['dv']
            self.S1dv.append(dv)
            self.S2dv.append(dv)

            # Calculate skew angles
            self.skew.append(np.round_(np.arctan((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])), 7))
            skew = self.skew[i]

            if i == 0:
                j1_tag = None
                j2_tag = self.model['General']['Connections'][i]
            elif i == self.num_spans - 1:
                j1_tag = self.model['General']['Connections'][-1]
                j2_tag = None
            else:
                j1_tag = self.model['General']['Connections'][i - 1]
                j2_tag = self.model['General']['Connections'][i]

            # Left end joint type
            if j1_tag and self.model[j1_tag]['Type'] == 'Continuous':
                l1_joint = 0  # Length of the joint is 0
            elif j1_tag:
                l1_joint = self.model[j1_tag]['L'] / 2
            else:
                l1_joint = 0  # Length of the joint is 0

            # Right end joint type
            if j2_tag and self.model[j2_tag]['Type'] == 'Continuous':
                l2_joint = 0  # Length of the joint is 0
            elif j2_tag:
                l2_joint = self.model[j2_tag]['L'] / 2
            else:
                l2_joint = 0  # Length of the joint is 0

            # Get span start (1) and end (2) coordinates
            if i == 0:
                coord_s1 = np.array([xs[i], ys[i], zs[i]])
                coord_s2 = np.array([xs[i + 1] - np.cos(skew) * l2_joint, ys[i + 1] - np.sin(skew) * l2_joint, zs[i + 1]])
            elif i == self.num_spans - 1:
                coord_s1 = np.array([xs[i] + np.cos(skew) * l1_joint, ys[i] + np.sin(skew) * l1_joint, zs[i]])
                coord_s2 = np.array([xs[i + 1], ys[i + 1], zs[i + 1]])
            else:
                coord_s1 = np.array([xs[i] + np.cos(skew) * l1_joint, ys[i] + np.sin(skew) * l1_joint, zs[i]])
                coord_s2 = np.array([xs[i + 1] - np.cos(skew) * l2_joint, ys[i + 1] - np.sin(skew) * l2_joint, zs[i + 1]])

            span_length = np.round_(get_distance(coord_s1, coord_s2), 7)
            num_deck_ele = int(np.ceil(span_length / deck_ele_max_length))
            d_xyz = (coord_s2 - coord_s1) / num_deck_ele

            # Start creating nodes
            for j in range(1, num_deck_ele + 1):

                if j == 1:  # Start of span
                    # Discontinuous or start of the first span
                    if not j1_tag or self.model[j1_tag]['Type'] != 'Continuous':
                        count += 1
                        node_i = int(str(count) + self.STag)
                        coord1 = np.round_(coord_s1.copy(), 5)
                        ops.node(node_i, *coord1)
                    # Continuous or start of the other spans
                    else:
                        node_i = self.S2Nodes[-1]
                        coord1 = np.array(ops.nodeCoord(node_i))
                    self.S1Nodes.append(node_i)

                else:  # Rest of the interior nodes
                    node_i = self.D2Nodes[i + 1][-1]
                    coord1 = np.array(ops.nodeCoord(node_i))
                count += 1
                node_j = int(str(count) + self.STag)
                coord2 = coord1 + d_xyz
                ops.node(node_j, *np.round_(coord2, 5))
                self.D1Nodes[i + 1].append(node_i)
                self.D2Nodes[i + 1].append(node_j)
            self.S2Nodes.append(node_j)

        #  ----------------------------------------------------------------------------
        #  2-) CREATE LINK SLAB NODES IF THERE ARE ANY
        #  ----------------------------------------------------------------------------
        self.JointNodes = {i: [] for i in range(1, self.num_spans)}
        count = 0
        for i in range(self.num_spans - 1):
            j_tag = self.model['General']['Connections'][i]
            # Check if there is a link slab
            if self.model[j_tag]['Type'] == 'Link Slab':
                dv_joint = self.model[j_tag]['dv']
                if dv_joint > 0:
                    coord1 = ops.nodeCoord(self.S2Nodes[i])
                    coord1[2] += dv_joint
                    coord2 = ops.nodeCoord(self.S1Nodes[i + 1])
                    coord2[2] += dv_joint
                    count += 1
                    node_i = int(str(count) + self.JTag)
                    count += 1
                    node_j = int(str(count) + self.JTag)
                    ops.node(node_i, *np.round_(coord1, 5))
                    ops.node(node_j, *np.round_(coord2, 5))
                    # Nodes for rigid connections
                    self.RigidLinkNodes.append([self.S2Nodes[i], node_i])
                    self.RigidLinkNodes.append([self.S1Nodes[i + 1], node_j])
                else:
                    node_i = self.S2Nodes[i]
                    node_j = self.S1Nodes[i + 1]
                self.JointNodes[i + 1] = [[node_i, node_j]]
            elif self.model[j_tag]['Type'] == 'Discontinuous':
                width_deck = self.model[j_tag]['w']
                node_L = self.S2Nodes[i]
                skew_L = self.skew[i]
                node_R = self.S1Nodes[i + 1]
                skew_R = self.skew[i + 1]
                coords_L = np.array(ops.nodeCoord(node_L))
                coords_R = np.array(ops.nodeCoord(node_R))

                coords_L1 = coords_L.copy()
                coords_L2 = coords_L.copy()
                coords_R1 = coords_R.copy()
                coords_R2 = coords_R.copy()

                coords_L1[0] = coords_L[0] - np.cos(skew_L + np.pi / 2) * width_deck / 2
                coords_L1[1] = coords_L[1] - np.sin(skew_L + np.pi / 2) * width_deck / 2
                coords_L2[0] = coords_L[0] + np.cos(skew_L + np.pi / 2) * width_deck / 2
                coords_L2[1] = coords_L[1] + np.sin(skew_L + np.pi / 2) * width_deck / 2
                coords_R1[0] = coords_R[0] - np.cos(skew_R + np.pi / 2) * width_deck / 2
                coords_R1[1] = coords_R[1] - np.sin(skew_R + np.pi / 2) * width_deck / 2
                coords_R2[0] = coords_R[0] + np.cos(skew_R + np.pi / 2) * width_deck / 2
                coords_R2[1] = coords_R[1] + np.sin(skew_R + np.pi / 2) * width_deck / 2

                count += 1
                node_L1 = int(str(count) + self.JTag)
                count += 1
                node_L2 = int(str(count) + self.JTag)
                count += 1
                node_R1 = int(str(count) + self.JTag)
                count += 1
                node_R2 = int(str(count) + self.JTag)
                ops.node(node_L1, *np.round_(coords_L1, 5))
                ops.node(node_L2, *np.round_(coords_L2, 5))
                ops.node(node_R1, *np.round_(coords_R1, 5))
                ops.node(node_R2, *np.round_(coords_R2, 5))
                self.JointNodes[i + 1] = [[node_L1, node_R1], [node_L2, node_R2]]

                self.RigidLinkNodes.append([node_L, node_L1])
                self.RigidLinkNodes.append([node_L, node_L2])
                self.RigidLinkNodes.append([node_R, node_R1])
                self.RigidLinkNodes.append([node_R, node_R2])

                # node_i = self.S2Nodes[i]
                # node_j = self.S1Nodes[i + 1]
                # self.JointNodes[i + 1] = [[node_i, node_j]]

            elif self.model[j_tag]['Type'] == 'Expansion Joint':
                # Save Joint nodes
                node_i = self.S2Nodes[i]
                node_j = self.S1Nodes[i + 1]
                self.JointNodes[i + 1] = [[node_i, node_j]]

        #  ----------------------------------------------------------------------------
        #  3-) BEARING NODES
        #  ----------------------------------------------------------------------------
        # Get unique bearing types
        confs = list(set(self.model['General']['Bearings']))
        types = []
        for i in range(len(confs)):
            conf_tag = confs[i]
            if 'bearings' in self.model[conf_tag].keys():
                types.extend(self.model[conf_tag]['bearings'])
            if 'bearingsL' in self.model[conf_tag].keys():
                types.extend(self.model[conf_tag]['bearingsL'])
            if 'bearingsR' in self.model[conf_tag].keys():
                types.extend(self.model[conf_tag]['bearingsR'])
        types = list(set(types))

        # Dictionary containing bottom and top bearing nodes for unique bearing types
        self.bearing_nodes = {bearing_type: [] for bearing_type in types}
        # These nodes need to be known to provide connections between bent-cap and bearing nodes
        self.bearing_botNodes = {}
        # Dictionary containing shear key nodes
        self.shear_key_nodes = {}

        # Loop through each joint
        count = 0
        for i in range(self.num_spans + 1):
            # initialize the list for bottom bearing nodes at ith joint
            self.bearing_botNodes[i] = []
            # initialize the list for shear key nodes at ith joint
            self.shear_key_nodes[i] = {'left': [], 'right': []}

            # RNode: span node on the right side of joint
            # LNode: span node on the left side of joint
            # TODO: e_uns is the limit for unseating, for now I assume large unseating length at abutments
            if i == 0:
                RNode = self.S1Nodes[i]
                Rdv = self.S1dv[i]
                Rskew = self.skew[0]
                LNode = None
                Ldv = None
                Lskew = None
                e_uns = 10
            elif i == self.num_spans:
                RNode = None
                Rdv = None
                Rskew = None
                LNode = self.S2Nodes[i - 1]
                Ldv = self.S2dv[i - 1]
                Lskew = self.skew[-1]
                e_uns = 10
            else:
                RNode = self.S1Nodes[i]
                Rdv = self.S1dv[i]
                Rskew = self.skew[i]
                LNode = self.S2Nodes[i - 1]
                Ldv = self.S2dv[i - 1]
                Lskew = self.skew[i - 1]
                # Unseating length
                j_tag = self.model['General']['Connections'][i - 1]
                conf_tag = self.model['General']['Bearings'][i]
                dh = self.model[conf_tag]['dh']
                cap_tag = self.model['General']['Bent Caps'][i - 1]
                cap_w = self.model[cap_tag]['width']
                if self.model[j_tag]['Type'] == 'Continuous':
                    L_joint = 0
                else:
                    L_joint = self.model[j_tag]['L']
                e_uns = cap_w / 2 - L_joint / 2 - dh

            # For left side on joint
            if LNode and RNode != LNode:
                Lcoord = ops.nodeCoord(LNode)
                # Get bearing configuration
                conf_tag = self.model['General']['Bearings'][i]
                dh = self.model[conf_tag]['dh']
                spacing = self.model[conf_tag]['spacing']
                if 'bearings' in self.model[conf_tag].keys():
                    bearings = self.model[conf_tag]['bearings']
                elif 'bearingsL' in self.model[conf_tag].keys():
                    bearings = self.model[conf_tag]['bearingsL']
                bh = self.model[conf_tag]['h']
                # Get the starting coordinates
                num_bearings = len(bearings)
                dist_mid = spacing * (num_bearings - 1) / 2
                coords_mid = [Lcoord[0] - np.cos(Lskew) * dh, Lcoord[1] - np.sin(Lskew) * dh, Lcoord[2] - Ldv]
                x0 = round(coords_mid[0] - np.cos(Lskew + np.pi / 2) * dist_mid, 5)
                y0 = round(coords_mid[1] - np.sin(Lskew + np.pi / 2) * dist_mid, 5)
                z0 = round(coords_mid[2], 5)
                z01 = round(coords_mid[2] - bh, 5)

                botNodes = []
                for j in range(num_bearings):
                    bearing_type = bearings[j]
                    count += 1
                    BotNode = int(str(count) + self.BTag)
                    count += 1
                    TopNode = int(str(count) + self.BTag)
                    ops.node(BotNode, x0, y0, z01)
                    ops.node(TopNode, x0, y0, z0)
                    # Moving to the next bearing, update the coordinates
                    x0 = round(x0 + np.cos(Lskew + np.pi / 2) * spacing, 5)
                    y0 = round(y0 + np.sin(Lskew + np.pi / 2) * spacing, 5)
                    # Saving the rigid links
                    self.RigidLinkNodes.append([LNode, TopNode])
                    # Saving the bearing node names, skew angle w.r.t global x-axis (1) and joint id
                    self.bearing_nodes[bearing_type].append([BotNode, TopNode, Lskew, i, e_uns, 'left'])
                    botNodes.append(BotNode)
                    # Saving the shear key nodes
                    if j in [0, num_bearings - 1]:
                        self.shear_key_nodes[i]['left'].append([BotNode, TopNode, Lskew])
                self.bearing_botNodes[i].append(botNodes)

            # For right side on joint
            if RNode:
                Rcoord = ops.nodeCoord(RNode)
                # Get bearing configuration
                conf_tag = self.model['General']['Bearings'][i]
                dh = self.model[conf_tag]['dh']
                spacing = self.model[conf_tag]['spacing']
                if 'bearings' in self.model[conf_tag].keys():
                    bearings = self.model[conf_tag]['bearings']
                elif 'bearingsR' in self.model[conf_tag].keys():
                    bearings = self.model[conf_tag]['bearingsR']
                bh = self.model[conf_tag]['h']
                # Get the starting coordinates
                num_bearings = len(bearings)
                dist_mid = spacing * (num_bearings - 1) / 2
                coords_mid = [Rcoord[0] + np.cos(Rskew) * dh, Rcoord[1] + np.sin(Rskew) * dh, Rcoord[2] - Rdv]
                x0 = round(coords_mid[0] - np.cos(Rskew + np.pi / 2) * dist_mid, 5)
                y0 = round(coords_mid[1] - np.sin(Rskew + np.pi / 2) * dist_mid, 5)
                z0 = round(coords_mid[2], 5)
                z01 = round(coords_mid[2] - bh, 5)

                botNodes = []
                for j in range(num_bearings):
                    bearing_type = bearings[j]
                    count += 1
                    BotNode = int(str(count) + self.BTag)
                    count += 1
                    TopNode = int(str(count) + self.BTag)
                    ops.node(BotNode, x0, y0, z01)
                    ops.node(TopNode, x0, y0, z0)
                    # Moving to the next bearing, update the coordinates
                    x0 = round(x0 + np.cos(Rskew + np.pi / 2) * spacing, 5)
                    y0 = round(y0 + np.sin(Rskew + np.pi / 2) * spacing, 5)
                    # Saving the rigid links
                    self.RigidLinkNodes.append([RNode, TopNode])
                    # Saving the bearing node names, skew angle w.r.t global x-axis (1) and joint id
                    self.bearing_nodes[bearing_type].append([BotNode, TopNode, Rskew, i, e_uns, 'right'])
                    botNodes.append(BotNode)
                    # Saving the shear key nodes
                    if j in [0, num_bearings - 1]:
                        self.shear_key_nodes[i]['right'].append([BotNode, TopNode, Rskew])
                self.bearing_botNodes[i].append(botNodes)

        #  ----------------------------------------------------------------------------
        #  4-) BENT NODES
        #  ----------------------------------------------------------------------------
        self.BentNodes = {}
        count = 0
        # Loop through each joint
        for i in range(1, self.num_spans):
            skew = (self.skew[i] + self.skew[i - 1]) / 2
            Cap_tag = self.model['General']['Bent Caps'][i - 1]
            bent_tag = self.model['General']['Bents'][i - 1]
            Hcap = self.model[Cap_tag]['height']
            Hpier = self.model[bent_tag]['height']
            Npier = len(self.model[bent_tag]['sections'])
            spacing = self.model[bent_tag]['spacing']
            num_pier_ele = int(np.ceil(Hpier / pier_ele_max_length))

            # calculate the centroid of nodes at bent top
            if len(self.bearing_botNodes[i]) == 2:
                botnodes = self.bearing_botNodes[i][0] + self.bearing_botNodes[i][1]
            else:
                botnodes = self.bearing_botNodes[i][0]
            coords = np.zeros((len(botnodes), 3))
            for j in range(len(botnodes)):
                coords[j] = np.array(ops.nodeCoord(botnodes[j]))
            coords_centr = np.mean(coords, axis=0)
            coords_centr[0] = xs[i]
            coords_centr[1] = ys[i]
            coords_centr[2] = coords_centr[2] - Hcap

            # Define define nodal coordinates for the first pier in bent
            dist_mid = spacing * (Npier - 1) / 2
            z_top = coords_centr[2]
            z_bot = coords_centr[2] - Hpier
            x0 = coords_centr[0] - np.cos(skew + np.pi / 2) * dist_mid
            y0 = coords_centr[1] - np.sin(skew + np.pi / 2) * dist_mid
            CoordBot = np.array([x0, y0, z_bot])
            CoordTop = np.array([x0, y0, z_top])

            # initialize the node list with the current bent type
            self.BentNodes[i] = []
            # Loop through number of piers per bent
            for j in range(Npier):
                dXYZ = (CoordTop - CoordBot) / num_pier_ele
                pier_nodes = []

                for k in range(num_pier_ele):

                    if k == 0:
                        count += 1
                        node_i = int(str(count) + self.PTag)
                        ops.node(node_i, *np.round_(CoordBot, 5))
                        pier_nodes.append(node_i)
                        coord_j = CoordBot.copy()

                    count += 1
                    node_j = int(str(count) + self.PTag)
                    coord_j = coord_j + dXYZ
                    ops.node(node_j, *np.round_(coord_j, 5))
                    pier_nodes.append(node_j)

                # save bent start, end nodes, rotation angle and bent id
                pier_nodes.append(skew)
                self.BentNodes[i].append(pier_nodes)

                CoordBot = np.array([CoordBot[0] + np.cos(skew + np.pi / 2) * spacing, CoordBot[1] + np.sin(skew + np.pi / 2) * spacing, CoordBot[2]])
                CoordTop = np.array([CoordBot[0], CoordBot[1], CoordTop[2]])

        #  ----------------------------------------------------------------------------
        #  5-) BENT CAP NODES
        #  ----------------------------------------------------------------------------
        count = 0
        self.CapNodes = {}
        self.CapLeftCoord = {}
        self.CapRightCoord = {}
        # Loop through each joint
        for i in range(1, self.num_spans):
            skew = (self.skew[i] + self.skew[i - 1]) / 2
            Cap_tag = self.model['General']['Bent Caps'][i - 1]
            Hcap = self.model[Cap_tag]['height']
            Lcap = self.model[Cap_tag]['length']

            # calculate coordinates of the centroid of nodes at bent top, and the exterior cap nodes
            if len(self.bearing_botNodes[i]) == 2:
                botnodes = self.bearing_botNodes[i][0] + self.bearing_botNodes[i][1]
            else:
                botnodes = self.bearing_botNodes[i][0]
            coords = np.zeros((len(botnodes), 3))
            for j in range(len(botnodes)):
                coords[j] = np.array(ops.nodeCoord(botnodes[j]))
            coords_centr = np.mean(coords, axis=0)
            coords_centr[0] = xs[i]
            coords_centr[1] = ys[i]
            coords_centr[2] = coords_centr[2] - Hcap / 2
            dist_mid = Lcap / 2
            x0 = coords_centr[0] - np.cos(skew + np.pi / 2) * dist_mid
            y0 = coords_centr[1] - np.sin(skew + np.pi / 2) * dist_mid
            x1 = coords_centr[0] + np.cos(skew + np.pi / 2) * dist_mid
            y1 = coords_centr[1] + np.sin(skew + np.pi / 2) * dist_mid
            Coord0 = np.round_([x0, y0, coords_centr[2]], 5)
            coord1 = np.round_([x1, y1, coords_centr[2]], 5)
            # Save this information because it is going to be useful while generating the bent cap elements
            self.CapLeftCoord[i] = Coord0
            self.CapRightCoord[i] = coord1

            self.CapNodes[i] = []
            if len(botnodes) == 1:  # just create a central node
                count += 1
                nodeTag = int(str(count) + self.CTag)
                ops.node(nodeTag, *np.round_(coords_centr, 5))
                self.RigidLinkNodes.append([nodeTag, self.bearing_botNodes[i][0][0]])
                for pier_nd in self.BentNodes[i]:
                    self.RigidLinkNodes.append([nodeTag, pier_nd[-2]])
                self.CapNodes[i].append(nodeTag)
            else:
                cap_coordinates = []
                # create cap nodes which are rigidly connected to bearings
                for j in range(len(self.bearing_botNodes[i][0])):
                    count += 1
                    nodeTag = int(str(count) + self.CTag)
                    self.RigidLinkNodes.append([nodeTag, self.bearing_botNodes[i][0][j]])
                    self.CapNodes[i].append(nodeTag)

                    if len(self.bearing_botNodes[i]) == 2:
                        self.RigidLinkNodes.append([nodeTag, self.bearing_botNodes[i][1][j]])
                        coords_bL = ops.nodeCoord(self.bearing_botNodes[i][0][j])
                        coords_bR = ops.nodeCoord(self.bearing_botNodes[i][1][j])
                        coords = np.mean(np.array([coords_bL, coords_bR]), axis=0)
                    else:
                        coords = ops.nodeCoord(self.bearing_botNodes[i][0][j])
                    coords[2] = coords[2] - Hcap / 2
                    coords = np.round_(coords, 5)
                    ops.node(nodeTag, *coords)
                    cap_coordinates.append(coords)

                # create cap nodes which are rigidly connected to piers
                # do not create if they coincide with previously generated nodes
                for j in range(len(self.BentNodes[i])):
                    pier_node = self.BentNodes[i][j][-2]
                    bc_coords = ops.nodeCoord(pier_node)
                    bc_coords[2] = bc_coords[2] + Hcap / 2
                    bc_coords = np.round_(bc_coords, 5)
                    if np.any(np.all(bc_coords == cap_coordinates, axis=1)):
                        idx = np.where(np.all(bc_coords == cap_coordinates, axis=1))[0][0]
                        bc_node = self.CapNodes[i][idx]
                    else:
                        count += 1
                        bc_node = int(str(count) + self.CTag)
                        ops.node(bc_node, *bc_coords)
                        cap_coordinates.append(bc_coords)
                        self.CapNodes[i].append(bc_node)
                    self.RigidLinkNodes.append([bc_node, pier_node])

                # create start and end cap nodes (do not create if they coincide with previously generated nodes)
                if not np.any(np.all(Coord0 == cap_coordinates, axis=1)):
                    count += 1
                    bc_node = int(str(count) + self.CTag)
                    ops.node(bc_node, *Coord0)
                    self.CapNodes[i].append(bc_node)
                if not np.any(np.all(coord1 == cap_coordinates, axis=1)):
                    count += 1
                    bc_node = int(str(count) + self.CTag)
                    ops.node(bc_node, *coord1)
                    self.CapNodes[i].append(bc_node)

        #  ----------------------------------------------------------------------------
        #  6-) ABUTMENT NODES
        #  ----------------------------------------------------------------------------
        abut_conf = self.model['General']['Abutments']
        Njoints = self.num_spans + 1
        # The retained node for start abutment
        count_node = 1
        AB1_Rnode = int(str(count_node) + self.ATag)
        summation = 0
        Nbearings = len(self.bearing_botNodes[0][0])
        for i in range(len(self.bearing_botNodes[0][0])):
            Bnode = self.bearing_botNodes[0][0][i]
            self.RigidLinkNodes.append([AB1_Rnode, Bnode])
            summation += np.array(ops.nodeCoord(Bnode))
        AB1_Rnode_coords = np.round_(summation / Nbearings, 7)
        ops.node(AB1_Rnode, *AB1_Rnode_coords)

        # The retained node for end abutment
        count_node += 1
        AB2_Rnode = int(str(count_node) + self.ATag)
        summation = 0
        Nbearings = len(self.bearing_botNodes[Njoints - 1][0])
        for i in range(len(self.bearing_botNodes[Njoints - 1][0])):
            Bnode = self.bearing_botNodes[Njoints - 1][0][i]
            self.RigidLinkNodes.append([AB2_Rnode, Bnode])
            summation += np.array(ops.nodeCoord(Bnode))
        AB2_Rnode_coords = np.round_(summation / Nbearings, 7)
        ops.node(AB2_Rnode, *AB2_Rnode_coords)

        if self.model[abut_conf]['Type'] == 'Fixed':
            # self.fixed_backfill = [self.bearing_botNodes[0][0], self.bearing_botNodes[Njoints - 1][0]]
            # The retained node for start abutment
            self.fixed_backfill = [[AB1_Rnode], [AB2_Rnode]]
        else:
            coords1 = ops.nodeCoord(AB1_Rnode)
            coords2 = ops.nodeCoord(AB2_Rnode)

            count_node += 1
            AB1_Fnode = int(str(count_node) + self.ATag)  # Tag for fixed node at start abutment
            count_node += 1
            AB2_Fnode = int(str(count_node) + self.ATag)  # Tag for fixed node at end abutment

            ops.node(AB1_Fnode, *coords1)
            ops.node(AB2_Fnode, *coords2)

            self.fixed_backfill = [[AB1_Fnode], [AB2_Fnode]]
            self.Abutment_NodesSpring = [[AB1_Fnode, AB1_Rnode], [AB2_Fnode, AB2_Rnode]]

        #  ----------------------------------------------------------------------------
        #  7-) FOUNDATION NODES
        #  ----------------------------------------------------------------------------
        # TODO: For now I am not going to do anything here since it is rather complex,
        #  we only have fixed base modelling approach atm
        # Backfill spring start nodes
        for i in range(2):
            for node in self.fixed_backfill[i]:
                self.fixed.append([node, 1, 1, 1, 1, 1, 1])

        # Bent start nodes
        self.fixed_bent = {}
        for i in self.BentNodes:
            self.fixed_bent[i] = []
            for nodes in self.BentNodes[i]:
                self.fixed_bent[i].append(nodes[0])
                self.fixed.append([nodes[0], 1, 1, 1, 1, 1, 1])

    def _joints(self):
        """
        ----------------------------------------
        MODELLING OF SPAN TO SPAN JOINT ELEMENTS
        ----------------------------------------
        """
        self.EleIDsJoint = {i: [] for i in range(1, self.num_spans)}  # Element Tags

        # Superimposed dead load per length
        wSDL = self.model['General']['wSDL']

        # Create the sections for Link Slabs
        tags = list(set(self.model['General']['Connections']))
        IntTags = {}
        wLink = {}
        transf_tags = {}
        PoundingMats = {}
        for i in range(len(tags)):
            j_tag = tags[i]
            if self.model[j_tag]['Type'] == 'Link Slab':
                A = self.model[j_tag]['A']  # Slab area
                E = self.model[j_tag]['E']  # Young's modulus
                G = self.model[j_tag]['G']  # Shear modulus
                J = self.model[j_tag]['J']  # Polar moment of inertia
                Iz = self.model[j_tag]['Iz']  # Moment of inertia in z-dir
                Iy = self.model[j_tag]['Iy']  # Moment of inertia in y-dir
                self.EndSecTag += 1
                self.EndIntTag += 1
                ops.section('Elastic', self.EndSecTag, E, A, Iz, Iy, G, J)
                ops.beamIntegration('Legendre', self.EndIntTag, self.EndSecTag, 2)
                IntTags[j_tag] = self.EndIntTag + 0
                wLink[j_tag] = gamma_rc * A + wSDL  # Weight of link slab per length
            elif self.model[j_tag]['Type'] == 'Discontinuous':
                # Create linear pounding material, (Muthukumar and DesRoches, 2006)
                self.EndMatTag += 1
                gap = self.model[j_tag]['L']
                ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, 1e7, -1e7, -gap, -0.001, 'NoDamage')
                PoundingMats[j_tag] = self.EndMatTag + 0
        count = 0
        for i in range(self.num_spans - 1):
            j_tag = self.model['General']['Connections'][i]
            if self.model[j_tag]['Type'] == 'Continuous':
                pass
            else:
                # get skew angle
                skew = (self.skew[i] + self.skew[i + 1]) / 2
                for nodes in self.JointNodes[i + 1]:
                    # Get Joint Element Nodes
                    nodeI, nodeJ = nodes
                    Coord1 = ops.nodeCoord(nodeI)
                    Coord2 = ops.nodeCoord(nodeJ)
                    # Compute vectors to define geometric transformation
                    Vx = np.asarray([Coord2[0] - Coord1[0], Coord2[1] - Coord1[1], Coord2[2] - Coord1[2]])
                    Vx = Vx / np.sqrt(Vx.dot(Vx))
                    Vy = np.asarray([np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0])  # -> super-elevation = 0
                    Vy = Vy / np.sqrt(Vy.dot(Vy))
                    Vz = np.cross(Vx, Vy)
                    Vz = Vz / np.sqrt(Vz.dot(Vz))
                    # Element tag
                    count += 1
                    eleTag = int(str(count) + self.JointTag)

                    if self.model[j_tag]['Type'] == 'Discontinuous':
                        matTag = PoundingMats[j_tag]
                        gap = self.model[j_tag]['L']
                        if gap == 0:
                            ops.element('zeroLength', eleTag, nodeI, nodeJ, '-mat', matTag, '-dir', 1, '-orient',
                                        *Vx.tolist(), *Vy.tolist())
                        else:
                            ops.element('twoNodeLink', eleTag, nodeI, nodeJ, '-mat', matTag, '-dir', 1, '-orient',
                                        *Vy.tolist())

                    # LINEAR ELASTIC LINK SLAB ELEMENTS
                    elif self.model[j_tag]['Type'] == 'Link Slab':
                        # Weight of the element
                        wTOT = wLink[j_tag]

                        # Beam element geometric transformation
                        props = Vz.tolist()
                        TransfTag = 0
                        for key, val in transf_tags.items():
                            if props == val:
                                TransfTag = key
                        if TransfTag == 0:
                            TransfTag = self.EndTransfTag + 1
                            self.EndTransfTag += 1
                            transf_tags[self.EndTransfTag] = props
                            ops.geomTransf('Linear', self.EndTransfTag, *Vz.tolist())

                        # Slab elements
                        ops.element('dispBeamColumn', eleTag, nodeI, nodeJ, TransfTag, IntTags[j_tag], '-mass',
                                    wTOT / g, self.mass_type)

                        # Save gravity loads and element IDs at each joint
                        self.DistributedLoads.append(
                            ['-ele', eleTag, '-type', '-beamUniform', -wTOT * Vy[2], -wTOT * Vz[2], -wTOT * Vx[2]])
                        self.EleIDsJoint[i + 1] = eleTag

                    # TODO:-1 Add expansion joints here, two node link elements
                    # elif self.model[j_tag]['Type'] == 'Expansion Joint':

    def _decks(self):
        """
        --------------------------
        DECK MODELLING
        --------------------------

        Notes:
            - Super-elevation is not considered in deck elements, it is always 0.

        """

        # Superimposed dead load per length
        wSDL = self.model['General']['wSDL']

        # Create the sections
        tags = list(set(self.model['General']['Decks']))
        IntTags = {}
        wDecks = {}
        transf_tags = {}
        for i in range(len(tags)):
            tag = tags[i]
            A = self.model[tag]['A']  # Deck area
            E = self.model[tag]['E']  # Young's modulus
            G = self.model[tag]['G']  # Shear modulus
            J = self.model[tag]['J']  # Polar moment of inertia
            Iz = self.model[tag]['Iz']  # Moment of inertia in z-dir
            Iy = self.model[tag]['Iy']  # Moment of inertia in y-dir
            self.EndSecTag += 1
            self.EndIntTag += 1
            ops.section('Elastic', self.EndSecTag, E, A, Iz, Iy, G, J)
            ops.beamIntegration('Legendre', self.EndIntTag, self.EndSecTag, 2)
            IntTags[tag] = self.EndIntTag
            wDecks[tag] = gamma_rc * A + wSDL  # Weight of deck per length

        # Necessary info to save
        self.EleIDsDeck = {i: [] for i in range(1, self.num_spans + 1)}  # Element Tags

        for i in range(self.num_spans):
            tag = self.model['General']['Decks'][i]
            # Deck loads
            wTOT = wDecks[tag]  # Total weight assigned on deck elements per length
            intTag = IntTags[tag]
            skew = self.skew[i]

            # Store these vectors, bridge could have in plane curvature as well.
            Coord1 = ops.nodeCoord(self.S1Nodes[i])
            Coord2 = ops.nodeCoord(self.S2Nodes[i])
            Vx = np.asarray([Coord2[0] - Coord1[0], Coord2[1] - Coord1[1], Coord2[2] - Coord1[2]])
            Vx = np.round_(Vx / np.sqrt(Vx.dot(Vx)), 6)
            Vy = np.asarray([np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0])  # -> super-elevation = 0
            Vy = np.round_(Vy / np.sqrt(Vy.dot(Vy)), 6)
            Vz = np.cross(Vx, Vy)
            Vz = np.round_(Vz / np.sqrt(Vz.dot(Vz)), 6)
            props = Vz.tolist()
            TransfTag = 0
            for key, val in transf_tags.items():
                if props == val:
                    TransfTag = key
            if TransfTag == 0:
                TransfTag = self.EndTransfTag + 1
                self.EndTransfTag += 1
                transf_tags[self.EndTransfTag] = props
                ops.geomTransf('Linear', self.EndTransfTag, *Vz.tolist())

            for j in range(len(self.D1Nodes[i + 1])):
                eleTag = int(str(i + 1) + str(j + 1) + self.DeckTag)
                nodeI = self.D1Nodes[i + 1][j]
                nodeJ = self.D2Nodes[i + 1][j]
                ops.element('dispBeamColumn', eleTag, nodeI, nodeJ, TransfTag,
                            intTag, '-mass', wTOT / g, self.mass_type)

                # Save gravity loads and element IDs at each span
                self.DistributedLoads.append(
                    ['-ele', eleTag, '-type', '-beamUniform', -wTOT * Vy[2], -wTOT * Vz[2], -wTOT * Vx[2]])
                self.EleIDsDeck[i + 1].append(eleTag)

    def _create_bearing_mats(self):
        """
        -----------------
        BEARING MATERIALS
        -----------------
        """
        # Get unique uni-axial material tags
        mtags = []
        ele_types = ['elastomericBearingBoucWen', 'elastomericBearingPlasticity', 'flatSliderBearing',
                     'singleFPBearing']
        for tag in self.bearing_nodes:
            eleType = self.model[tag]['model']
            if eleType == 'twoNodeLink':  # Two Node Link element
                mtags.append(self.model[tag]['dir-1'])
                mtags.append(self.model[tag]['dir-2'])
                mtags.append(self.model[tag]['dir-3'])
                mtags.append(self.model[tag]['dir-4'])
                mtags.append(self.model[tag]['dir-5'])
                mtags.append(self.model[tag]['dir-6'])

            elif eleType in ele_types:
                mtags.append(self.model[tag]['dir-1'])
                mtags.append(self.model[tag]['dir-4'])
                mtags.append(self.model[tag]['dir-5'])
                mtags.append(self.model[tag]['dir-6'])

        mtags = list(set(mtags))

        # Create materials
        materials = {}
        for mat in mtags:
            self.EndMatTag += 1
            materials[mat] = self.EndMatTag
            ops.uniaxialMaterial(self.model[mat][0], self.EndMatTag, *self.model[mat][1:])

        return materials

    def _create_bearing_friction_models(self):
        """
        -----------------------
        BEARING FRICTION MODELS
        -----------------------
        """
        # Get unique uni-axial material tags
        tags = []
        eleTypes = ['flatSliderBearing', 'singleFPBearing']
        for tag in self.bearing_nodes:
            eleType = self.model[tag]['model']
            if eleType in eleTypes:
                tags.append(self.model[tag]['friction_model'])

        tags = list(set(tags))

        # Create friction models
        frictions = {}
        frn_tag = 0
        for tag in tags:
            frn_tag += 1
            frictions[tag] = frn_tag
            frn_type = self.model[tag][0]
            frn_args = self.model[tag][1:]
            ops.frictionModel(frn_type, frn_tag, *frn_args)

        return frictions

    def _get_bearing_mat_args(self, tag, materials, frictions):

        optionalArgs = None
        matArgs = None
        matTags = []

        if self.model[tag]['model'] == 'twoNodeLink':
            for i in range(1, 7):
                mat = self.model[tag]['dir-' + str(i)]
                matTags.append(materials[mat])
            # note that local x is in z direction
            matArgs = ['-mat', *matTags, '-dir', 1, 2, 3, 4, 5, 6]

        elif self.model[tag]['model'] == 'flatSliderBearing':
            for i, dof, in zip([1, 4, 5, 6], ['-P', '-T', '-My', '-Mz']):
                mat = self.model[tag]['dir-' + str(i)]
                matTags.extend([dof, materials[mat]])
            frn_name = self.model[tag]['friction_model']
            frn_tag = frictions[frn_name]  # tag of friction model
            k_init = self.model[tag]['k_init']  # initial elastic stiffness in local shear direction
            # note that local x is in global direction 3
            matArgs = [frn_tag, k_init, *matTags]
            optionalArgs = [0.0, 0.0, '-iter', 20, 1e-8]

        elif self.model[tag]['model'] == 'singleFPBearing':
            for i, dof, in zip([1, 4, 5, 6], ['-P', '-T', '-My', '-Mz']):
                mat = self.model[tag]['dir-' + str(i)]
                matTags.extend([dof, materials[mat]])
            frn_name = self.model[tag]['friction_model']
            frn_tag = frictions[frn_name]  # tag of friction model
            k_init = self.model[tag]['k_init']  # initial elastic stiffness in local shear direction
            r_eff = self.model[tag]['r_eff']  # effective radius of concave sliding surface

            # note that local x is in global direction 3
            matArgs = [frn_tag, r_eff, k_init, *matTags]

        elif self.model[tag]['model'] == 'elastomericBearingBoucWen':
            for i, dof, in zip([1, 4, 5, 6], ['-P', '-T', '-My', '-Mz']):
                mat = self.model[tag]['dir-' + str(i)]
                matTags.extend([dof, materials[mat]])
            k_init = self.model[tag]['k_init']  # initial elastic stiffness in local shear direction
            Fb = self.model[tag]['Fb']  # characteristic strength
            alpha1 = self.model[tag]['alpha1']  # post yield stiffness ratio of linear hardening component
            alpha2 = self.model[tag]['alpha2']  # post yield stiffness ratio of non-linear hardening component
            mu = self.model[tag]['mu']  # exponent of non-linear hardening component
            # yielding exponent (sharpness of hysteresis loop corners) (default = 1.0)
            if 'eta' in self.model[tag]:
                eta = self.model[tag]['eta']
            else:
                eta = 1.0
            # first hysteresis shape parameter (default = 0.5)
            if 'beta' in self.model[tag]:
                beta = self.model[tag]['beta']
            else:
                beta = 0.5
            # second hysteresis shape parameter (default = 0.5)
            if 'gamma' in self.model[tag]:
                gamma = self.model[tag]['gamma']
            else:
                gamma = 0.5
            # note that local x is in global direction 3
            matArgs = [k_init, Fb, alpha1, alpha2, mu, eta, beta, gamma, *matTags]

        elif self.model[tag]['model'] == 'elastomericBearingPlasticity':
            for i, dof, in zip([1, 4, 5, 6], ['-P', '-T', '-My', '-Mz']):
                mat = self.model[tag]['dir-' + str(i)]
                matTags.extend([dof, materials[mat]])
            k_init = self.model[tag]['k_init']  # initial elastic stiffness in local shear direction
            Fb = self.model[tag]['Fb']  # characteristic strength
            alpha1 = self.model[tag]['alpha1']  # post yield stiffness ratio of linear hardening component
            alpha2 = self.model[tag]['alpha2']  # post yield stiffness ratio of non-linear hardening component
            mu = self.model[tag]['mu']  # exponent of non-linear hardening component
            matArgs = [k_init, Fb, alpha1, alpha2, mu, *matTags]

        elif self.model[tag]['model'] == 'ElastomericX':
            Fy = self.model[tag]['Fy']  # yield strength
            alpha = self.model[tag]['alpha']  # post-yield stiffness ratio
            G = self.model[tag]['Gr']  # shear modulus of elastomeric bearing
            K = self.model[tag]['Kbulk']  # bulk modulus of rubber
            D1 = self.model[tag]['D1']  # internal diameter
            D2 = self.model[tag]['D2']  # outer diameter (excluding cover thickness)
            ts = self.model[tag]['ts']  # single steel shim layer thickness
            tr = self.model[tag]['tr']  # single rubber layer thickness
            nr = self.model[tag]['nr']  # number of rubber layers

            if 'kc' not in self.model[tag]:  # cavitation parameter (optional, default = 10.0)
                kc = 10
            else:
                kc = self.model[tag]['kc']
            if 'PhiM' not in self.model[tag]:  # damage parameter (optional, default = 0.5)
                PhiM = 0.5
            else:
                PhiM = self.model[tag]['PhiM']
            if 'ac' not in self.model[tag]:  # strength reduction parameter (optional, default = 1.0)
                ac = 1.0
            else:
                ac = self.model[tag]['ac']
            # shear distance from iNode as fraction of the element length (optional, default = 0.5)
            if 'sDratio' not in self.model[tag]:
                sDratio = 0.5
            else:
                sDratio = self.model[tag]['sDratio']
            if 'tc' not in self.model[tag]:  # cover thickness (optional, default = 0.0)
                tc = 0.0
            else:
                tc = self.model[tag]['tc']
            if 'cd' not in self.model[tag]:  # Viscous damping parameter (optional, default = 0.0)
                cd = 0.0
            else:
                cd = self.model[tag]['cd']
            # For strength and stiffness degradation effects
            if 'tag1' not in self.model[tag]:  # Tag to include cavitation and post-cavitation (optional, default = 0)
                tag1 = 0.0
            else:
                tag1 = self.model[tag]['tag1']
            if 'tag2' not in self.model[tag]:  # Tag to include buckling load variation (optional, default = 0)
                tag2 = 0.0
            else:
                tag2 = self.model[tag]['tag2']
            if 'tag3' not in self.model[tag]:  # Tag to include horizontal stiffness variation (optional, default = 0)
                tag3 = 0.0
            else:
                tag3 = self.model[tag]['tag3']
            if 'tag4' not in self.model[tag]:  # Tag to include vertical stiffness variation (optional, default = 0)
                tag4 = 0.0
            else:
                tag4 = self.model[tag]['tag4']

            mass = 0.0  # element mass (optional, default = 0.0)

            matArgs = [Fy, alpha, G, K, D1, D2, ts, tr, nr]
            optionalArgs = [kc, PhiM, ac, sDratio, mass, cd, tc, tag1, tag2, tag3, tag4]

        elif self.model[tag]['model'] == 'LeadRubberX':
            Fy = self.model[tag]['Fy']  # yield strength
            alpha = self.model[tag]['alpha']  # post-yield stiffness ratio
            G = self.model[tag]['Gr']  # shear modulus of elastomeric bearing
            K = self.model[tag]['Kbulk']  # bulk modulus of rubber
            D1 = self.model[tag]['D1']  # internal diameter
            D2 = self.model[tag]['D2']  # outer diameter (excluding cover thickness)
            ts = self.model[tag]['ts']  # single steel shim layer thickness
            tr = self.model[tag]['tr']  # single rubber layer thickness
            nr = self.model[tag]['nr']  # number of rubber layers

            if 'kc' not in self.model[tag]:  # cavitation parameter (optional, default = 10.0)
                kc = 10
            else:
                kc = self.model[tag]['kc']
            if 'PhiM' not in self.model[tag]:  # damage parameter (optional, default = 0.5)
                PhiM = 0.5
            else:
                PhiM = self.model[tag]['PhiM']
            if 'ac' not in self.model[tag]:  # strength reduction parameter (optional, default = 1.0)
                ac = 1.0
            else:
                ac = self.model[tag]['ac']
            # shear distance from iNode as fraction of the element length (optional, default = 0.5)
            if 'sDratio' not in self.model[tag]:
                sDratio = 0.5
            else:
                sDratio = self.model[tag]['sDratio']
            if 'tc' not in self.model[tag]:  # cover thickness (optional, default = 0.0)
                tc = 0.0
            else:
                tc = self.model[tag]['tc']
            if 'cd' not in self.model[tag]:  # Viscous damping parameter (optional, default = 0.0)
                cd = 0.0
            else:
                cd = self.model[tag]['cd']
            # For strength and stiffness degradation effects
            if 'tag1' not in self.model[tag]:  # Tag to include cavitation and post-cavitation (optional, default = 0)
                tag1 = 0.0
            else:
                tag1 = self.model[tag]['tag1']
            if 'tag2' not in self.model[tag]:  # Tag to include buckling load variation (optional, default = 0)
                tag2 = 0.0
            else:
                tag2 = self.model[tag]['tag2']
            if 'tag3' not in self.model[tag]:  # Tag for buckling load variation (optional, default = 0)
                tag3 = 0.0
            else:
                tag3 = self.model[tag]['tag3']
            if 'tag4' not in self.model[tag]:  # Tag horizontal stiffness variation (optional, default = 0)
                tag4 = 0.0
            else:
                tag4 = self.model[tag]['tag4']
            if 'tag5' not in self.model[tag]:  # Tag for vertical stiffness variation (optional, default = 0)
                tag5 = 0.0
            else:
                tag5 = self.model[tag]['tag5']

            # For heating effects in case of lead rubber bearings
            if 'qL' not in self.model[tag]:  # density of lead (optional, default = 11200 kg/m3)
                qL = 11200 * kg / m ** 3
            else:
                qL = self.model[tag]['qL']
            if 'cL' not in self.model[tag]:  # specific heat of lead (optional, default = 130 N-m/kg oC)
                cL = 130 * N * m / kg
            else:
                cL = self.model[tag]['cL']
            if 'kS' not in self.model[tag]:  # thermal conductivity of steel (optional, default = 50 W/m oC)
                kS = 50 * N * m / sec
            else:
                kS = self.model[tag]['kS']
            if 'aS' not in self.model[tag]:  # thermal diffusivity of steel (optional, default = 1.41e-05 m2/s)
                aS = 1.41e-05 * m ** 2 / sec
            else:
                aS = self.model[tag]['aS']

            mass = 0.0  # element mass (optional, default = 0.0)

            matArgs = [Fy, alpha, G, K, D1, D2, ts, tr, nr]
            optionalArgs = [kc, PhiM, ac, sDratio, mass, cd, tc, qL, cL, kS, aS, tag1, tag2, tag3, tag4, tag5]

        elif self.model[tag]['model'] == 'HDR':
            Gr = self.model[tag]['Gr']  # Effective shear modulus
            Kbulk = self.model[tag]['Kbulk']  # Bulk modulus of rubber
            ts = self.model[tag]['ts']  # Thickness of steel shim plates
            tr = self.model[tag]['tr']  # Thickness of a single rubber layer
            nr = self.model[tag]['nr']  # Number of rubber layers
            D1 = self.model[tag]['D1']  # Internal diameter of lead rubber bearing
            D2 = self.model[tag]['D2']  # Outer diameter of lead rubber bearing
            # parameters of the Grant model
            a1 = self.model[tag]['a1']
            a2 = self.model[tag]['a2']
            a3 = self.model[tag]['a3']
            b1 = self.model[tag]['b1']
            b2 = self.model[tag]['b2']
            b3 = self.model[tag]['b3']
            c1 = self.model[tag]['c1']
            c2 = self.model[tag]['c2']
            c3 = self.model[tag]['c3']
            c4 = self.model[tag]['c4']
            if 'kc' not in self.model[tag]:  # cavitation parameter (optional, default = 10.0)
                kc = 10
            else:
                kc = self.model[tag]['kc']
            if 'PhiM' not in self.model[tag]:  # damage parameter (optional, default = 0.5)
                PhiM = 0.5
            else:
                PhiM = self.model[tag]['PhiM']
            if 'ac' not in self.model[tag]:  # strength reduction parameter (optional, default = 1.0)
                ac = 1.0
            else:
                ac = self.model[tag]['ac']
            # shear distance from iNode as fraction of the element length (optional, default = 0.5)
            if 'sDratio' not in self.model[tag]:
                sDratio = 0.5
            else:
                sDratio = self.model[tag]['sDratio']
            if 'tc' not in self.model[tag]:  # cover thickness (optional, default = 0.0)
                tc = 0.0
            else:
                tc = self.model[tag]['tc']
            mass = 0.0  # element mass (optional, default = 0.0)
            matArgs = [Gr, Kbulk, D1, D2, ts, tr, nr, a1, a2, a3, b1, b2, b3, c1, c2, c3, c4]
            optionalArgs = [kc, PhiM, ac, sDratio, mass, tc]

        return matArgs, optionalArgs

    def _bearings(self):
        """
        -----------------
        BEARING MODELLING
        -----------------
        """
        materials = self._create_bearing_mats()
        frictions = self._create_bearing_friction_models()

        # Create bearing elements
        self.EleIDsBearing = {i: [] for i in range(self.num_spans + 1)}
        vx = np.array([0, 0, 1])
        count = 0

        for tag in self.bearing_nodes:
            eleType = self.model[tag]['model']
            Wbearing = self.model[tag]['weight']

            for nodeI, nodeJ, angle, joint, e_uns, side in self.bearing_nodes[tag]:
                vy = np.round_(np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2), 0]), 7)
                length = np.round_(get_distance(ops.nodeCoord(nodeI), ops.nodeCoord(nodeJ)), 7)

                # save the element tag and other important information
                count += 1
                eleTag = int(str(count) + self.BearingTag)
                self.EleIDsBearing[joint].append([eleTag, nodeI, nodeJ, vx, vy,
                                                  e_uns, side, self.model[tag]['model']])
                # assign mass and save point loads
                if Wbearing > 0:
                    ops.mass(nodeI, Wbearing / g, Wbearing / g, Wbearing / g, 0, 0, 0)
                    self.PointLoads.append([nodeI, 0, 0, -Wbearing, 0, 0, 0])

                matArgs, optionalArgs = self._get_bearing_mat_args(tag, materials, frictions)

                if eleType not in ['ElastomericX', 'LeadRubberX', 'HDR']:
                    if length == 0:
                        eleArgs = [*matArgs, '-orient', *vx.tolist(), *vy.tolist()]  # zero-length element
                        if self.model[tag]['model'] == 'twoNodeLink':
                            eleType = 'zeroLength'
                    else:
                        eleArgs = [*matArgs, '-orient', *vy.tolist()]
                        if self.model[tag]['model'] == 'twoNodeLink':
                            eleType = 'twoNodeLink'
                else:
                    eleArgs = [*matArgs, *vx, *vy, *optionalArgs]

                # Generate element
                ops.element(eleType, eleTag, nodeI, nodeJ, *eleArgs)

    def _bentcaps(self):
        """
        --------------------------
        BENTCAP MODELLING
        --------------------------
        """
        # Create the sections
        tags = list(set(self.model['General']['Bent Caps']))
        IntTags = {}
        wBentCaps = {}
        for i in range(len(tags)):
            tag = tags[i]
            A = self.model[tag]['A']  # Deck area
            E = self.model[tag]['E']  # Young's modulus
            G = self.model[tag]['G']  # Shear modulus
            J = self.model[tag]['J']  # Polar moment of inertia
            Iz = self.model[tag]['Iz']  # Moment of inertia in z-dir
            Iy = self.model[tag]['Iy']  # Moment of inertia in y-dir
            self.EndSecTag += 1
            self.EndIntTag += 1
            ops.section('Elastic', self.EndSecTag, E, A, Iz, Iy, G, J)
            ops.beamIntegration('Legendre', self.EndIntTag, self.EndSecTag, 2)
            IntTags[tag] = self.EndIntTag
            wBentCaps[tag] = gamma_rc * A  # Weight of deck per length

        # Assuming the bent caps are never inclined.
        self.EndTransfTag += 1  # local z is same as global z
        ops.geomTransf('Linear', self.EndTransfTag, 0, 0, 1)

        # Necessary info to save
        self.EleIDsCap = {i: [] for i in range(1, self.num_spans)}  # Element Tags

        count = 0
        for i in range(self.num_spans - 1):
            tag = self.model['General']['Bent Caps'][i]

            if len(self.CapNodes[i + 1]) == 1:
                Pload = wBentCaps[tag] * self.model[tag]['length']
                nodeTag = self.CapNodes[i + 1][0]
                self.PointLoads.append([nodeTag, 0, 0, -Pload, 0, 0, 0])
                ops.mass(nodeTag, Pload / g, Pload / g, Pload / g, 0, 0, 0)

            else:
                # sort nodes by their distance to the start node
                dist_list = []
                Coords0 = self.CapLeftCoord[i + 1]
                for j in range(len(self.CapNodes[i + 1])):
                    Coords1 = np.round_(ops.nodeCoord(self.CapNodes[i + 1][j]), 5)
                    dist_list.append(get_distance(Coords0, Coords1))
                self.CapNodes[i + 1] = [x for _, x in sorted(zip(dist_list, self.CapNodes[i + 1]))]

                # create elements
                for j in range(len(self.CapNodes[i + 1]) - 1):
                    count += 1
                    nodeI = self.CapNodes[i + 1][j]
                    nodeJ = self.CapNodes[i + 1][j + 1]
                    eleTag = int(str(count) + self.BentCapTag)
                    wTOT = wBentCaps[tag]
                    intTag = IntTags[tag]
                    ops.element('dispBeamColumn', eleTag, nodeI, nodeJ, self.EndTransfTag, intTag, '-mass', wTOT / g,
                                self.mass_type)
                    # Save gravity loads and element IDs at each joint
                    self.DistributedLoads.append(['-ele', eleTag, '-type', '-beamUniform', 0, -wTOT, 0])
                    self.EleIDsCap[i + 1].append(eleTag)

    def _bents(self):
        """
        --------------------------
        BENT MODELLING
        --------------------------
        """

        # Create sections
        eleType = self.model['General']['Pier Element Type']
        sec_tags = {}
        int_tags = {}
        transf_tags = {}

        count = 0
        if not hasattr(self, 'EleIDsPier'):
            self.EleIDsPier = {}

        # Default section properties
        self._get_bent_section_properties()

        # define pier properties
        for bent_no in self.BentNodes:
            bent_name = self.model['General']['Bents'][bent_no - 1]

            for pier_idx, pier_nodes in enumerate(self.BentNodes[bent_no]):
                # section name and section weight
                sec_name = self.model[bent_name]['sections'][pier_idx]
                sec_weight = self.model[sec_name]['weight']
                # local axis rotation angle
                rot_angle = pier_nodes[-1]
                # pier tag
                pier_tag = f'B{bent_no}-P{pier_idx + 1}'
                # save pier properties
                if not pier_tag in self.EleIDsPier:
                    self.EleIDsPier[pier_tag] = {'section': sec_name}
                self.EleIDsPier[pier_tag]['nodes'] = pier_nodes[:-1]
                self.EleIDsPier[pier_tag]['H'] = self.model[bent_name]['height']
                num_ele = len(self.EleIDsPier[pier_tag]['nodes']) - 1

                # save yield curvature, yield displacement, plastic hinge length for this pier element
                self._get_plastic_hinge_properties(pier_tag)

                # Now we have to create different section for each pier element since their axial load is different.
                if eleType == 1:
                    sec_tag = self._add_pier_sec(pier_tag)

                else:
                    if sec_name in sec_tags:
                        # Do not create the sections again and again, use the sections which were created earlier
                        sec_tag = sec_tags[sec_name]['tag']
                    else:
                        sec_tag = self._add_pier_sec(pier_tag)
                        sec_tags[sec_name] = {'tag': sec_tag}

                # Define integration scheme
                IntTag = 0
                # Gauss-Lobatto Integration with 5 integration points
                if eleType < 3:
                    for key, val in int_tags.items():
                        if sec_tag == val:
                            IntTag = key
                    if IntTag == 0:
                        IntTag = self.EndIntTag + 1
                        self.EndIntTag += 1
                        int_tags[IntTag] = sec_tag
                        if eleType == 2:
                            ops.beamIntegration('Lobatto', IntTag, sec_tag, 5)
                        else:
                            ops.beamIntegration('Legendre', IntTag, sec_tag, 2)

                # Modified two-point Gauss-Radau integration
                elif eleType == 3:
                    sec_tag_el = self._add_pier_sec(pier_tag, el_sec_flag=1)
                    h_ele = self.EleIDsPier[pier_tag]['H']
                    Lp = self.EleIDsPier[pier_tag]['Plastic Hinge Length']
                    if 2 * Lp >= h_ele / num_ele:
                        raise ValueError('The pier element discretization length is very small. Hele < 2*Lp!')

                    # HingeRadau / SecI - inelastic, SecJ - inelastic, SecInt - elastic / Single element case
                    if num_ele == 1:
                        IntTag1 = self.EndIntTag + 1
                        self.EndIntTag += 1
                        ops.beamIntegration('HingeRadau', IntTag1, sec_tag, Lp, sec_tag, Lp, sec_tag_el)
                    # HingeRadau / SecI - elastic, SecJ - inelastic, SecInt - elastic / First element
                    if num_ele > 1:
                        IntTag2 = self.EndIntTag + 1
                        self.EndIntTag += 1
                        ops.beamIntegration('HingeRadau', IntTag2, sec_tag, Lp, sec_tag_el, Lp, sec_tag_el)
                    # HingeRadau / SecI - inelastic, SecJ - elastic, SecInt - elastic / Last element
                    if num_ele > 1:
                        IntTag3 = self.EndIntTag + 1
                        self.EndIntTag += 1
                        ops.beamIntegration('HingeRadau', IntTag3, sec_tag_el, Lp, sec_tag, Lp, sec_tag_el)
                    # Elastic section / Interior elements
                    if num_ele > 2:
                        IntTag4 = self.EndIntTag + 1
                        self.EndIntTag += 1
                        ops.beamIntegration('Legendre', IntTag4, sec_tag_el, 2)

                # Define geometric transformation of the pier, for 0 rotation, local z is in global x.
                vecxz = np.round_(np.array([np.cos(rot_angle), np.sin(rot_angle), 0]), 6)
                self.EleIDsPier[pier_tag]['vecxz'] = vecxz
                props = vecxz.tolist()
                TransfTag = 0
                for key, val in transf_tags.items():
                    if props == val:
                        TransfTag = key
                if TransfTag == 0:
                    TransfTag = self.EndTransfTag + 1
                    self.EndTransfTag += 1
                    transf_tags[TransfTag] = props
                    ops.geomTransf('PDelta', self.EndTransfTag, *vecxz.tolist())

                self.EleIDsPier[pier_tag]['elements'] = []
                for i in range(num_ele):
                    # Element tag
                    count += 1
                    eleTag = int(str(count) + self.PierTag)
                    self.EleIDsPier[pier_tag]['elements'].append(eleTag)
                    # Element nodes
                    nodeI = self.EleIDsPier[pier_tag]['nodes'][i]
                    nodeJ = self.EleIDsPier[pier_tag]['nodes'][i + 1]
                    # Save gravity loads and element IDs at each joint
                    self.DistributedLoads.append(['-ele', eleTag, '-type', '-beamUniform', 0, 0, -sec_weight])
                    if eleType == 3 and num_ele == 1:  # Single element case
                        IntTag = IntTag1 + 0
                    elif eleType == 3 and num_ele > 1 and i == 0:  # First element
                        IntTag = IntTag2 + 0
                    elif eleType == 3 and num_ele > 1 and i == num_ele - 1:  # Last element
                        IntTag = IntTag3 + 0
                    elif eleType == 3 and num_ele > 2:  # Interior elements
                        IntTag = IntTag4 + 0
                    # Create the element
                    ops.element('forceBeamColumn', eleTag, nodeI, nodeJ, TransfTag, IntTag, '-mass', sec_weight / g, self.mass_type)

    def _get_plastic_hinge_properties(self, pier_tag):
        # TODO: we need such calculations for other section types as well
        # TODO: since in transverse direction bent caps can act like a fixed boundary condition,
        #  we may need such calculations for fixed-fixed boundary conditions as well.
        # TODO: alternatively, we can estimate these directly from analysis routine in OpenSees.
        Dy = None
        Ky = None
        sec_tag = self.EleIDsPier[pier_tag]['section']
        # Calculate some section properties for EDP calculates
        if self.model[sec_tag]['Type'] == 'Solid Circular':
            C1 = 1 / 3  # coefficient for yield displacement calculations - cantilever case
            C2 = 0.022  # coefficient for strain penetration length - cantilever case

            # yield curvature (Eq. 10.1)
            Ky = 2.25 * self.model[sec_tag]['Fyle'] / self.model[sec_tag]['Es'] / self.model[sec_tag]['D']

            # Strain penetration lengths (Eq. 4.30) - cantilever case
            Lsp = C2 * self.model[sec_tag]['Fyle'] / MPa * self.model[sec_tag]['dl']

            # Yield displacements (Eq. 10.6) - cantilever case
            Dy = C1 * Ky * (self.EleIDsPier[pier_tag]['H'] + Lsp) ** 2

        # Plastic hinge length (Eq. 10.11) - cantilever case
        Lc = self.EleIDsPier[pier_tag]['H']
        # k = min(0.08,0.2*(Fu/Fy-1))
        k = 0.08
        Lp = (k * Lc / mm + 0.022 * self.model[sec_tag]['dl'] * self.model[sec_tag]['Fyle']) * mm

        if Ky:
            self.EleIDsPier[pier_tag]['Yield Curvature -Y'] = Ky
            self.EleIDsPier[pier_tag]['Yield Curvature -Z'] = Ky
        if Dy:
            self.EleIDsPier[pier_tag]['Yield Displacement -Y'] = Dy
            self.EleIDsPier[pier_tag]['Yield Displacement -Z'] = Dy
        self.EleIDsPier[pier_tag]['Plastic Hinge Length'] = Lp

    def _get_bent_section_properties(self):
        bent_tags = list(set(self.model['General']['Bents']))
        sec_tags = []
        for bent_tag in bent_tags:
            sec_tags.extend(self.model[bent_tag]['sections'])
        sec_tags = list(set(sec_tags))

        for sec_tag in sec_tags:
            """ Minimum required parameter for any section is the section type ('Type') """

            # Default material parameters for any section type
            if 'R0' not in self.model[sec_tag]:
                self.model[sec_tag]['R0'] = 18  # control the transition from elastic to plastic branches
            if 'cR1' not in self.model[sec_tag]:
                self.model[sec_tag]['cR1'] = 0.925  # control the transition from elastic to plastic branches
            if 'cR2' not in self.model[sec_tag]:
                self.model[sec_tag]['cR2'] = 0.15  # control the transition from elastic to plastic branches
            if 'eps1U' not in self.model[sec_tag]:
                self.model[sec_tag]['eps1U'] = 0.002  # Unconfined concrete strain at maximum compressive strength
            if 'eps2U' not in self.model[sec_tag]:
                self.model[sec_tag]['eps2U'] = 0.006  # Unconfined concrete strain at crushing strength
            if 'epssm' not in self.model[sec_tag]:
                self.model[sec_tag]['epssm'] = 0.10  # max transverse steel strain (usually ~0.10-0.15)*
            if 'Es' not in self.model[sec_tag]:  # Steel young's modulus
                self.model[sec_tag]['Es'] = 200 * GPa
            if 'Ec' not in self.model[sec_tag]:  # Concrete young's modulus
                self.model[sec_tag]['Ec'] = 5000 * MPa * (self.model[sec_tag]['Fce'] / MPa) ** 0.5
            if 'rs' not in self.model[sec_tag]:  # Steel hardening ratio
                self.model[sec_tag]['rs'] = 0.001
            if 'nu_c' not in self.model[sec_tag]:  # Poisson's ratio of concrete
                self.model[sec_tag]['nu_c'] = 0.2
            if 'nu_s' not in self.model[sec_tag]:  # Poisson's ratio of steel
                self.model[sec_tag]['nu_s'] = 0.3
            if 'min_eps_steel' not in self.model[sec_tag]:  # minimum steel strain in the fibers (steel buckling)
                self.model[sec_tag]['min_eps_steel'] = -0.1
            if 'max_eps_steel' not in self.model[sec_tag]:  # maximum steel strain in the fibers (steel rupture)
                self.model[sec_tag]['max_eps_steel'] = 0.1
            if 'Gc' not in self.model[sec_tag]:  # Concrete shear modulus
                self.model[sec_tag]['Gc'] = self.model[sec_tag]['Ec'] / (2 * (1 + self.model[sec_tag]['nu_c']))

            # Get the properties for solid circular section
            if self.model[sec_tag]['Type'] == 'Solid Circular':
                """
                Minimum required parameters for solid circular section
                'D': Section diameter
                'Fce': Unconfined concrete compressive strength
                'number of bars': Number of longitudinal bars
                'dl': Nominal diameter of longitudinal rebars
                'Fyle': Steel yield strength of longitudinal rebars
                """

                # The default properties for solid circular section
                if 'cover' not in self.model[sec_tag]:  # clear cover
                    self.model[sec_tag]['cover'] = 5 * cm
                if 'Transverse Reinforcement Type' not in self.model[sec_tag]:
                    # Type of transversal steel, 'Hoops' or 'Spirals'
                    self.model[sec_tag]['Transverse Reinforcement Type'] = 'Hoops'
                if 'Confinement' not in self.model[sec_tag]:
                    self.model[sec_tag]['Confinement'] = None  # Concrete/Confinement model

                # Determine the basic mechanical properties
                cc = self.model[sec_tag]['cover']
                D = self.model[sec_tag]['D']
                fc1U = self.model[sec_tag]['Fce']
                numBars = self.model[sec_tag]['number of bars']
                dl = self.model[sec_tag]['dl']
                Es = self.model[sec_tag]['Es']
                Ec = self.model[sec_tag]['Ec']
                Gc = self.model[sec_tag]['Gc']
                nu_s = self.model[sec_tag]['nu_s']
                barArea = np.pi * (dl ** 2) / 4
                Asl = numBars * barArea
                Ag = np.pi * D ** 2 / 4
                Ac = Ag - Asl  # concrete area
                Gs = Es / (2 * (1 + nu_s))
                k = 0.9  # Timoshenko beam shear coefficient for circular section

                # Save the calculated parameters
                self.model[sec_tag]['Kv'] = k * (Gs * Asl + Gc * Ac)  # shear stiffness for the section
                self.model[sec_tag]['At'] = Es / Ec * Asl + Ec * Ac  # Transformed section area
                self.model[sec_tag]['Iy'] = np.pi * D ** 4 / 64  # Moment of inertia around y-y
                self.model[sec_tag]['Iz'] = np.pi * D ** 4 / 64  # Moment of inertia around z-z
                self.model[sec_tag]['J'] = np.pi * D ** 4 / 32  # Polar moment of inertia
                self.model[sec_tag]['weight'] = Ag * gamma_rc  # Weight per length (kN/m)

                # Determine material properties for fiber section with single layer of circular reinforcement
                if self.model[sec_tag]['Confinement'] == 'Mander_et_al_1988':
                    """
                    Minimum required parameters for fiber section with 'Mander_et_al_1988' confinement model
                    Fyhe: Expected yield strength of transverse rebars
                    dh: Nominal diameter of transversal bars
                    sh: Vertical spacing between the centroid of spirals or hoops
                    """
                    Fyhe = self.model[sec_tag]['Fyhe']
                    dh = self.model[sec_tag]['dh']
                    sh = self.model[sec_tag]['sh']
                    TransReinfType = self.model[sec_tag]['Transverse Reinforcement Type']
                    eps1U = self.model[sec_tag]['eps1U']
                    eps2U = self.model[sec_tag]['eps2U']
                    epssm = self.model[sec_tag]['epssm']
                    ds = D - 2 * cc  # Core diameter
                    sp = sh - dh  # Clear vertical spacing between spiral or hoop bars
                    Acc = np.pi * ds ** 2 / 4  # Area of core of section enclosed by the center lines of the perimeter spiral or hoop
                    pcc = Asl / Acc  # Ratio of area of longitudinal reinforcement to area of core of section
                    if TransReinfType == 'Hoops':
                        ke = (1 - sp / 2 / ds) ** 2 / (1 - pcc)  # Confinement effectiveness coefficient for hoops
                    elif TransReinfType == 'Spirals':
                        ke = (1 - sp / 2 / ds) / (1 - pcc)  # Confinement effectiveness coefficient for spirals
                    Ash = np.pi * dh ** 2 / 4  # Area of transverse steel
                    ps = 4 * Ash / (ds * sh)  # Ratio of the volume of transverse confining steel to the volume of confined concrete core
                    fpl = ke * 0.5 * Fyhe * ps  # Confinement pressure
                    Kfc = (-1.254 + 2.254 * (1 + 7.94 * fpl / fc1U) ** 0.5 - 2 * fpl / fc1U)  # Confinement factor

                    # Save the calculated parameters
                    if 'fc1C_c04' not in self.model[sec_tag]:  # Confined concrete compressive strength
                        self.model[sec_tag]['fc1C_c04'] = Kfc * fc1U
                    if 'eps1C_c04' not in self.model[sec_tag]:  # Confined concrete strain at maximum stress
                        self.model[sec_tag]['eps1C_c04'] = eps1U * (1 + 5 * (self.model[sec_tag]['fc1C_c04'] / fc1U - 1))
                    if 'eps2C_c04' not in self.model[sec_tag]:  # Concrete strain at ultimate stress
                        self.model[sec_tag]['eps2C_c04'] = max(0.004 + 1.4 * ps * Fyhe * epssm / self.model[sec_tag]['fc1C_c04'], eps2U)
                        # 1.5 is recommended by Kowalsky, this equation is very conservative, but whatever.
                        # self.model[sec_tag]['eps2C_c04'] = max(1.5 * (0.004 + 1.4 * ps * Fyhe * epssm / self.model[sec_tag]['fc1C_c04']), eps2U)
                    if 'fct_c04' not in self.model[sec_tag]:  # Tensile-strength
                        self.model[sec_tag]['fct_c04'] = 0.56 * MPa * (fc1U / MPa) ** 0.5
                    if 'et_c04' not in self.model[sec_tag]:  # Floating point value defining the maximum tensile strength of concrete
                        self.model[sec_tag]['et_c04'] = self.model[sec_tag]['fct_c04'] / Ec
                    if 'beta_c04' not in self.model[sec_tag]:  # Floating point value defining the exponential curve parameter to define the residual stress (as a factor of ft) at etu
                        self.model[sec_tag]['beta_c04'] = 0.1

                elif self.model[sec_tag]['Confinement'] == 'Kent_Park_1971':
                    """
                    Minimum required parameters for fiber section with 'Mander_et_al_1988' confinement model
                    Fyhe: Expected yield strength of transverse rebars
                    dh: Nominal diameter of transversal bars
                    sh: Vertical spacing between the centroid of spirals or hoops
                    """
                    Fyhe = self.model[sec_tag]['Fyhe']
                    dh = self.model[sec_tag]['dh']
                    sh = self.model[sec_tag]['sh']
                    eps1U = self.model[sec_tag]['eps1U']
                    eps2U = self.model[sec_tag]['eps2U']
                    ds = D - 2 * cc  # Core diameter
                    Ash = np.pi * dh ** 2 / 4  # Area of transverse steel
                    ps = 4 * Ash / (ds * sh)  # Ratio of the volume of transverse confining steel to the volume of confined concrete core
                    eps_50u = (3 + eps1U * fc1U / psi) / (fc1U / psi - 1000)  # Strain softening slope
                    eps_50h = 0.75 * ps * (ds / sh) ** 0.5

                    # Save the calculated parameters
                    if 'fc2U_c02' not in self.model[sec_tag]:  # Unconfined concrete residual strength
                        self.model[sec_tag]['fc2U_c02'] = 0
                    if 'fc1C_c02' not in self.model[sec_tag]:  # Confined concrete compressive strength
                        Kfc = 1 + ps * Fyhe / fc1U  # Confinement factor
                        self.model[sec_tag]['fc1C_c02'] = Kfc * fc1U
                    else:
                        Kfc = self.model[sec_tag]['fc1C_c02'] / fc1U
                    if 'fc2C_c02' not in self.model[sec_tag]:  # Confined concrete crushing strength
                        self.model[sec_tag]['fc2C_c02'] = 0.2 * self.model[sec_tag]['fc1C_c02']
                    if 'eps1C_c02' not in self.model[sec_tag]:  # Confined concrete strain at maximum stress
                        self.model[sec_tag]['eps1C_c02'] = Kfc * eps1U
                    if 'eps2C_c02' not in self.model[sec_tag]:  # Confined concrete strain at crushing strength
                        Z = 0.5 / (eps_50u + eps_50h - self.model[sec_tag]['eps1C_c02'])
                        self.model[sec_tag]['eps2C_c02'] = max(eps2U, 0.8 / Z + self.model[sec_tag]['eps1C_c02'])
                    if 'Lambda_c02' not in self.model[sec_tag]:  # Ratio between unloading slope at $epscu and initial slope
                        self.model[sec_tag]['Lambda_c02'] = 0.1
                    if 'fct_c02' not in self.model[sec_tag]:  # Tensile-strength
                        self.model[sec_tag]['fct_c02'] = 0.56 * MPa * (fc1U / MPa) ** 0.5
                    if 'Ets_c02' not in self.model[sec_tag]:  # Tension softening stiffness (absolute value) (slope of the linear tension softening branch)
                        self.model[sec_tag]['Ets_c02'] = self.model[sec_tag]['fct_c02'] / 0.002

    def _add_pier_sec(self, pier_tag, mphi=0, el_sec_flag=0):

        # Section name defined by the user for the pier section
        sec_name = self.EleIDsPier[pier_tag]['section']

        if self.model[sec_name]['Type'] == 'Solid Circular':
            #  ---------------------------------
            #  SOLID CIRCULAR SECTION PROPERTIES
            #  ---------------------------------
            cc = self.model[sec_name]['cover']
            D = self.model[sec_name]['D']
            Es = self.model[sec_name]['Es']
            Ec = self.model[sec_name]['Ec']
            Gc = self.model[sec_name]['Gc']
            J = self.model[sec_name]['J']
            Kv = self.model[sec_name]['Kv']
            At = self.model[sec_name]['At']
            Iy = self.model[sec_name]['Iz']
            Iz = self.model[sec_name]['Iy']
            numBars = int(self.model[sec_name]['number of bars'])
            Abar = np.pi * (self.model[sec_name]['dl'] ** 2) / 4

            # Material IDs
            steelID = self.EndMatTag + 1
            self.EndMatTag += 1
            coverID = self.EndMatTag + 1
            self.EndMatTag += 1
            coreID = self.EndMatTag + 1
            self.EndMatTag += 1
            torsionID = self.EndMatTag + 1
            self.EndMatTag += 1
            ShearID = self.EndMatTag + 1
            self.EndMatTag += 1
            MinMaxID = self.EndMatTag + 1
            self.EndMatTag += 1

            # fiber configuration, consider shear deformations
            yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT = self._circ_fiber_config(D)

            if mphi == 0 and self.model['General']['Pier Element Type'] == 0:
                #  ---------------------
                #  ELASTIC FIBER SECTION
                #  ---------------------
                # Elastic Material to represent torsional behaviour of the section
                ops.uniaxialMaterial('Elastic', torsionID, Gc * J)
                # Shear Material
                ops.uniaxialMaterial('Elastic', ShearID, Kv)
                # Concrete Material
                ops.uniaxialMaterial('Elastic', coreID, Ec)
                # Reinforcing Steel Material
                ops.uniaxialMaterial('Elastic', steelID, Es)
                # Define the fiber section, consider shear deformations
                self.EndSecTag += 1
                ops.section('Fiber', self.EndSecTag, '-torsion', torsionID)
                # Define the core patch
                rc = ro - cc  # Core radius
                ops.patch('circ', coreID, nfCoreT, nfCoreR + nfCoverR, yC, zC, ri, rc, startAng, endAng)
                # Define the reinforcing layer
                theta = endAng / numBars  # Determine angle increment between bars
                ops.layer('circ', steelID, numBars, Abar, yC, zC, rc, theta, endAng)
                # Aggregate shear
                self.EndSecTag += 1
                ops.section('Aggregator', self.EndSecTag, ShearID, 'Vy', ShearID, 'Vz', '-section', self.EndSecTag - 1)

            elif mphi == 0 and (self.model['General']['Pier Element Type'] == 1 or el_sec_flag == 1):
                #  -----------------------------------------------------------------------------
                #  ELASTIC CRACKED SECTION - USES INITIAL STIFFNESS OF BI-LINEARIZED M-PHI CURVE
                #  -----------------------------------------------------------------------------
                # Reduction factors for moment of inertia
                RF = self.EleIDsPier[pier_tag]['RF']
                # Shear Material
                ops.uniaxialMaterial('Elastic', ShearID, Kv)
                # Elastic cracked section
                self.EndSecTag += 1
                ops.section('Elastic', self.EndSecTag, Ec, At, RF * Iz, RF * Iy, Gc, J)
                # Aggregate shear material
                self.EndSecTag += 1
                ops.section('Aggregator', self.EndSecTag, ShearID, 'Vy', ShearID, 'Vz', '-section', self.EndSecTag - 1)

            elif mphi == 1 or self.model['General']['Pier Element Type'] in [2, 3]:
                #  -----------------------
                #  INELASTIC FIBER SECTION
                #  -----------------------
                fc1U = self.model[sec_name]['Fce']
                eps1U = self.model[sec_name]['eps1U']
                eps2U = self.model[sec_name]['eps2U']
                Fy = self.model[sec_name]['Fyle']
                rs = self.model[sec_name]['rs']
                minStrain = self.model[sec_name]['min_eps_steel']
                maxStrain = self.model[sec_name]['max_eps_steel']

                # Elastic Material to represent torsional behaviour of the section
                ops.uniaxialMaterial('Elastic', torsionID, Gc * J)  # define elastic torsional stiffness
                # Shear Material
                ops.uniaxialMaterial('Elastic', ShearID, Kv)  # define elastic shear stiffness

                if self.model[sec_name]['Confinement'] == 'Mander_et_al_1988':
                    #  ------------------------------------------------------------
                    #  CONFINEMENT MODEL: Mander et al. 1988
                    #  CONCRETE MODEL: Concrete04 with strength degradation
                    #  STEEL MODEL: Steel02 Giuffré-Menegotto-Pinto Model (1973) with Isotropic Strain Hardening
                    #  ------------------------------------------------------------
                    fct = self.model[sec_name]['fct_c04']
                    et = self.model[sec_name]['et_c04']
                    beta = self.model[sec_name]['beta_c04']
                    fc1C = self.model[sec_name]['fc1C_c04']
                    eps1C = self.model[sec_name]['eps1C_c04']
                    eps2C = self.model[sec_name]['eps2C_c04']
                    R0 = self.model[sec_name]['R0']
                    cR1 = self.model[sec_name]['cR1']
                    cR2 = self.model[sec_name]['cR2']

                    # Cover concrete material
                    ops.uniaxialMaterial('Concrete04', coverID, -fc1U, -eps1U, -eps2U, Ec, fct, et, beta)
                    # Core concrete material
                    ops.uniaxialMaterial('Concrete04', coreID, -fc1C, -eps1C, -eps2C, Ec, fct, et, beta)
                    # Reinforcing Steel Materials
                    ops.uniaxialMaterial('Steel02', steelID, Fy, Es, rs, R0, cR1, cR2)

                elif self.model[sec_name]['Confinement'] == 'Kent_Park_1971':
                    #  ------------------------------------------------------------------------------
                    #  CONFINEMENT MODEL: Kent and Park 1971
                    #  CONCRETE MODEL: Concrete01 material with strength degradation and linear tension softening
                    #  STEEL MODEL: Steel01 bi-linear material
                    #  ------------------------------------------------------------------------------
                    fct = self.model[sec_name]['fct_c02']
                    Ets = self.model[sec_name]['Ets_c02']
                    Lambda = self.model[sec_name]['Lambda_c02']
                    fc1C = self.model[sec_name]['fc1C_c02']
                    fc2C = self.model[sec_name]['fc2C_c02']
                    fc2U = self.model[sec_name]['fc2U_c02']
                    eps1C = self.model[sec_name]['eps1C_c02']
                    eps2C = self.model[sec_name]['eps2C_c02']

                    # Cover concrete material
                    ops.uniaxialMaterial('Concrete02', coverID, -fc1U, -eps1U, -fc2U, -eps2U, Lambda, fct, Ets)
                    # Core concrete material
                    ops.uniaxialMaterial('Concrete02', coreID, -fc1C, -eps1C, -fc2C, -eps2C, Lambda, fct, Ets)
                    # Reinforcing Steel Materials
                    ops.uniaxialMaterial('Steel01', steelID, Fy, Es, rs)

                else:
                    #  ---------------------------------------------------------------
                    #  CONFINEMENT MODEL: NONE
                    #  CONCRETE MODEL: Concrete01 without any strength degradation
                    #  STEEL MODEL: Steel01 bi-linear material
                    #  ---------------------------------------------------------------
                    coreID = int(coverID)
                    ops.uniaxialMaterial('Concrete01', coreID, -fc1U, -eps1U, -fc1U, -eps2U)
                    # Reinforcing Steel Materials
                    ops.uniaxialMaterial('Steel01', steelID, Fy, Es, rs)

                # Define rupture and buckling strains for steel reinforcements
                ops.uniaxialMaterial('MinMax', MinMaxID, steelID, '-min', minStrain, '-max', maxStrain)
                # Define the fiber section
                self.EndSecTag += 1
                ops.section('Fiber', self.EndSecTag, '-torsion', torsionID)
                rc = ro - cc  # Core radius
                # Define the core patch
                ops.patch('circ', coreID, nfCoreT, nfCoreR, yC, zC, ri, rc, startAng, endAng)
                # Define the cover patch
                ops.patch('circ', coverID, nfCoverT, nfCoverR, yC, zC, rc, ro, startAng, endAng)
                theta = endAng / numBars  # Determine angle increment between bars
                # Define the reinforcing layer
                ops.layer('circ', MinMaxID, numBars, Abar, yC, zC, rc, theta, endAng)
                # Aggregate shear to account for shear deformations
                self.EndSecTag += 1
                ops.section('Aggregator', self.EndSecTag, ShearID, 'Vy', ShearID, 'Vz', '-section', self.EndSecTag - 1)

        return self.EndSecTag

    def _abutments(self):
        """
        --------------------------------------------
        MODELLING OF ABUTMENTS
        --------------------------------------------
        """
        abut_conf = self.model['General']['Abutments']
        if self.model[abut_conf]['Type'] != 'Fixed':
            count_ele = 0
            skew1 = self.skew[0]
            skew2 = self.skew[-1]
            # TODO: Approximate dynamic mass can be included at deck level as h*w*b/3

            if self.model[abut_conf]['Type'] == 'SDC 2019':
                # Notes:
                # Embankment width = Abutment Width, Bc
                # Embankment depth = Abutment Height, H
                # Embankment has slope of 60 degrees, S = 0.866
                # Both abutment sides have the same properties
                # For short bridges usually this is the case
                # For long bridges who cares the abutment

                # Abutment is assumed to be a rectangular prism,
                # Approximate dynamic mass at top is h*w*b/3

                # INPUTS
                h = self.model[abut_conf]['h']
                w = self.model[abut_conf]['w']
                NumPile = self.model[abut_conf]['NumPile']

                # The backfill passive behaviour in longitudinal direction
                # CALTRANS - SDC 2019: Section 6.3.1.2
                if h < 2 * ft:
                    h = 2 * ft
                if h > 10 * ft:
                    h = 10 * ft
                h_ft = h / ft
                w_ft = w / ft  # convert to ft
                theta = 0
                Rsk = np.exp(theta / 45)  # theta < 66
                FL_passive = w_ft * (5.5 * h_ft ** 2.5) / (1 + 2.37 * h_ft) * Rsk * kip  # convert to kN
                KL_passive = w_ft * (5.5 * h_ft + 20) * Rsk * kip / inch  # convert to kN/m
                dyL_passive = FL_passive / KL_passive
                # In longitudinal direction
                KxMat1 = self.EndMatTag + 1
                self.EndMatTag += 1
                ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, KL_passive, -FL_passive, 0, 0, 'Damage')

                # Maroney and Chai (1994) can be followed to account for presence of Wing-wall
                # Transverse backfill pressure factor is CL*CW = 2/3*4/3 according to Maroney and Chai (1994).
                CL = 2 / 3  # Wing-wall effectiveness coefficient
                CW = 4 / 3  # Wing-wall participation coefficient
                ratio = 0.5  # The wing wall length can be assumed 1/2-1/3 of the back-wall length
                FT_wingwall = FL_passive * ratio * CL * CW
                KT_wingwall = KL_passive * ratio * CL * CW
                dyT_wingwall = FT_wingwall / KT_wingwall
                # In transverse direction
                KyMat1 = self.EndMatTag + 1
                self.EndMatTag += 1
                ops.uniaxialMaterial('ElasticPP', self.EndMatTag, KT_wingwall, dyT_wingwall)

                # TODO: foundation springs must be defined separately at foundation level
                #  Nonetheless let's ignore this for now, and define all the springs including backfill at deck level.
                # The active soil behaviour of piles in both directions
                Fpile = 119 * kN  # ultimate pile strength (Nielson, 2005; Avsar, 2009)
                Kpile = 40 * kip / inch  # CALTRANS - SDC 2019: Section 6.3.2
                PileMat = self.EndMatTag + 1
                KyMat = self.EndMatTag + 1
                self.EndMatTag += 1
                ops.uniaxialMaterial('Steel01', self.EndMatTag, NumPile * Fpile, NumPile * Kpile, 1e-8)
                KxMat = self.EndMatTag + 1
                self.EndMatTag += 1
                ops.uniaxialMaterial('Parallel', KxMat, PileMat, KxMat1)
                # Comment on the following three lines to remove wing-wall effect
                KyMat = self.EndMatTag + 1
                self.EndMatTag += 1
                ops.uniaxialMaterial('Parallel', KyMat, PileMat, KyMat1)

            elif self.model[abut_conf]['Type'] == 'Pinho et al. 2009':
                # According to:
                # Pinho, R., Monteiro, R., Casarotti, C., & Delgado, R. (2009).
                # Assessment of continuous span bridges through nonlinear static procedures.
                # Earthquake Spectra, 25(1), 143–159. doi:10.1193/1.3050449
                # Perdomo, C., Monteiro, R., & Sucuoğlu, H. (2020).
                # Development of Fragility Curves for Single-Column RC Italian Bridges Using Nonlinear Static Analysis.
                # In Journal of Earthquake Engineering (pp. 1–25). Informa UK Limited.
                # https://doi.org/10.1080/13632469.2020.1760153
                KxMat = self.EndMatTag + 1
                self.EndMatTag += 1
                KyMat = self.EndMatTag + 1
                self.EndMatTag += 1
                Fy = 4800 * kN
                b = 0.005
                Kx = 210 * kN / mm
                Ky = 1400 * kN / mm
                ops.uniaxialMaterial('Steel01', KxMat, Fy, Kx, b)
                ops.uniaxialMaterial('Steel01', KyMat, Fy, Ky, b)

            MatTags1 = [KxMat, KyMat, self.BigMat, self.BigMat, self.BigMat, self.BigMat]
            MatTags2 = [KxMat, KyMat, self.BigMat, self.BigMat, self.BigMat, self.BigMat]
            dirs = [1, 2, 3, 4, 5, 6]
            count_ele += 1
            AB1_ele = int(str(count_ele) + self.AbutTag)
            count_ele += 1
            AB2_ele = int(str(count_ele) + self.AbutTag)
            vz = np.array([0, 0, 1])
            vx_1 = np.round_(np.array([np.cos(skew1), np.sin(skew1), 0]), 7)
            vx_2 = np.round_(np.array([np.cos(skew2 + 2 * np.pi / 2), np.sin(skew2 + 2 * np.pi / 2), 0]), 7)
            vy_1 = np.cross(vz, vx_1)
            vy_2 = np.cross(vz, vx_2)

            ops.element('zeroLength', AB1_ele, *self.Abutment_NodesSpring[0], '-mat', *MatTags1, '-dir',
                        *dirs, '-orient', *vx_1.tolist(), *vy_1.tolist())
            ops.element('zeroLength', AB2_ele, *self.Abutment_NodesSpring[1], '-mat', *MatTags2, '-dir',
                        *dirs, '-orient', *vx_2.tolist(), *vy_2.tolist())

            self.EleIDsAbut = [AB1_ele, AB2_ele]

    def _shearkeys(self):
        """
        --------------------------------------------
        MODELLING OF SHEAR KEYS
        --------------------------------------------
        Shear keys are defined at the last and first bearings only.
        Hence, user must define lumped properties for shear keys.

        """
        conf = self.model['General']['Shear Keys']
        sk_type = self.model[conf]['Type']
        count = 0
        vx = np.array([0, 0, 1])
        joints = list(self.shear_key_nodes.keys())
        self.EleIDsShearKey = {i: [] for i in range(self.num_spans + 1)}
        if sk_type == 'None':
            pass

        elif sk_type == 'Non-Sacrificial':
            gapT = self.model[conf]['gapT']
            gapL = self.model[conf]['gapL']
            matTag_Transverse = self.EndMatTag + 1
            self.EndMatTag += 1
            ops.uniaxialMaterial('ElasticPPGap', matTag_Transverse, 1e7, -1e7, -gapT, -0.001, 'noDamage')
            matTag_CompLong = self.EndMatTag + 1
            self.EndMatTag += 1
            ops.uniaxialMaterial('ElasticPPGap', matTag_CompLong, 1e7, -1e7, -gapL, -0.001, 'noDamage')
            matTag_TensLong = self.EndMatTag + 1
            self.EndMatTag += 1
            ops.uniaxialMaterial('ElasticPPGap', matTag_TensLong, 1e7, 1e7, gapL, 0.001, 'noDamage')

            for joint in self.shear_key_nodes:
                for side in ['left', 'right']:
                    if self.shear_key_nodes[joint][side]:
                        # save the element tag and other important information
                        count += 1
                        eleTag_1 = int(str(count) + self.ShearKeyTag)  # first bearing location on the bent cap/abutment
                        count += 1
                        eleTag_2 = int(str(count) + self.ShearKeyTag)  # last bearing location on the bent cap/abutment

                        node_i1, node_j1, angle = self.shear_key_nodes[joint][side][0]
                        # Check if there is only one bearing
                        if len(self.shear_key_nodes[joint][side]) > 1:
                            node_i2, node_j2, _ = self.shear_key_nodes[joint][side][1]
                        else:
                            node_i2, node_j2 = node_i1, node_j1
                        vy_1 = np.round_(np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2), 0]), 7)
                        vy_2 = np.round_(np.array([np.cos(angle + 3 * np.pi / 2), np.sin(angle + 3 * np.pi / 2), 0]), 7)
                        length = np.round_(get_distance(ops.nodeCoord(node_i1), ops.nodeCoord(node_j1)), 7)

                        # TODO: comment/uncomment next line and unindent/indent rest to add/remove shear keys at piers
                        # if joint in [joints[0], joints[-1]]:
                        if joint == joints[0]:  # start abutment
                            matArgs_1 = ['-mat', matTag_Transverse, matTag_TensLong, '-dir', 2, 3]
                            matArgs_2 = ['-mat', matTag_Transverse, matTag_CompLong, '-dir', 2, 3]
                        elif joint == joints[-1]:  # end abutment
                            matArgs_1 = ['-mat', matTag_Transverse, matTag_CompLong, '-dir', 2, 3]
                            matArgs_2 = ['-mat', matTag_Transverse, matTag_TensLong, '-dir', 2, 3]
                        else:
                            matArgs_1 = ['-mat', matTag_Transverse, '-dir', 2]
                            matArgs_2 = ['-mat', matTag_Transverse, '-dir', 2]

                        if length == 0:
                            eleType = 'zeroLength'
                            eleArgs_1 = [*matArgs_1, '-orient', *vx.tolist(), *vy_1.tolist()]
                            eleArgs_2 = [*matArgs_2, '-orient', *vx.tolist(), *vy_2.tolist()]
                        else:
                            eleType = 'twoNodeLink'
                            eleArgs_1 = [*matArgs_1, '-orient', *vy_1.tolist()]
                            eleArgs_2 = [*matArgs_2, '-orient', *vy_2.tolist()]
                        # Generate elements
                        ops.element(eleType, eleTag_1, node_i1, node_j1, *eleArgs_1)
                        ops.element(eleType, eleTag_2, node_i2, node_j2, *eleArgs_2)
                        self.EleIDsShearKey[joint] = [eleTag_1, eleTag_2]

        elif sk_type == 'Sacrificial':
            # TODO: need to add some options here
            # Transverse Springs / CALTRANS - SDC 2019: Section 4.3.1, Figure 4.3.1-2
            # The assumption is that sacrificial shear key is used where gap is 2 inches
            dyT = 2 * inch
            Fy = 0.3 * (self.AB1AxialForces + self.AB2AxialForces) / 2
            K = Fy / dyT

    def _foundations(self):
        """
        --------------------------------------------
        MODELLING OF FOUNDATIONS
        --------------------------------------------
        """
        pass

    def _constraints(self):
        """
        --------------------------------------------
        ASSIGNING CONSTRAINTS
        --------------------------------------------
        """

        matTags = [self.BigMat, self.BigMat, self.BigMat, self.BigMat, self.BigMat, self.BigMat]
        dirs = [1, 2, 3, 4, 5, 6]
        RigidCount = 1

        # Use beam column elements with very high stiffness
        if self.const_opt == 1:
            for i in range(len(self.RigidLinkNodes)):
                eleTag = int(str(RigidCount) + self.RigidTag)
                RigidCount += 1
                eleNodes = self.RigidLinkNodes[i]
                coords1 = np.array(ops.nodeCoord(eleNodes[0]))
                coords2 = np.array(ops.nodeCoord(eleNodes[1]))
                if (coords1 == coords2).all():
                    ops.element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs)
                else:
                    ops.element('dispBeamColumn', eleTag, *eleNodes, self.RigidTransfTag, self.BigInt)

        # Use opensees multiple point constraint options
        else:
            for i in range(len(self.RigidLinkNodes)):
                eleNodes = self.RigidLinkNodes[i]
                coords1 = np.array(ops.nodeCoord(eleNodes[0]))
                coords2 = np.array(ops.nodeCoord(eleNodes[1]))
                if (coords1 == coords2).all():
                    ops.equalDOF(*self.RigidLinkNodes[i], 1, 2, 3, 4, 5, 6)
                else:
                    ops.rigidLink('beam', *self.RigidLinkNodes[i])

        for i in range(len(self.fixed)):
            ops.fix(*self.fixed[i])

    def _get_pier_mphi(self):

        for pier_tag in self.EleIDsPier:
            # TODO: need to implement 3D moment curvature analysis
            sec_name = self.EleIDsPier[pier_tag]['section']
            secType = self.model[sec_name]['Type']  # Section type
            Ec = self.model[sec_name]['Ec']  # Concrete modulus
            Es = self.model[sec_name]['Es']  # Steel modulus
            Fy = self.model[sec_name]['Fyle']  # Steel yield strength
            P = -self.EleIDsPier[pier_tag]['AxialForce']  # Axial force on element

            if 'Circular' in secType:
                dof_list = [5]
            else:
                dof_list = [5, 6]

            for ctrlDOF in dof_list:
                # Remove any existing model
                ops.wipe()
                ops.wipeAnalysis()
                ops.model('basic', '-ndm', 3, '-ndf', 6)
                sec_tag = self._add_pier_sec(pier_tag, mphi=1)

                # TODO: change this for other type of sections, depends on push direction
                # Yield curvature estimate
                if 'Circular' in secType:
                    Ky = self.EleIDsPier[pier_tag]['Yield Curvature -Y']
                # Target ductility for analysis
                mu = 8
                # Number of analysis increments
                numIncr = 500
                # Ultimate curvature defined for analysis
                Kmax = Ky * mu
                # Call the section analysis procedure
                M, Phi, eps = self._mca(sec_tag, P, Kmax, ctrlDOF, numIncr)

                # Obtaining initial stiffness from idealized bi-linear moment-curvature curves
                M = np.array(M)
                Phi = np.array(Phi)
                eps = np.array(eps)
                # Parameters to identify first yield point
                epsY_c = 0.002  # eps_c0
                epsY_s = Fy / Es  # Fy/Es
                # Parameters to identify nominal capacity
                epsN_c = 0.004  # serviceability limit state for concrete (Priestley et al. 2007)
                epsN_s = 0.015  # serviceability limit state for steel (Priestley et al. 2007)

                # TODO: change this for other type of sections
                if 'Circular' in secType:
                    D = self.model[sec_name]['D']
                    cc = self.model[sec_name]['cover']
                    EIgross = Ec * np.pi * D ** 4 / 64
                    d = D - cc
                    c = -(eps / Phi - D / 2)

                eps_c = Phi * c
                eps_s = Phi * (d - c)
                eps_c[0] = eps[0]
                eps_s[0] = eps[0]
                phi_eps_c = interp1d(eps_c, Phi)
                phi_eps_s = interp1d(eps_s, Phi)
                m_phi = interp1d(Phi, M)
                m_eps_c = interp1d(eps_c, M)
                m_eps_s = interp1d(eps_s, M)

                # First Yield Point
                Phi_y = min(phi_eps_c(epsY_c), phi_eps_s(epsY_s))
                M_y = m_phi(Phi_y)

                # Nominal Capacity Point
                try:
                    M_n = min(m_eps_c(epsN_c), m_eps_s(epsN_s))
                except ValueError:  # probably we never reach epsN_s
                    M_n = m_eps_c(epsN_c)
                # M_n = max(M)
                Phi_n = M_n / M_y * Phi_y

                # Maximum Capacity Point
                M_max = max(M)
                Phi_max = Phi[M == M_max]

                # Get final estimates to save
                MPhi_bilin = np.array([[0, 0], [float(M_n), float(Phi_n)], [float(M_max), float(Phi_max)]])
                M = np.append(0, M).reshape(-1, 1)
                Phi = np.append(0, Phi).reshape(-1, 1)
                MPhi = np.append(M, Phi, axis=1)
                EIeff = M_n / Phi_n
                RF = EIeff / EIgross  # reduction factors for cracked section

                # TODO: change this for other type of sections
                if 'Circular' in secType:
                    self.EleIDsPier[pier_tag]['RF'] = RF
                    self.EleIDsPier[pier_tag]['MPhi'] = MPhi
                    self.EleIDsPier[pier_tag]['MPhi_bilin'] = MPhi_bilin

    @staticmethod
    def _circ_fiber_config(D, ri=0.0):
        # The center of the reinforcing bars are placed at the inner radius
        # The core concrete ends at the inner radius (same as reinforcing bars)
        # The reinforcing bars are all the same size
        # The center of the section is at (0,0) in the local axis system
        # Zero degrees is along section y-axis
        # ri, diameter of hollow section

        yC = 0  # y-axis for center of circular section
        zC = 0  # z-axis for center of circular section
        startAng = 0  # start angle of circular section
        endAng = 360  # end angle of circular section
        ro = D / 2  # outer diameter of section
        nfCoreR = 10  # number of radial divisions in the core (number of "rings")
        nfCoreT = 20  # number of theta divisions in the core (number of "wedges")
        nfCoverR = 2  # number of radial divisions in the cover
        nfCoverT = 20  # number of theta divisions in the cover

        return yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT

    @staticmethod
    def getRectProp(h, b):
        # Get Mechanical Properties
        A = b * h  # area
        Izz = h * b ** 3 / 12  # moment of inertia z-z
        Iyy = b * h ** 3 / 12  # moment of inertia y-y
        J = Izz + Iyy  # moment of inertia x-x

        # Torsional stiffness of section
        a = max(b, h)
        b = min(b, h)
        ab_ratio = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0]
        beta_values = [0.141, 0.196, 0.229, 0.249, 0.263, 0.281, 0.291, 0.299, 0.312]
        if a / b > 10:
            beta = 0.333
        else:
            beta = np.interp(a / b, ab_ratio, beta_values)
        J = beta * a * b ** 3  # torsional constant

        return A, Izz, Iyy, J

    @staticmethod
    def getCircProp(D):
        # D: Pier Diameter
        pi = 3.141592653589793
        A = pi * D ** 2 / 4
        J = pi * D ** 4 / 32
        Iy = pi * D ** 4 / 64
        Iz = pi * D ** 4 / 64

        return A, Iy, Iz, J


def _discretize_member(ndI, ndJ, numEle, eleType, integrTag, transfTag, nodeTag, eleTag):
    nodeList = []
    eleList = []

    if eleType not in ['dispBeamColumn', 'forceBeamColumn', 'mixedBeamColumn']:
        print(f'Discretization does not work for {eleType}')
        return eleList, nodeList

    if numEle <= 1:
        ops.element(eleType, eleTag, ndI, ndJ, transfTag, integrTag)
        eleList.append(eleTag)
        return eleList, nodeList
    Xi = ops.nodeCoord(ndI, 'X')
    Yi = ops.nodeCoord(ndI, 'Y')
    Xj = ops.nodeCoord(ndJ, 'X')
    Yj = ops.nodeCoord(ndJ, 'Y')
    dX = (Xj - Xi) / numEle
    dY = (Yj - Yi) / numEle
    threeD = True
    if len(ops.nodeCoord(ndI)) < 3:
        threeD = False
    else:
        Zi = ops.nodeCoord(ndI, 'Z')
        Zj = ops.nodeCoord(ndJ, 'Z')
        dZ = (Zj - Zi) / numEle
    nodes = [None] * (numEle + 1)
    nodes[0] = ndI
    nodes[numEle] = ndJ
    for i in range(1, numEle):
        if threeD:
            ops.node(nodeTag, Xi + i * dX, Yi + i * dY, Zi + i * dZ)
        else:
            ops.node(nodeTag, Xi + i * dX, Yi + i * dY)
        nodes[i] = nodeTag
        nodeList.append(nodeTag)
        nodeTag = nodeTag + 1
    # print(eleType,eleTag,ndI,nodes[1],transfTag,integrTag)
    ops.element(eleType, eleTag, ndI, nodes[1], transfTag, integrTag)
    eleList.append(eleTag)
    eleTag = eleTag + 1
    for i in range(1, numEle - 1):
        ops.element(eleType, eleTag, nodes[i], nodes[i + 1], transfTag, integrTag)
        eleList.append(eleTag)
        eleTag = eleTag + 1
    ops.element(eleType, eleTag, nodes[numEle - 1], ndJ, transfTag, integrTag)
    eleList.append(eleTag)
    return eleList, nodeList


class _analysis:

    @staticmethod
    def _mca(secTag, axialLoad, maxK, ctrl_dof, num_steps=1000):
        """
        -------------------------------
        MOMENT-CURVATURE ANALYSIS (MCA)
        -------------------------------
        This method carries out a moment-curvature analysis of a section.
        in particular, displacement controlled algorithm is being used.
        It is intended use internally only.

        Parameters
        ----------
        secTag: int
            Section tag to use
        axialLoad: float
            Axial load on the section
        maxK: float
            Maximum curvature considered in analysis
        ctrl_dof: int
            Control dof (5 or 6)
        num_steps: int
            Number of analysis steps

        Returns
        -------
            Phi: list
                Curvatures
            M: list
                Moments
            eps: list
                Axial strains at section centroid
        """

        # Control node
        ctrl_node = 2

        # Define two nodes at (0,0)
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, 0.0, 0.0, 0.0)

        # Fix all degrees of freedom except axial and bending
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        if ctrl_dof == 5:
            ops.fix(2, 0, 1, 1, 1, 0, 1)
        elif ctrl_dof == 6:
            ops.fix(2, 0, 1, 1, 1, 1, 0)
        else:
            raise ValueError('Wrong dof entry... Use ctrlDOF = 5 or 6!')

        # Define element
        ops.element('zeroLengthSection', 1, 1, 2, secTag)

        # Define constant axial load
        ops.timeSeries('Constant', 1)  # define the timeSeries for the load pattern
        ops.pattern('Plain', 1, 1)  # define load pattern -- generalized
        ops.load(ctrl_node, axialLoad, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Define analysis parameters
        ops.integrator('LoadControl', 0, 1, 0, 0)
        ops.system('SparseGeneral', '-piv')  # Overkill, but may need the pivoting!
        ops.test('EnergyIncr', 1.0e-9, 10)
        ops.numberer('Plain')
        ops.constraints('Plain')
        ops.algorithm('Newton')
        ops.analysis('Static')

        # Do one analysis for constant axial load
        ops.analyze(1)

        # Reset time to 0.0
        ops.loadConst('-time', 0.0)

        # Define reference moment
        ops.timeSeries('Linear', 2)  # define the timeSeries for the load pattern
        ops.pattern('Plain', 2, 2)  # define load pattern -- generalized
        if ctrl_dof == 5:
            ops.load(2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        elif ctrl_dof == 6:
            ops.load(2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Compute curvature increment
        du = maxK / num_steps

        # Use displacement control at node control node for section analysis
        ops.integrator('DisplacementControl', ctrl_node, ctrl_dof, du, 1, du, du)

        # Do section analysis
        curvature = []  # list to store curvatures
        moment = []  # list to store moments
        strain = []  # list to store axial strains
        for ii in range(num_steps):
            ok = ops.analyze(1)
            if ok != 0:
                break
            else:
                curvature.append(ops.nodeDisp(ctrl_node, ctrl_dof))
                moment.append(ops.getLoadFactor(2))
                strain.append(ops.nodeDisp(ctrl_node, 1))

        # ----------------------------------------------if convergence failure-------------------------
        tol_static = 1.e-9
        test_type_static = 'EnergyIncr'
        max_num_iter_static = 6
        algorithm_type_static = 'Newton'
        if ok != 0:
            # if analysis fails, we try some other stuff, performance is slower inside this loop
            step = 0.0
            ok = 0
            while step <= 1.0 and ok == 0:
                control_disp = ops.nodeDisp(ctrl_node, ctrl_dof)
                step = control_disp / maxK
                ok = ops.analyze(1)  # this will return zero if no convergence problems were encountered
                if ok != 0:  # reduce step size if still fails to converge
                    nk = 4  # reduce step size
                    du_reduced = du / nk
                    ops.integrator('DisplacementControl', ctrl_node, ctrl_dof, du_reduced)
                    for ik in range(1, nk + 1, 1):
                        ok = ops.analyze(1)  # this will return zero if no convergence problems were encountered
                        if ok != 0:
                            # if analysis fails, we try some other stuff
                            # performance is slower inside this loop    global maxNumIterStatic;
                            # max no. of iterations performed before "failure to converge" is returned
                            print("Trying Newton with Initial Tangent ..")
                            ops.test('NormDispIncr', tol_static, 2000, 0)
                            ops.algorithm('Newton', '-initial')
                            ok = ops.analyze(1)
                            ops.test(test_type_static, tol_static, max_num_iter_static, 0)
                            ops.algorithm(algorithm_type_static)

                        if ok != 0:
                            print("Trying Broyden ..")
                            ops.algorithm('Broyden', 8)
                            ok = ops.analyze(1)
                            ops.algorithm(algorithm_type_static)

                        if ok != 0:
                            print("Trying NewtonWithLineSearch ..")
                            ops.algorithm('NewtonLineSearch', 0.8)
                            ok = ops.analyze(1)
                            ops.algorithm(algorithm_type_static)

                        if ok == 0:
                            curvature.append(ops.nodeDisp(ctrl_node, ctrl_dof))
                            moment.append(ops.getLoadFactor(2))
                            strain.append(ops.nodeDisp(ctrl_node, 1))

                        if ok != 0:  # stop if still fails to converge
                            return moment, curvature, strain

                    ops.integrator('DisplacementControl', ctrl_node, ctrl_dof, du)  # bring back to original increment

        return moment, curvature, strain

    @staticmethod
    def _nspa(ref_disp, ctrlNode, ctrlDOF, nSteps, IOflag=0):
        """
        ------------------------------------------
        NONLINEAR STATIC PUSHOVER ANALYSIS (NSPA)
        ------------------------------------------
        This method performs a nonlinear static pushover analysis (NSPA) of a model.
        In particular, displacement controlled algorithm is being used.
        It is intended use internally only.

        Parameters
        ----------
        ref_disp: float
            Reference displacement to which cycles are run
        ctrlNode : int
            Node to control with the displacement integrator
        ctrlDOF : float
            Control degrees of freedom for which displacement values are checked
        nSteps : int
            Number of analysis steps
        IOflag : int, optional (The default is 0)
            1; Option to print analysis info at each analysis step

        Returns
        -------

        """

        print(f"Push node {ctrlNode:.0f} to {ref_disp:.3f} m in DOF: {ctrlDOF:.0f}")
        load_factor = [0]
        disp_ctrl_node = [0]
        base_shear = [0]

        # Algorithm Types
        algorithms = [
            ['Newton'],
            ['Newton', '-initial'],
            ['ModifiedNewton', '-initial'],
            ['KrylovNewton'],
            ['NewtonLineSearch', '-InitialInterpolated', 0.8]
        ]

        test_type = 'EnergyIncr'  # Set the initial test type (default)
        tol_init = 1e-6  # Set the initial Tolerance, so it can be referred back to (default)
        iter_init = 20  # Set the initial Max Number of Iterations (default)
        algorithm_type = algorithms[0]
        ops.test(test_type, tol_init, iter_init, 0, 2)  # lets start with energy increment as test
        ops.algorithm(*algorithm_type)
        du = ref_disp / nSteps
        ops.integrator('DisplacementControl', ctrlNode, ctrlDOF, du)
        ops.analysis('Static')

        # Set the initial values to start the while loop
        ok = 0.0
        step = 1.0

        # Restrained nodes in horizontal directions 1 and 2
        fixed_nodes1 = []
        fixed_nodes2 = []
        for node in ops.getNodeTags():
            if ops.nodeDOFs(node)[0] == -1:
                fixed_nodes1.append(node)
            if ops.nodeDOFs(node)[1] == -1:
                fixed_nodes2.append(node)

        # This feature of disabling the possibility of having a negative loading has been included.
        while step <= nSteps and ok == 0 and load_factor[-1] >= 0:
            ok = ops.analyze(1)

            # Change the test type
            if ok != 0 and test_type == 'EnergyIncr':
                print(f" ~~~ Failed at Control Disp: {ops.nodeDisp(ctrlNode, ctrlDOF):.3f} - "
                      "Changing test to NormDispIncr...")

                test_type = 'NormDispIncr'
                ops.test(test_type, tol_init, iter_init, 0, 2)
                ok = ops.analyze(1)
                if ok == 0:
                    print('This has worked...')

            # Change the solution algorithm
            if ok != 0:
                # lets increase the number of iterations for other algorithms than regular newton
                ops.test(test_type, iter_init * 5, 0, 2)
                for algorithm_type in algorithms[1:]:
                    print(f" ~~~ Failed at Control Disp:{ops.nodeDisp(ctrlNode, ctrlDOF):.3f} - "
                          f"Changing algorithm to {algorithm_type}...")
                    ops.algorithm(*algorithm_type)
                    ok = ops.analyze(1)
                    if ok == 0:
                        print(f"The algorithm {algorithm_type} has worked...")
                        break

            if ok != 0:
                # Next change both algorithm and tolerance to achieve convergence if this doesn't work
                # in bocca al lupo....
                ops.test(test_type, tol_init * 100, iter_init * 25, 0, 2)
                for algorithm_type in algorithms:
                    print(f" ~~~ Failed at Control Disp: {ops.nodeDisp(ctrlNode, ctrlDOF):.3f}"
                          " - Relaxing the convergence criteria, increasing maximum number of iterations,"
                          f" and changing algorithm to {algorithm_type}...")
                    ops.algorithm(*algorithm_type)
                    ok = ops.analyze(1)
                    if ok == 0:
                        print(" ~~~ The relaxed convergence criteria has worked...")
                        break

            # Shit...  Failed to converge, exit the analysis.
            if ok != 0:
                print(f" ~~~ Failed at Control Disp:{ops.nodeDisp(ctrlNode, ctrlDOF):.3f} - exit the analysis...")
                ops.wipe()

            else:
                disp1 = ops.nodeDisp(ctrlNode, 1)  # disp in dir 1
                disp2 = ops.nodeDisp(ctrlNode, 2)  # disp in dir 2
                step += 1.0  # update the step number
                load_factor.append(ops.getTime())
                ops.reactions('-dynamic', '-rayleigh')
                shear1 = sum([ops.nodeReaction(node)[0] for node in fixed_nodes1])
                shear2 = sum([ops.nodeReaction(node)[1] for node in fixed_nodes2])
                base_shear.append((shear1 ** 2 + shear2 ** 2) ** 0.5)  # SRSS of base shear in two dirs
                disp_ctrl_node.append((disp1 ** 2 + disp2 ** 2) ** 0.5)  # SRSS of disps in two dirs

                # Print the current displacement
                if IOflag >= 1:
                    print(f"Current Test: {test_type} | Current Algorithm: {algorithm_type[0]} "
                          f"| Control Disp:{ops.nodeDisp(ctrlNode, ctrlDOF):.3f} | Load Factor:{load_factor[-1]:.0f}")

        if ok != 0:
            print("Displacement Control Analysis is FAILED")
            print('-------------------------------------------------------------------------')

        else:
            print("Displacement Control Analysis is SUCCESSFUL")
            print('-------------------------------------------------------------------------')

        if load_factor[-1] <= 0:
            print(f"Stopped because of Load factor below zero:{load_factor[-1]}")
            print('-------------------------------------------------------------------------')

        return base_shear, disp_ctrl_node

    def _get_edp_nrha(self, edps=None):
        """
        -----------------------------------
        ENGINEERING DEMAND PARAMETERS (EDP)
        -----------------------------------
        This method updates edps for each component if they are bigger in current analysis step
        It is intended use internally only.

        Parameters
        ----------
        edps: dictionary
            The dictionary contains maximum of edps from previous analysis steps

        Returns
        -------
        edps: dictionary
            The dictionary contains maximum of edps in the current analysis step
        """

        if edps is None:
            # For 3D fiber sections dofs are: [P,Mz,My,T]
            # If the section is aggregated with shear springs then dofs are: [P,Mz,My,T,Vy,Vz]

            # Initialize the engineering demand parameters to calculate
            edps = {'drift_ratio': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Drift Ratio
                    'Vy': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Shear Force (-z)
                    'Vz': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Shear Force (-y)
                    'V': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Shear Force (srss)
                    'mu_curv_z': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Curvature Ductility (-z)
                    'mu_curv_y': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Curvature Ductility (-y)
                    'mu_curv': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Curvature Ductility (srss)
                    'mu_disp_z': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Displacement Ductility (-z)
                    'mu_disp_y': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Displacement Ductility (-y)
                    'mu_disp': [0 for _ in range(len(self.EleIDsPier))],  # Maximum Pier Displacement Ductility (srss)
                    'abut_disp_long': [0, 0],  # Maximum Abutment Displacements in Longitudinal Direction
                    'abut_disp_active': [0, 0],  # Maximum Abutment Displacements in Active Direction (long)
                    'abut_disp_passive': [0, 0],  # Maximum Abutment Displacements in Passive Direction (long)
                    'abut_disp_transv': [0, 0],  # Maximum Abutment Displacements in Transverse Direction
                    'abut_disp': [0, 0],  # Maximum Abutment Displacements (srss)
                    # Maximum Bearing Displacements in Longitudinal Direction
                    'bearing_disp_long': {joint: [0 for _ in self.EleIDsBearing[joint]] for joint in
                                          self.EleIDsBearing},
                    # Maximum Bearing Displacements in Transverse Direction
                    'bearing_disp_transv': {joint: [0 for _ in self.EleIDsBearing[joint]] for joint in
                                            self.EleIDsBearing},
                    # Maximum Bearing Displacements (srss)
                    'bearing_disp': {joint: [0 for _ in self.EleIDsBearing[joint]] for joint in self.EleIDsBearing},
                    # Unseating
                    'unseating_disp_long': {joint: [0 for _ in self.EleIDsBearing[joint]] for joint in
                                            self.EleIDsBearing}}
            # # Shearkey displacement
            # 'shearkey_disp_long': {joint: [0 for _ in self.EleIDsShearKey[joint]] for joint in
            #                        self.EleIDsShearKey},
            # 'shearkey_disp_transv': {joint: [0 for _ in self.EleIDsShearKey[joint]] for joint in
            #                          self.EleIDsShearKey}}
        # time_array = np.array([0])
        # joints = list(self.EleIDsShearKey.keys())
        # nodes_11 = ops.eleNodes(self.EleIDsShearKey[joints[0]][0])
        # nodes_12 = ops.eleNodes(self.EleIDsShearKey[joints[0]][1])
        # nodes_21 = ops.eleNodes(self.EleIDsShearKey[joints[-1]][0])
        # nodes_22 = ops.eleNodes(self.EleIDsShearKey[joints[-1]][1])
        #
        # sk_11_rel_disp = np.array([[0, 0, 0, 0, 0, 0]])
        # sk_12_rel_disp = np.array([[0, 0, 0, 0, 0, 0]])
        # sk_21_rel_disp = np.array([[0, 0, 0, 0, 0, 0]])
        # sk_22_rel_disp = np.array([[0, 0, 0, 0, 0, 0]])
        #
        # nodes_1 = ops.eleNodes(self.EleIDsShearKey[joints[2]][0])
        # nodes_2 = ops.eleNodes(self.EleIDsShearKey[joints[2]][1])
        # sk_1_rel_disp = np.array([[0, 0, 0, 0, 0, 0]])
        # sk_2_rel_disp = np.array([[0, 0, 0, 0, 0, 0]])

        else:
            # Calculate EDPs for piers

            for i, pier_tag in enumerate(self.EleIDsPier):
                elements = self.EleIDsPier[pier_tag]['elements']
                sec_def = ops.eleResponse(elements[0], 'section', 1, 'deformation')
                nodes = self.EleIDsPier[pier_tag]['nodes']
                node_i = nodes[0]
                node_j = nodes[-1]
                # Vectors defining element orientation - use these to get displacement in local coordinates
                vxz = self.EleIDsPier[pier_tag]['vecxz']
                vx = np.array(ops.nodeCoord(node_j)) - np.array(ops.nodeCoord(node_i))
                vx = vx / np.sqrt(vx.dot(vx))
                vy = np.cross(vxz, vx)
                vz = np.cross(vx, vy)
                global_ele_def = np.array(ops.nodeDisp(node_j)) - np.array(ops.nodeDisp(node_i))
                # ops.reactions('-dynamic', '-rayleigh')
                # global_ele_force = np.array(ops.nodeReaction(node_i))
                local_ele_def = np.append(np.array([vx, vy, vz]).dot(global_ele_def[:3]),
                                          np.array([vx, vy, vz]).dot(global_ele_def[3:]))
                # local_ele_force = np.append(np.array([vx, vy, vz]).dot(global_ele_force[:3]),
                #                             np.array([vx, vy, vz]).dot(global_ele_force[3:]))
                local_ele_force = ops.eleResponse(elements[0], 'localForce')

                # Get shear force
                v_y = abs(local_ele_force[1])
                v_z = abs(local_ele_force[2])
                v = (v_y ** 2 + v_z ** 2) ** 0.5

                # Get drift ratio
                disp_y = abs(local_ele_def[1])
                disp_z = abs(local_ele_def[2])
                drift_ratio = ((disp_z ** 2 + disp_y ** 2) ** 0.5) / self.EleIDsPier[pier_tag]['H']

                # Get displacement ductility (for circular piers)
                mu_disp_y = disp_y / self.EleIDsPier[pier_tag]['Yield Displacement -Y']
                mu_disp_z = disp_z / self.EleIDsPier[pier_tag]['Yield Displacement -Z']
                mu_disp = (mu_disp_y ** 2 + mu_disp_z ** 2) ** 0.5

                # Get curvature ductility (for circular piers)
                mu_curv_y = sec_def[2] / self.EleIDsPier[pier_tag]['Yield Curvature -Y']
                mu_curv_z = sec_def[1] / self.EleIDsPier[pier_tag]['Yield Curvature -Z']
                mu_curv = (mu_curv_y ** 2 + mu_curv_z ** 2) ** 0.5

                # Update EDPs
                if drift_ratio > edps['drift_ratio'][i]:
                    edps['drift_ratio'][i] = drift_ratio
                if mu_disp_z > edps['mu_disp_z'][i]:
                    edps['mu_disp_z'][i] = mu_disp_z
                if mu_disp_y > edps['mu_disp_y'][i]:
                    edps['mu_disp_y'][i] = mu_disp_y
                if mu_disp > edps['mu_disp'][i]:
                    edps['mu_disp'][i] = mu_disp
                if mu_curv_z > edps['mu_curv_z'][i]:
                    edps['mu_curv_z'][i] = mu_curv_z
                if mu_curv_y > edps['mu_curv_y'][i]:
                    edps['mu_curv_y'][i] = mu_curv_y
                if mu_curv > edps['mu_curv'][i]:
                    edps['mu_curv'][i] = mu_curv
                if v_z > edps['Vz'][i]:
                    edps['Vz'][i] = v_z
                if v_y > edps['Vy'][i]:
                    edps['Vy'][i] = v_y
                if v > edps['V'][i]:
                    edps['V'][i] = v

            # Abutment Displacements
            # TODO: save passive and activate displacement separately in longitudinal direction
            abut_conf = self.model['General']['Abutments']
            if self.model[abut_conf]['Type'] != 'Fixed':
                for i, abut_tag in enumerate(self.EleIDsAbut):
                    ele_def = np.array(ops.eleResponse(abut_tag, 'deformation'))
                    ele_nodes = ops.eleNodes(abut_tag)
                    disp_l = ele_def[0]
                    disp_t = ele_def[1]
                    disp = (disp_l ** 2 + disp_t ** 2) ** 0.5
                    if abs(disp_l) > edps['abut_disp_long'][i]:
                        edps['abut_disp_long'][i] = abs(disp_l)
                    if abs(disp_t) > edps['abut_disp_transv'][i]:
                        edps['abut_disp_transv'][i] = abs(disp_t)
                    if disp > edps['abut_disp'][i]:
                        edps['abut_disp'][i] = disp
                    if disp_l < -edps['abut_disp_passive'][i]:
                        edps['abut_disp_passive'][i] = abs(disp_l)
                    if disp_l > edps['abut_disp_active'][i]:
                        edps['abut_disp_active'][i] = disp_l

            # Bearing Displacements
            for joint in self.EleIDsBearing:
                for i, ele_prop in enumerate(self.EleIDsBearing[joint]):
                    # Element info
                    node_i = ele_prop[1]
                    node_j = ele_prop[2]
                    vx = ele_prop[3]
                    vy = ele_prop[4]
                    e_uns = ele_prop[5]
                    side = ele_prop[6]

                    # Get local element deformations
                    vz = np.cross(vx, vy)
                    global_ele_disp = np.array(ops.nodeDisp(node_j)) - np.array(ops.nodeDisp(node_i))
                    local_ele_disp = np.append(np.array([vx, vy, vz]).dot(global_ele_disp[:3]),
                                               np.array([vx, vy, vz]).dot(global_ele_disp[3:]))

                    # bearing_tag = ele_prop[0]
                    # ele_disp = ops.eleResponse(bearing_tag, 'localDisplacement')
                    # local_ele_disp = abs(np.array(ele_disp[6:]) - np.array(ele_disp[:6]))
                    # if not any(local_ele_def):
                    #     local_ele_disp = np.abs(np.array(ops.eleResponse(bearing_tag, 'deformation')))

                    # Update edps
                    disp_l = abs(local_ele_disp[2])
                    disp_t = abs(local_ele_disp[1])
                    disp = (disp_l ** 2 + disp_t ** 2) ** 0.5
                    if edps['bearing_disp_long'][joint][i] < disp_l:
                        edps['bearing_disp_long'][joint][i] = disp_l
                    if edps['bearing_disp_transv'][joint][i] < disp_t:
                        edps['bearing_disp_transv'][joint][i] = disp_t
                    if edps['bearing_disp'][joint][i] < disp:
                        edps['bearing_disp'][joint][i] = disp
                    if side == 'left' and local_ele_disp[2] < -edps['unseating_disp_long'][joint][i]:
                        edps['unseating_disp_long'][joint][i] = disp_l
                    elif side == 'right' and local_ele_disp[2] > edps['unseating_disp_long'][joint][i]:
                        edps['unseating_disp_long'][joint][i] = disp_l
                    else:
                        continue

            # time_array = np.append(time_array, control_time)
            # rel_disp_11 = np.array([ops.nodeDisp(nodes_11[1])]) - np.array([ops.nodeDisp(nodes_11[0])])
            # rel_disp_12 = np.array([ops.nodeDisp(nodes_12[1])]) - np.array([ops.nodeDisp(nodes_12[0])])
            # rel_disp_21 = np.array([ops.nodeDisp(nodes_21[1])]) - np.array([ops.nodeDisp(nodes_21[0])])
            # rel_disp_22 = np.array([ops.nodeDisp(nodes_22[1])]) - np.array([ops.nodeDisp(nodes_22[0])])
            #
            # rel_disp_1 = np.array([ops.nodeDisp(nodes_1[1])]) - np.array([ops.nodeDisp(nodes_1[0])])
            # rel_disp_2 = np.array([ops.nodeDisp(nodes_2[1])]) - np.array([ops.nodeDisp(nodes_2[0])])
            #
            # sk_11_rel_disp = np.append(sk_11_rel_disp, rel_disp_11, axis=0)
            # sk_12_rel_disp = np.append(sk_12_rel_disp, rel_disp_12, axis=0)
            # sk_21_rel_disp = np.append(sk_21_rel_disp, rel_disp_21, axis=0)
            # sk_22_rel_disp = np.append(sk_22_rel_disp, rel_disp_22, axis=0)
            #
            # sk_1_rel_disp = np.append(sk_1_rel_disp, rel_disp_1, axis=0)
            # sk_2_rel_disp = np.append(sk_2_rel_disp, rel_disp_2, axis=0)

        return edps

    def _nrha(self, dt, duration, dc, pflag=0):
        """
        ------------------------------------------
        NONLINEAR RESPONSE HISTORY ANALYSIS (NRHA)
        ------------------------------------------
        This method performs the nonlinear response history analysis (NRHA)
        It is intended use internally only.

        Notes on Integrators
        --------------------
        The default integrator is TRBDF2.

        -   ops.integrator('Newmark', gamma, beta): does not introduce numerical damping.
            gamma = 1/2, beta = 1/4 --> Average Acceleration Method; unconditionally stable
            gamma = 1/2, beta = 1/6 --> Linear Acceleration Method; conditionally stable dt/T <0.551

        -   ops.integrator('HHT', alpha): Hilber-Hughes-Taylor integrator introduces numerical damping.
            alpha = 1.0 = Newmark Method. Smaller alpha means greater numerical damping.
            0.67<alpha<1.0 # recommended. Leave beta and gamma as default for unconditional stability.

        -   ops.integrator('GeneralizedAlpha', alphaM, alphaF): introduces numerical damping.
            alphaF and alphaM are defined differently that in the paper, we use alpha_F = (1-alpha_f) and
            alpha_M=(1-alpha_m) where alpha_f and alpha_m are those used in the paper.
            1. Like Newmark and all the implicit schemes, the unconditional stability of this method
            applies to linear problems. There are no results showing stability of this method over the
            wide range of nonlinear problems that potentially exist. Experience indicates that the time
            step for implicit schemes in nonlinear situations can be much greater than those for explicit schemes.
            2. alphaM = 1.0, alphaF = 1.0 produces the Newmark Method.
            3. alphaM = 1.0 corresponds to the HHT method.
            4. The method is second-order accurate provided gamma=0.5+alphaM-alphaF, beta=(1+alphaM-alphaF)**2/4
            These are optional parameters that can be used, but default values satisfy this condition
            5. The method is unconditionally stable provided alphaM >= alphaF >= 0.5, beta >= 0.25+0.5*(alphaM-alphaF)
            The following relationships minimize low-frequency damping and maximize high-frequency damping
            pinf = 1 no dissipation of high frequency response
            pinf = 0 annihilation of high-frequency response
            alpha_m = (2*pinf-1)/(pinf+1)
            alpha_f = pinf/(1+pinf)
            alphaM = 1-alpha_m
            alphaF = 1-alpha_f

        -   ops.integrator('TRBDF2'):
            Is a composite scheme that alternates between the Trapezoidal scheme and a 3 point backward Euler scheme.
            It does this in an attempt to conserve energy and momentum, something Newmark does not always do.

        Notes on Algorithms
        -------------------
        -   ops.algorithm('Newton'): Classic Newton-Raphson is the most efficient algorithm.
            There are however more complicated instances where the Newton-Raphson algorithm is less robust.
            For example, when there is bad tangent as nonlinearity is significant it may not converge.
        -   ops.algorithm('KrylowNewton') is slower than classic Newton-Raphson since it may require more iterations
            to reach convergence. Yet, it is still computationally efficient
        -   ops.algorithm('Newton', '-initial'): it uses initial stiffness and is slower than both Newton-Raphson
            and KrylovNewton since it may require more iterations to reach convergence.

        Parameters
        ----------
        dt: float
            Analysis time step
        duration : float
            Total analysis duration
        dc : float
            Drift capacity (%) to define local collapse in NRHA. The analysis stops if pier reaches this drift level
        pflag : int, optional (The default is 0)
            Flag to print details of the analysis on the screen

        Returns
        -------
        edps: dictionary
            Dictionary containing maximum of edps obtained throughout the analysis
        c_index: int
            Index that states the analysis status in the end
            -1: There is a convergence issue
            0: Local collapse is observed
            1: Analysis is successful
        Analysis: str
            Text in which essential results are stated
        """

        analysis = 0
        max_drift_ele_id = 0
        max_drift = 0.0  # Set initially the maximum of all pier drifts (SRSS)
        # Initialize the engineering demand parameters to calculate
        edps = self._get_edp_nrha()  # initialize the edp calculations

        algorithms = [
            ['Newton'],
            ['KrylovNewton'],
            ['NewtonLineSearch', '-InitialInterpolated', 0.8]
            # ['Newton', '-Initial']

        ]

        max_iterations = [20, 50, 100]
        # Define the Initial Analysis Parameters
        tol = 1.0e-6  # Set the initial Tolerance, so it can be referred back to (default)
        test = 'NormDispIncr'  # Set the initial test type (default)
        # Set up analysis parameters
        c_index = 0  # Initially define the control index (-1 for non-converged, 0 for stable, 1 for global collapse)
        control_time = 0.0  # Start the control time
        ok = 0  # Set the convergence to 0 (initially converged)
        dtt = dt / 1  # analysis time step
        ops.integrator('TRBDF2')  # set the integrator

        # Run the actual analysis now
        while c_index == 0 and control_time <= duration and ok == 0:
            if pflag > 1:
                print(f"Completed {control_time:.2f} of {duration:.2f} seconds")

            # Gradually increase the time increment size
            if dtt == dt / 4:
                dtt = dt / 2
            elif dtt == dt / 2:
                dtt = dt / 1

            ops.test(test, tol, max_iterations[0], 0, 2)  # set the test
            ops.algorithm(*algorithms[0])  # set the algorithm
            ops.analysis('Transient')  # set the type of analysis
            ok = ops.analyze(1, dtt)  # Run a step of the analysis
            control_time = ops.getTime()  # Update the control time

            # If the analysis fails, try the following changes to achieve convergence
            # Analysis will be slower in here though...
            # Reduce analysis the time step increment
            if ok != 0 and dtt > dt / 2:
                print(f" ~~~ Failed at {control_time:.3f} of {duration:.3f} - Reduced time step by half...")
                dtt = dt / 2
                ok = ops.analyze(1, dtt)
                if ok == 0:
                    print(f" ~~~ The reduced time step has worked {dtt}")

            if ok != 0 and dtt > dt / 4:
                print(f" ~~~ Failed at {control_time:.3f} of {duration:.3f} - Reduced time step by quarter...")
                dtt = dt / 4
                ok = ops.analyze(1, dtt)
                if ok == 0:
                    print(f" ~~~ The reduced time step has worked: {dtt}")

            if ok != 0:
                # Try changing the algorithm
                for algorithm, max_iter in zip(algorithms[1:], max_iterations[1:]):
                    print(f" ~~~ Failed at {control_time:.3f} of {duration:.3f} - Moving to the next algorithm...")
                    ops.test(test, tol, max_iter, 0, 2)
                    ops.algorithm(*algorithm)
                    ok = ops.analyze(1, dtt)  # Run a step of the analysis
                    if ok == 0:
                        print(f"The algorithm has worked: {algorithm}")
                        break

            if ok != 0:
                # Next change both algorithm, tolerance to achieve convergence and maximum number of iterations
                # if this doesn't work in bocca al lupo....
                print(
                    f"~~~ Failed at {control_time:.3f} of {duration:.3f} - Relaxing the convergence criteria and increasing maximum "
                    f"number of iterations...")
                for algorithm, max_iter in zip(algorithms, max_iterations):
                    print(f" ~~~ Failed at {control_time:.2f} - Moving to the next algorithm...")
                    ops.test(test, tol * 100, max_iter * 2, 0, 2)
                    ops.algorithm(*algorithm)
                    ok = ops.analyze(1, dtt)
                    if ok == 0:
                        print(" ~~~ The relaxed convergence criteria has worked.")
                        break

            # Shit...  Failed to converge, exit the analysis.
            if ok != 0:
                print(f" ~~~ Failed at {control_time:.3f} of {duration:.3f} - exit the analysis......")
                ops.wipe()
                c_index = -1

            if ok == 0:
                # Update the engineering demand parameters
                edps = self._get_edp_nrha(edps)
                if max(edps['drift_ratio']) * 100 > max_drift:
                    max_drift = max(edps['drift_ratio']) * 100
                    max_drift_location = edps['drift_ratio'].index(max(edps['drift_ratio']))
                    max_drift_ele_id = list(self.EleIDsPier.keys())[max_drift_location]

                
                if max_drift >= dc: # check if drift limit for local collapse is exceeded
                    c_index = 1 # Set the state of the model to local collapse (=1)
                    max_drift = dc
                    ops.wipe()

        if c_index == -1:
            analysis = f"Analysis is FAILED to converge at {control_time:.3f} of {duration:.3f}"
        if c_index == 0:
            analysis = f"Analysis is SUCCESSFULLY completed, Peak Pier Drift: {max_drift:.2f}%% at {max_drift_ele_id}"
        if c_index == 1:
            analysis = f"Analysis is STOPPED, peak column drift ratio, {dc}%%, is exceeded, global COLLAPSE is observed"

        print(analysis)

        return edps, c_index, analysis

    def _eigen(self, numEigen, pflag=0):
        """
        -------------------
        EIGENVALUE ANALYSIS
        -------------------
        This method performs eigenvalue analysis
        It is intended use internally only.

        Parameters
        ----------
        numEigen: int
            Number of eigenvalues to calculate
        pflag : int, optional (The default is 0)
            Eigenvalues

        Returns
        -------
        eigen_values: numpy.ndarray
            Dictionary containing maximum of edps obtained throughout the analysis
        """

        eigen_values = 0
        if pflag == 1:
            print('Performing eigenvalue analysis.')
        list_solvers = ['-genBandArpack', '-fullGenLapack', '-symmBandLapack']
        ok = 1
        for s in list_solvers:
            if pflag == 1:
                print(f"Using {s[1:]} as solver...")
            try:
                eigen_values = ops.eigen(s, numEigen)
                catch_ok = 0
                ok = 0
            except BaseException as e:
                catch_ok = 1
                if pflag == 1:
                    print(f"Error: {e}")

            if catch_ok == 0:
                for i in range(numEigen):
                    if eigen_values[i] < 0 or eigen_values[i] > 1e+300:
                        ok = 1
                if ok == 0:
                    if pflag == 1:
                        print('Eigenvalue analysis is completed.')
                    break

        if ok != 0:
            error = "Could not complete the eigenvalue analysis, something is wrong...\n" + \
                    "Try to reduce number of modes to determine..."
            raise ValueError(error)
        else:
            eigen_values = np.asarray(eigen_values)
            return eigen_values


class _rendering:

    def set_animation(self, Movie, FrameStep, scale, fps):
        """
        -------------------------------------
        CONFIGURATION OF ANIMATION PARAMETERS
        -------------------------------------
        This function is used to set parameters to animate structural response.

        Parameters
        ----------
        Movie: int
            flag to save animation  (0 or 1).
        FrameStep: int
            frame step size to use for plotting
        scale: int
            scaling factor to use for animation
        fps: int
            frames per second to use in animation

        Returns
        -------

        """

        self.animate = 1
        self.Movie = Movie
        self.FrameStep = FrameStep
        self.scale = scale
        self.fps = fps

    def _animation_recorders(self):
        """
        This function saves the nodes and elements for an active model, in a
        standardized format. The OpenSees model must be active in order for the
        function to work.
        """

        # Get nodes and elements
        nodeList = ops.getNodeTags()
        eleList = ops.getEleTags()
        dofList = [1, 2, 3]
        # Consider making these optional arguments
        nodeName = 'Nodes'
        eleName = 'Elements'
        delim = ' '
        fmt = '%.10e'
        ftype = '.out'

        dispFile = os.path.join(self.animation_dir, 'NodeDisp_All.out')
        ops.recorder('Node', '-file', dispFile, '-time',
                     '–node', *nodeList, '-dof', *dofList, 'disp')

        ops.record()

        # Check Number of dimensions and initialize variables
        ndm = len(ops.nodeCoord(nodeList[0]))
        Nnodes = len(nodeList)
        nodes = np.zeros([Nnodes, ndm + 1])

        # Get Node list
        for ii, node in enumerate(nodeList):
            nodes[ii, 0] = node
            nodes[ii, 1:] = ops.nodeCoord(nodeList[ii])

        Nele = len(eleList)
        elements = [None] * Nele

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
        nodeFile = os.path.join(self.animation_dir, nodeName + ftype)
        np.savetxt(nodeFile, nodes, delimiter=delim, fmt=fmt)

        # Save element arrays
        eleFile = os.path.join(self.animation_dir, eleName + ftype)
        np.savetxt(eleFile, eleNode, delimiter=delim, fmt=fmt)

    def plot_sections(self, save=1, show=0):
        # Get unique bearing types
        confs = list(set(self.model['General']['Bents']))
        types = []
        for i in range(len(confs)):
            conf_tag = confs[i]
            types.extend(self.model[conf_tag]['sections'])
        types = list(set(types))

        for sec_type in types:
            D = self.model[sec_type]['D']  # Section diameter
            cc = self.model[sec_type]['cover']  # Section diameter
            numBars = self.model[sec_type]['number of bars']  # Section diameter
            dbl = self.model[sec_type]['dl']  # Section diameter
            barArea = np.pi * dbl ** 2 / 4
            yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT = self._circ_fiber_config(D)
            rc = ro - cc
            barRplotfactor = 1
            filename = os.path.join(self.out_dir, sec_type + '.svg')
            plt.figure()
            self._plot_patchcirc(yC, zC, nfCoverT, nfCoverR, rc, ro, startAng, endAng,
                                 'lightgrey')  # unconfined concrete
            self._plot_patchcirc(yC, zC, nfCoreT, nfCoreR, ri, rc, startAng, endAng, 'grey')  # confined concrete
            self._plot_layercirc(numBars, barArea, rc, startAng, endAng, barRplotfactor)
            plt.title('Pier Section - %s' % sec_type)
            plt.xlim([-ro, ro])
            plt.ylim([-ro, ro])
            plt.xlabel('z-coord (m)')
            plt.ylabel('y-coord (m)')
            if show == 1:
                plt.show()
            if save == 1:
                plt.savefig(filename, bbox_inches='tight')

            plt.close()

    def _plot_patchcirc(self, yC, zC, nfp, nft, intR, extR, startAng, endAng, mcolor):
        # yC, zC: y & z-coordinates of the center of the circle
        # nft: number of radial divisions in the core (number of "rings")
        # nfp: number of theta divisions in the core (number of "wedges")
        # intR:	internal radius
        # extR:	external radius
        # startAng:	starting angle
        # endAng:	ending angle
        # mcolor: color to use for fiber fill

        Rvals = np.linspace(intR, extR, nft + 1)
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

    def _plot_layercirc(self, nbars, barA, midR, startAng, endAng, barRplotfactor=1):
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

    def plot_model(self, show_node_tags='no', show_element_tags='no', show_node='no', show=0, save=1):
        #  ------------------------------------------------------------------------------------------------------------
        #  MODEL PLOTTING
        #  ------------------------------------------------------------------------------------------------------------
        nodeList = ops.getNodeTags()
        eleList = ops.getEleTags()

        beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': '-'}  # beam elements
        bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # bearing elements
        shear_key_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # shear key elements
        soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': '-'}  # spring elements
        rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # rigid links
        joint_style = {'color': 'blue', 'linewidth': 1, 'linestyle': '-'}  # joint
        node_style = {'s': 3, 'color': 'black', 'marker': 'o', 'facecolor': 'black'}
        node_text_style = {'fontsize': 6, 'fontweight': 'regular', 'color': 'green'}
        ele_text_style = {'fontsize': 6, 'fontweight': 'bold', 'color': 'darkred'}

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ELEMENTS
        for element in eleList:
            if str(element)[-2:] == self.BearingTag[-2:]:
                ele_style = bearing_style
            elif str(element)[-2:] == self.ShearKeyTag[-2:]:
                ele_style = shear_key_style
            elif str(element)[-2:] == self.JointTag[-2:]:
                ele_style = joint_style
            elif str(element)[-2:] == self.RigidTag[-2:]:
                ele_style = rigid_style
            elif str(element)[-2:] == self.SpringEleTag[-2:]:
                ele_style = soil_style
            elif str(element)[-2:] == self.AbutTag[-2:]:
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
        if self.const_opt != 1:
            for i in range(len(self.RigidLinkNodes)):
                iNode = ops.nodeCoord(self.RigidLinkNodes[i][0])
                jNode = ops.nodeCoord(self.RigidLinkNodes[i][1])
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
        ax.set_xlim(xViewCenter - (view_range / 3), xViewCenter + (view_range / 3))
        ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
        ax.set_zlim(zViewCenter - (view_range / 5), zViewCenter + (view_range / 5))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.text2D(0.10, 0.95, "Undeformed shape", transform=ax.transAxes, fontweight="bold")
        ax.view_init(elev=30, azim=140)
        plt.tight_layout()
        plt.axis('off')
        filename = os.path.join(self.out_dir, 'Model.svg')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def plot_deformedshape(self, ax=None, scale=5, save=1, show=0):
        # scale: scale factor to be applied
        # ax: the axis handler to plot the deformed shape
        if ax is None:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(1, 1, 1, projection='3d')

        eleList = ops.getEleTags()
        dofList = [1, 2, 3]
        Disp_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
        beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': ':'}  # beam elements
        bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': ':'}  # bearing elements
        soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
        rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': ':'}  # rigid links
        joint_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # joint elements
        shear_key_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # shear key elements

        x = []
        y = []
        z = []
        for element in eleList:
            if str(element)[-2:] == self.BearingTag[-2:]:
                ele_style = bearing_style
            elif str(element)[-2:] == self.ShearKeyTag[-2:]:
                ele_style = shear_key_style
            elif str(element)[-2:] == self.JointTag[-2:]:
                ele_style = joint_style
            elif str(element)[-2:] == self.RigidTag[-2:]:
                ele_style = rigid_style
            elif str(element)[-2:] == self.SpringEleTag[-2:]:
                ele_style = soil_style
            elif str(element)[-2:] == self.AbutTag[-2:]:
                ele_style = soil_style
            else:
                ele_style = beam_style
            Disp_style['color'] = ele_style['color']

            Nodes = ops.eleNodes(element)
            iNode = ops.nodeCoord(Nodes[0])
            jNode = ops.nodeCoord(Nodes[1])

            iNode_Disp = []
            jNode_Disp = []
            for dof in dofList:
                iNode_Disp.append(ops.nodeDisp(Nodes[0], dof))
                jNode_Disp.append(ops.nodeDisp(Nodes[1], dof))
            # Add original and deformed shape to get final node coordinates
            iNode_final = [iNode[0] + scale * iNode_Disp[0], iNode[1] +
                           scale * iNode_Disp[1], iNode[2] + scale * iNode_Disp[2]]
            jNode_final = [jNode[0] + scale * jNode_Disp[0], jNode[1] +
                           scale * jNode_Disp[1], jNode[2] + scale * jNode_Disp[2]]

            x.append(iNode[0])
            x.append(jNode[0])  # list of x coordinates to define plot view area
            y.append(iNode[1])
            y.append(jNode[1])  # list of y coordinates to define plot view area
            z.append(iNode[2])
            z.append(iNode[2])  # list of z coordinates to define plot view area

            ax.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)
            ax.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                    (iNode_final[2], jNode_final[2]),
                    marker='', **Disp_style)

        # RIGID LINKS if constraints are used
        if self.const_opt != 1:
            Eig_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
            for i in range(len(self.RigidLinkNodes)):
                iNode = ops.nodeCoord(self.RigidLinkNodes[i][0])
                jNode = ops.nodeCoord(self.RigidLinkNodes[i][1])
                iNode_Disp = []
                jNode_Disp = []
                for dof in dofList:
                    iNode_Disp.append(ops.nodeDisp(self.RigidLinkNodes[i][0], dof))
                    jNode_Disp.append(ops.nodeDisp(self.RigidLinkNodes[i][1], dof))

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
        ax.set_xlim(xViewCenter - (view_range / 3), xViewCenter + (view_range / 3))
        ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
        ax.set_zlim(zViewCenter - (view_range / 5), zViewCenter + (view_range / 5))
        ax.text2D(0.10, 0.95, "Deformed shape", transform=ax.transAxes, fontweight="bold")
        ax.text2D(0.10, 0.90, "Scale Factor: " + str(scale), transform=ax.transAxes, fontweight="bold")
        ax.view_init(elev=30, azim=140)
        ax.axis('off')
        filename = os.path.join(self.out_dir, 'Deformed_Shape.svg')
        plt.tight_layout()
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def plot_modeshape(self, modeNumber=1, scale=200, show=0, save=1):
        #  ------------------------------------------------------------------------------------------------------------
        #  MODE SHAPE PLOTTING
        #  ------------------------------------------------------------------------------------------------------------
        # Get periods
        periods = self.modal_properties['Periods'].copy()

        eleList = ops.getEleTags()
        beam_style = {'color': 'black', 'linewidth': 1, 'linestyle': ':'}  # beam elements
        bearing_style = {'color': 'red', 'linewidth': 1, 'linestyle': ':'}  # bearing elements
        soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
        rigid_style = {'color': 'gray', 'linewidth': 1, 'linestyle': ':'}  # rigid links
        Eig_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
        joint_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # joint elements
        shear_key_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # shear key elements

        x = []
        y = []
        z = []
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for element in eleList:
            if str(element)[-2:] == self.BearingTag[-2:]:
                ele_style = bearing_style
            elif str(element)[-2:] == self.ShearKeyTag[-2:]:
                ele_style = shear_key_style
            elif str(element)[-2:] == self.JointTag[-2:]:
                ele_style = joint_style
            elif str(element)[-2:] == self.RigidTag[-2:]:
                ele_style = rigid_style
            elif str(element)[-2:] == self.SpringEleTag[-2:]:
                ele_style = soil_style
            elif str(element)[-2:] == self.AbutTag[-2:]:
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
        if self.const_opt != 1:
            Eig_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
            for i in range(len(self.RigidLinkNodes)):
                iNode = ops.nodeCoord(self.RigidLinkNodes[i][0])
                jNode = ops.nodeCoord(self.RigidLinkNodes[i][1])
                iNode_Eig = ops.nodeEigenvector(self.RigidLinkNodes[i][0], modeNumber)
                jNode_Eig = ops.nodeEigenvector(self.RigidLinkNodes[i][1], modeNumber)

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
        ax.set_xlim(xViewCenter - (view_range / 3), xViewCenter + (view_range / 3))
        ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
        ax.set_zlim(zViewCenter - (view_range / 5), zViewCenter + (view_range / 5))
        ax.text2D(0.10, 0.95, "Mode " + str(modeNumber), transform=ax.transAxes, fontweight="bold")
        ax.text2D(0.10, 0.90, f"T = {periods[modeNumber - 1]:.3f} s", transform=ax.transAxes,
                  fontweight="bold")
        ax.view_init(elev=30, azim=140)
        plt.axis('off')
        filename = os.path.join(self.out_dir, 'Mode_Shape_' + str(modeNumber) + '.svg')
        plt.tight_layout()
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        ops.wipeAnalysis()

    def _animate_nspa(self, loading_scheme, BaseShear, DispCtrlNode, ctrlNode):
        nodeFile = os.path.join(self.animation_dir, 'Nodes.out')
        eleFile = os.path.join(self.animation_dir, 'Elements.out')
        dispFile = os.path.join(self.animation_dir, 'NodeDisp_All.out')
        Movie = self.Movie
        scale = self.scale
        fps = self.fps
        FrameStep = self.FrameStep
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
        soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
        Disp_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
        node_style = {'s': 12, 'color': 'red', 'marker': 'o', 'facecolor': 'blue'}
        node_text_style = {'fontsize': 10, 'fontweight': 'regular', 'color': 'green'}
        joint_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # joint elements
        shear_key_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # shear key elements

        fig = plt.figure(figsize=(18, 8))
        plt.suptitle('%s pushover with control node: %d' % (loading_scheme, ctrlNode), fontweight="bold", y=0.92)
        plt.tight_layout()
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 5])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], projection='3d')

        # plot the pushover curve
        temp = ax1.plot(DispCtrlNode[:0], BaseShear[:0], lw=2, color='red')
        dynLines1 = temp[0]
        ax1.grid(True)
        ax1.set_xlabel('$u_{ctrl}$ [m]')
        ax1.set_ylabel('$V_{base}$ [kN]')
        ax1.set_xlim([0, 1.2 * max(DispCtrlNode)])
        ax1.set_ylim([0, 1.2 * max(BaseShear)])

        x = []
        y = []
        z = []
        ctrl_plot = 0
        for i in range(len(elements[:, 0])):
            element = int(elements[i, 0])
            if str(element)[-2:] == self.BearingTag[-2:]:
                ele_style = bearing_style
            elif str(element)[-2:] == self.ShearKeyTag[-2:]:
                ele_style = shear_key_style
            elif str(element)[-2:] == self.JointTag[-2:]:
                ele_style = joint_style
            elif str(element)[-2:] == self.RigidTag[-2:]:
                ele_style = rigid_style
            elif str(element)[-2:] == self.SpringEleTag[-2:]:
                ele_style = soil_style
            elif str(element)[-2:] == self.AbutTag[-2:]:
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

            ax2.plot((iNode[0], jNode[0]), (iNode[1], jNode[1]), (iNode[2], jNode[2]), marker='', **ele_style)
            temp = ax2.plot((iNode_final[0], jNode_final[0]), (iNode_final[1], jNode_final[1]),
                            (iNode_final[2], jNode_final[2]),
                            marker='', **Disp_style)
            dynLines2.append(temp[0])

            if ctrlNode == nodes[idx_i, 0] and ctrl_plot == 0:
                ctrl_plot = 1
                ax2.scatter(iNode[0], iNode[1], iNode[2], **node_style)
                dynNode = ax2.scatter(iNode_final[0], iNode_final[1], iNode_final[2], **node_style)
                ax2.text(iNode[0] * 1.02, iNode[1] * 1.02, iNode[2] * 1.02, str(ctrlNode),
                         **node_text_style)  # label nodes

            if ctrlNode == nodes[idx_j, 0] and ctrl_plot == 0:
                ctrl_plot = 1
                ax2.scatter(jNode[0], jNode[1], jNode[2], **node_style)
                dynNode = ax2.scatter(jNode_final[0], jNode_final[1], jNode_final[2], **node_style)
                ax2.text(jNode[0] * 1.02, jNode[1] * 1.02, jNode[2] * 1.02, str(ctrlNode),
                         **node_text_style)  # label nodes

        # RIGID LINKS if constraints are used
        if self.const_opt != 1:
            Disp_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
            for i in range(len(self.RigidLinkNodes)):
                idx_i = np.where(nodes[:, 0] == self.RigidLinkNodes[i][0])[0][0]
                idx_j = np.where(nodes[:, 0] == self.RigidLinkNodes[i][1])[0][0]
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
        xmin = xViewCenter - (view_range / 4 * 1.3)
        xmax = xViewCenter + (view_range / 4) * 1.1
        ymin = yViewCenter - (view_range / 6)
        ymax = yViewCenter + (view_range / 6)
        zmin = zViewCenter - (view_range / 10)
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
                if is_paused:
                    is_paused = False
                elif not is_paused:
                    is_paused = True

        def animate_slider(Time):

            global is_paused
            is_paused = True
            now = framesTime[(np.abs(framesTime - plotSlider.val)).argmin()]
            tStep = (np.abs(timeArray - now)).argmin()
            dynLines1.set_data(DispCtrlNode[:tStep], BaseShear[:tStep])

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

            if self.const_opt != 1:
                for i in range(len(self.RigidLinkNodes)):
                    idx_i = np.where(nodes[:, 0] == self.RigidLinkNodes[i][0])[0][0]
                    idx_j = np.where(nodes[:, 0] == self.RigidLinkNodes[i][1])[0][0]
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
            Movfile = os.path.join(self.animation_dir, 'NSPA.mp4')
            ani = animation.FuncAnimation(fig, update_plot, aniFrames, interval=FrameInterval)
            ani.save(Movfile, writer='ffmpeg')
            print('Animation is saved!')

    def _animate_nrha(self):
        nodeFile = os.path.join(self.animation_dir, 'Nodes.out')
        eleFile = os.path.join(self.animation_dir, 'Elements.out')
        dispFile = os.path.join(self.animation_dir, 'NodeDisp_All.out')
        Movie = self.Movie
        scale = self.scale
        fps = self.fps
        FrameStep = self.FrameStep
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
        Disp_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # elements
        soil_style = {'color': 'green', 'linewidth': 1, 'linestyle': ':'}  # spring elements
        joint_style = {'color': 'blue', 'linewidth': 1, 'linestyle': ':'}  # joint elements
        shear_key_style = {'color': 'red', 'linewidth': 1, 'linestyle': '-'}  # shear key elements

        fig = plt.figure(figsize=(14.4, 8.1))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        dynLines = []
        dynLines2 = []
        x = []
        y = []
        z = []
        for i in range(len(elements[:, 0])):
            element = int(elements[i, 0])
            if str(element)[-2:] == self.BearingTag[-2:]:
                ele_style = bearing_style
            elif str(element)[-2:] == self.ShearKeyTag[-2:]:
                ele_style = shear_key_style
            elif str(element)[-2:] == self.JointTag[-2:]:
                ele_style = joint_style
            elif str(element)[-2:] == self.RigidTag[-2:]:
                ele_style = rigid_style
            elif str(element)[-2:] == self.SpringEleTag[-2:]:
                ele_style = soil_style
            elif str(element)[-2:] == self.AbutTag[-2:]:
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
        if self.const_opt != 1:
            Disp_style = {'color': 'gray', 'linewidth': 1, 'linestyle': '-'}  # elements
            for i in range(len(self.RigidLinkNodes)):
                idx_i = np.where(nodes[:, 0] == self.RigidLinkNodes[i][0])[0][0]
                idx_j = np.where(nodes[:, 0] == self.RigidLinkNodes[i][1])[0][0]
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
        ax.set_xlim(xViewCenter - (view_range / 3), xViewCenter + (view_range / 3))
        ax.set_ylim(yViewCenter - (view_range / 4), yViewCenter + (view_range / 4))
        ax.set_zlim(zViewCenter - (view_range / 5), zViewCenter + (view_range / 5))
        ax.text2D(0.10, 0.95, "Deformed shape", transform=ax.transAxes, fontweight="bold")
        ax.text2D(0.10, 0.90, "Scale Factor: " + str(scale), transform=ax.transAxes, fontweight="bold")
        ax.view_init(elev=30, azim=140)
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

            if self.const_opt != 1:
                for i in range(len(self.RigidLinkNodes)):
                    idx_i = np.where(nodes[:, 0] == self.RigidLinkNodes[i][0])[0][0]
                    idx_j = np.where(nodes[:, 0] == self.RigidLinkNodes[i][1])[0][0]
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
            Movfile = os.path.join(self.animation_dir, 'NRHA.mp4')
            ani = animation.FuncAnimation(fig, update_plot, aniFrames, interval=FrameInterval)
            ani.save(Movfile, writer='ffmpeg')
            print('Animation is saved!')

    def _bilin_pushover(self, out_dir, BaseShear, DispCtrlNode, M_star, Bilin_approach, loading_scheme, ctrlNode):
        # idealized elasto-perfectly plastic force-displacement relationship using equal energy approach
        if Bilin_approach == 'EC':  # Eurocode 8 Approach
            idx_star = BaseShear.index(max(BaseShear))
            Fy_star = BaseShear[idx_star]
            Fu_star = Fy_star
            Du_star = DispCtrlNode[idx_star]
            E_star = np.trapz(BaseShear[:idx_star], DispCtrlNode[:idx_star])
            Dy_star = 2 * (Du_star - E_star / Fy_star)

        elif Bilin_approach == 'ASCE':  # ASCE 7-16 approach
            ninterp = 1e4
            idx_star = BaseShear.index(max(BaseShear))
            Fu_star = BaseShear[idx_star]
            Du_star = DispCtrlNode[idx_star]
            E_star = np.trapz(BaseShear[:idx_star], DispCtrlNode[:idx_star])
            tolcheck = 1e-2  # 1% error is accepted
            Fy_range = np.arange(0.3 * Fu_star, 0.95 * Fu_star, (0.65 * Fu_star) / ninterp)
            Dy_range = interp1d(BaseShear, DispCtrlNode)(Fy_range)
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
            idx_min = BaseShear.index(max(BaseShear))
            diff_F = abs(np.asarray(BaseShear[idx_min:]) - max(BaseShear) * 0.85)
            diff_F = diff_F.tolist()
            idx_max = idx_min + diff_F.index(min(diff_F))
            tolcheck = 1e-2  # 1% error is accepted
            Du_range = np.arange(DispCtrlNode[idx_min], DispCtrlNode[idx_max],
                                 (DispCtrlNode[idx_max] - DispCtrlNode[idx_min]) / ninterp)
            Fu_range = interp1d(DispCtrlNode, BaseShear)(Du_range)
            Dy_range = np.arange(0, DispCtrlNode[idx_min], DispCtrlNode[idx_min] / ninterp)
            Fy_range = interp1d(DispCtrlNode, BaseShear)(Dy_range)

            for idx in range(len(Fu_range)):
                Fu_trial = Fu_range[idx]
                Du_trial = Du_range[idx]
                Fy_trial = Fu_trial
                diff_D = abs(np.asarray(DispCtrlNode) - Du_trial)
                diff_D = diff_D.tolist()
                idx_trial = diff_D.index(min(diff_D))
                E_trial = np.trapz(BaseShear[:idx_trial], DispCtrlNode[:idx_trial])
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
        plt.plot(DispCtrlNode, BaseShear, label='Analytical')
        plt.plot(depp, fepp, label='Idealised')
        plt.legend(frameon=False, loc='lower right')
        plt.grid(True)
        plt.xlabel('$u_{ctrl}$ [m]')
        plt.ylabel('$V_{base}$ [kN]')
        plt.title('%s pushover with control node: %d' % (loading_scheme, ctrlNode))
        ax = plt.gca()
        ax.text(0.57, 0.32, '$T^{*}$ = ' + "{:.2f}".format(float(T_star)) + ' sec', transform=ax.transAxes)
        fname = os.path.join(out_dir, 'PushOver Curve.svg')
        plt.savefig(fname, bbox_inches='tight')

    def plot_mphi(self, save=1, show=0):
        for pier_tag in self.EleIDsPier:
            sec_name = self.EleIDsPier[pier_tag]['section']
            P = -self.EleIDsPier[pier_tag]['AxialForce']
            fig, ax = plt.subplots()
            if 'Circular' in self.model[sec_name]['Type']:
                RF = self.EleIDsPier[pier_tag]['RF']
                MPhi = self.EleIDsPier[pier_tag]['MPhi']
                MPhi_bilin = self.EleIDsPier[pier_tag]['MPhi_bilin']

                ax.plot(MPhi[:, 1], MPhi[:, 0], label='Analytical')
                ax.plot(MPhi_bilin[:, 1], MPhi_bilin[:, 0], label='Idealised')
                ax.set_title(
                    'Moment-Curvature Analysis\nID: %s, P = %d kN, $I_{eff}=%.2fI_{gross}$' % (pier_tag, P, RF))
            ax.set_xlabel('Curvature [$m^{-1}$]')
            ax.set_ylabel('Moment [kN.m]')
            ax.grid(True)
            ax.legend(frameon=False)
            if show == 1:
                plt.show()
            if save == 1:
                fname = os.path.join(self.out_dir, str(pier_tag) + '_mca.svg')
                plt.savefig(fname, bbox_inches='tight')
            plt.close('all')


class main(_builder, _rendering, _analysis):

    def __init__(self, model, output_dir='Outputs', const_opt=0):
        """
        ----------------------------------------------------------------------------
        OBJECT INITIATION
        ----------------------------------------------------------------------------
        This method must be called to initialize the EzBridge object with user defined
        bridge model parameters.

        Parameters
        ----------
        model : dictionary
            dictionary containing model parameters
        output_dir    : str, optional. The default is 'Outputs'.
            the directory where any output file is stored.
        const_opt: int, optional. The default is 0.
            option to handle multiple-point constraints 0: via rigid links, 1: via rigid like elements)

        Returns
        -------

        """

        # PRINT BASIC PROGRAM INFO
        print(program_info())

        # Definition of constraints
        self.const_opt = const_opt

        # INITIALIZE THE CLASS OBJECTS USING INHERITANCE
        _builder.__init__(self)

        # MODEL INPUT
        self.model = model

        # CREATE DIRECTORIES
        self.out_dir = output_dir
        create_dir(self.out_dir)
        self.animation_dir = os.path.join(output_dir, 'Animation')
        self.nrha_dir = os.path.join(output_dir, 'NRHA')

        # DEFAULT ANIMATION PARAMETERS
        self.animate = 0
        self.Movie = 0
        self.FrameStep = 5
        self.scale = 50
        self.fps = 50

        # DEFAULT ANALYSIS PARAMETERS
        self.constraintType = 'Transformation'
        self.numbererType = 'RCM'
        self.systemType = 'UmfPack'
        self.alphaS = 1e14
        self.alphaM = 1e14

        # DEFAULT DAMPING OPTIONS
        self.damping_option = 3
        self.damping_modes = 1
        self.damping_xi = 0.02
        self.damping_periods = None

        global end_text
        # The following end_text is used for separation of some text outputs
        end_text = '-------------------------------------------------------------------------'

        # Plotting settings
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIG_SIZE = 24
        BIGGER_SIZE = 26

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
        plt.rcParams['mathtext.it']= 'Times New Roman:italic'
        plt.rcParams['mathtext.cal']= 'Times New Roman:italic'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams["mathtext.fontset"] ='custom'

    def generate_model(self, mphi=0):
        """
        ------------------
        GENERATE THE MODEL
        ------------------
        This method must be called to generate the structural model.
        It will create a model, perform gravity analysis to obtain pier axial loads,
        and then a series of moment-curvature analyses to obtain the moment-curvature behaviour of piers.

        Parameters
        ----------
        mphi : int, optional. The default is 0.
            flag (0 or 1) to obtain moment curvature response of piers.

        Returns
        -------

        """

        # Get bent axial forces
        self._get_axial_pier()
        # Moment-curvature analysis
        if self.model['General']['Pier Element Type'] == 1 or mphi == 1:
            self._get_pier_mphi()
        # Build the model
        self._build()
        # Save the object
        with open(os.path.join(self.out_dir, 'bridge_data.pkl'), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_analysis_parameters(self, constraintType, numbererType, systemType, alphaS=1e18, alphaM=1e18):
        """
        -----------------------------------------
        DEFINE NEW STRUCTURAL ANALYSIS PARAMETERS
        -----------------------------------------
        This method is used to set some new analysis parameters;
        namely, constraints, system, and numberer in OpenSees

        Parameters
        ----------
        constraintType: str
            Set the constraint type for analysis
        numbererType: str
            Set the numberer type for analysis (mapping between equation numbers and DOFs)
        systemType: str
            Set the system type for analysis (Linear Equation Solvers)
        alphaS:
            penalty factor on single point constraints, default is 1e14
        alphaM:
            penalty factor on multi-point constraints, default is 1e14

        Returns
        -------

        """

        # Set analysis parameters
        self.constraintType = constraintType
        self.numbererType = numbererType
        self.systemType = systemType
        self.alphaS = alphaS
        self.alphaM = alphaM

    def _assign_analysis_parameters(self):
        """
        --------------------------
        ASSIGN ANALYSIS PARAMETERS
        --------------------------
        This method is intended to be used internally only.

        Parameters
        ----------

        Returns
        -------

        """

        ops.wipeAnalysis()
        if self.constraintType == 'Penalty' or 'Lagrange':
            ops.constraints(self.constraintType, self.alphaS, self.alphaM)
        else:
            ops.constraints(self.constraintType)
        ops.numberer(self.numbererType)
        ops.system(self.systemType)

    def _get_axial_pier(self):
        """
        -----------------------
        GET AXIAL LOAD ON PIERS
        ------------------------
        This method is intended to be used internally only.

        Parameters
        ----------

        Returns
        -------

        """
        const_opt = self.const_opt
        abut_conf = self.model['General']['Abutments']
        abut_type = self.model[abut_conf]['Type']
        bent_ele_type = self.model['General']['Pier Element Type']
        self.model['General']['Pier Element Type'] = 0  # use ele type 0 (elastic)
        self.model[abut_conf]['Type'] = 'Fixed'  # use fixed abutment to get axial loads
        self.const_opt = 1
        self._build()  # build the model
        f_abut1, f_abut2, f_bent, f_total = self.do_gravity(pflag=0)  # perform gravity analysis

        for pier_tag in self.EleIDsPier:
            avg_force = 0
            for eleTag in self.EleIDsPier[pier_tag]['elements']:
                avg_force += ops.eleForce(eleTag, 3)
            self.EleIDsPier[pier_tag]['AxialForce'] = avg_force / len(self.EleIDsPier[pier_tag]['elements'])

        self.bearing_axial_forces = {}  # average bearing axial forces per joint
        for joint in self.EleIDsBearing:
            self.bearing_axial_forces[joint] = {'right': [], 'left': []}
            for bearing in self.EleIDsBearing[joint]:
                self.bearing_axial_forces[joint][bearing[6]].append(ops.eleForce(bearing[0], 3))

        self.AB1AxialForces = f_abut1[0, 2]
        self.AB2AxialForces = f_abut2[0, 2]

        self.model[abut_conf]['Type'] = abut_type
        self.model['General']['Pier Element Type'] = bent_ele_type
        self.const_opt = const_opt

    def do_gravity(self, pflag=1):
        """
        ----------------
        GRAVITY ANALYSIS
        ----------------
        This method is used to perform gravity analysis.

        Parameters
        ----------
        pflag : int, optional. The default is 1.
            flag (0 or 1) to print base reactions.

        Returns
        -------

        """

        # CREATE TIME SERIES
        self.EndPtag += 1
        self.EndTsTag += 1
        ops.timeSeries("Constant", self.EndTsTag)

        # CREATE A PLAIN LOAD PATTERN
        ops.pattern('Plain', self.EndPtag, self.EndTsTag)

        # APPLY DISTRIBUTED GRAVITY LOADS
        for load_entry in self.DistributedLoads:
            ops.eleLoad(*load_entry)

        # APPLY POINT GRAVITY LOADS
        for load_entry in self.PointLoads:
            ops.load(*load_entry)

        # SET ANALYSIS PARAMETERS
        self._assign_analysis_parameters()
        ops.test('NormDispIncr', 1e-8, 500)
        ops.algorithm('Newton')
        ng = 500
        ops.integrator('LoadControl', 1 / ng)
        ops.analysis('Static')

        # DO THE ANALYSIS
        ops.analyze(ng)

        # maintain constant gravity loads and reset time to zero
        ops.loadConst('-time', 0.0)

        # Get the reaction forces in vertical direction, this is for verification purposes
        ops.reactions('-dynamic', '-rayleigh')
        f_abut1 = np.zeros([1, 6])
        f_abut2 = np.zeros([1, 6])
        f_total = np.zeros([1, 6])

        for node in self.fixed_backfill[0]:
            f_abut1 += np.array(ops.nodeReaction(node))
        text1 = ('%4s   | %6s | %6s | %6s | %6s | %6s | %6s |' % ('A1',
                                                                  "{:.0f}".format(f_abut1[0, 0]),
                                                                  "{:.0f}".format(f_abut1[0, 1]),
                                                                  "{:.0f}".format(f_abut1[0, 2]),
                                                                  "{:.0f}".format(f_abut1[0, 3]),
                                                                  "{:.0f}".format(f_abut1[0, 4]),
                                                                  "{:.0f}".format(f_abut1[0, 5])))
        text2 = []
        f_bent = {}
        for i in self.fixed_bent:
            f_bent[i] = np.zeros([1, 6])
            for j in range(len(self.fixed_bent[i])):
                node = self.fixed_bent[i][j]
                f_pier = np.array([ops.nodeReaction(node)])
                text2.append('%6s | %6s | %6s | %6s | %6s | %6s | %6s |' %
                             ('B' + str(i) + '-P' + str(j + 1), "{:.0f}".format(f_pier[0, 0]),
                              "{:.0f}".format(f_pier[0, 1]),
                              "{:.0f}".format(f_pier[0, 2]), "{:.0f}".format(f_pier[0, 3]),
                              "{:.0f}".format(f_pier[0, 4]), "{:.0f}".format(f_pier[0, 5])))
                f_total += f_pier
            f_bent[i] += f_pier

        text2 = '\n'.join(text2)

        for node in self.fixed_backfill[1]:
            f_abut2 += np.array(ops.nodeReaction(node))
        text3 = ('%4s   | %6s | %6s | %6s | %6s | %6s | %6s |' % ('A2',
                                                                  "{:.0f}".format(f_abut2[0, 0]),
                                                                  "{:.0f}".format(f_abut2[0, 1]),
                                                                  "{:.0f}".format(f_abut2[0, 2]),
                                                                  "{:.0f}".format(f_abut2[0, 3]),
                                                                  "{:.0f}".format(f_abut2[0, 4]),
                                                                  "{:.0f}".format(f_abut2[0, 5])))

        f_total = f_total + f_abut1 + f_abut2

        text4 = ('%5s  | %6s | %6s | %6s | %6s | %6s | %6s |' % ('SUM',
                                                                 "{:.0f}".format(f_total[0, 0]),
                                                                 "{:.0f}".format(f_total[0, 1]),
                                                                 "{:.0f}".format(f_total[0, 2]),
                                                                 "{:.0f}".format(f_total[0, 3]),
                                                                 "{:.0f}".format(f_total[0, 4]),
                                                                 "{:.0f}".format(f_total[0, 5])))

        if pflag == 1:
            print("#########################################################################")
            print("                      Performing Gravity Analysis...                     ")
            print("#########################################################################")
            print('  Loc. | Fx[kN] | Fy[kN] | Fz[kN] | Mx[kN] | My[kN] | Mz[kN] |')
            print(text1)
            print(text2)
            print(text3)
            print(text4)
            print(end_text)

        if os.path.exists(os.path.join(self.out_dir, 'gravity_log.txt')):
            os.remove(os.path.join(self.out_dir, 'gravity_log.txt'))
        f = open(os.path.join(self.out_dir, 'gravity_log.txt'), "a")
        # f.write('\n')
        f.write('  Loc. | Fx[kN] | Fy[kN] | Fz[kN] | Mx[kN] | My[kN] | Mz[kN] |')
        f.write('\n')
        f.write(text1)
        f.write('\n')
        f.write(text2)
        f.write('\n')
        f.write(text3)
        f.write('\n')
        f.write(text4)
        f.write('\n')
        f.close()

        return f_abut1, f_abut2, f_bent, f_total  # total vertical reaction force --> total weight

    def do_modal(self, num_eigen=1, pflag=1):
        """
        --------------
        MODAL ANALYSIS
        --------------
        This method is used to perform modal analysis.
        It makes use of the built-in OpenSees function created by Massimo Petracca.

        Parameters
        ----------
        num_eigen : int, optional (The default is 1)
            Number of eigenvalues to calculate.
        pflag : int (1 or 0)
            flag to print modal analysis outputs on the screen

        Returns
        -------

        """
        print("#########################################################################")
        print("                         Performing Modal Analysis...                    ")
        print("#########################################################################")

        # compute the modal properties
        self._assign_analysis_parameters()
        eigenvalue = self._eigen(num_eigen, pflag)
        omega = eigenvalue ** 0.5
        T = 2 * np.pi / omega
        out_name = os.path.join(self.out_dir, "Modal_Properties.txt")
        args = ["-print", "-file", out_name, "-unorm"]
        if pflag == 0:
            args.remove("-print")
        ops.modalProperties(*args)
        # Read back stuff from the modal properties.txt it could be necessary
        with open(out_name, "r") as file:
            info = file.readlines()
            for i in range(len(info)):
                line = info[i]
                if '9. MODAL PARTICIPATION MASS RATIOS' in line:
                    idx = i
                    break
            mratios = info[idx + 4:idx + 4 + num_eigen]

            for i in range(len(info)):
                line = info[i]
                if '4. TOTAL FREE MASS OF THE STRUCTURE' in line:
                    idx = i
                    break
            self.M_total = np.array([float(val) for val in info[idx + 5].split()])

        mratio_array = np.ones((num_eigen, 6))
        for i in range(len(mratios)):
            mratio_array[i, :] = np.array([float(val) for val in mratios[i].split()[1:]])

        self.modal_properties = {'Modal Mass Ratios': mratio_array, 'Periods': T}

    def do_rsa(self, target_spectrum, num_modes, method='srss', xi=0.05, analysis_direction=2):
        """
        --------------------------
        RESPONSE SPECTRUM ANALYSIS
        --------------------------
        This method is used to perform response spectrum analysis.
        It makes use of the built-in OpenSees function created by Massimo Petracca.

        Parameters
        ----------
        target_spectrum : numpy.ndarray (Nx2).
            Target response spectrum to use.
            Periods are defined in first column, and acceleration values are defined in second column.
        num_modes : int
            Number of modes to be used for analysis.
        method : str, optional (The default is 'srss').
            Modal combination method to use.
            'cqc': Complete Quadratic Combination.
            'srss': Square Root of Sum of Squares.
        xi : float, optional (The default is 0.05).
            Damping for cqc modal combination.
        analysis_direction: int, optional (The default is 2).
            Considered analysis direction to perform the analysis.

        Returns
        -------
        eleForces: dictionary
            Dictionary containing element joint forces
        nodalDisps: dictionary
            Dictionary containing nodal displacements
        baseReactions: dictionary
            Dictionary containing base reaction forces
        bentDrifts: dictionary
            Dictionary containing absolute bent drifts
        """

        # Definition of cqc function
        # ------------------------------------------------------------------------
        def cqc(mu, eigenvalues, damping):
            """
            -----------------------
            MODAL COMBINATION - CQC
            -----------------------
            This method is used to carry out Complete Quadratic Combination (cqc) of modes
            It is intended to be used internally only.

            Parameters
            ----------
            mu : numpy.ndarray
                Analysis results of a node or an element for each mode
                (forces, displacements, deformations etc.)
            eigenvalues : list
                Eigenvalues
            damping : list
                Damping ratio for modes

            Returns
            -------
            u : numpy.ndarray
                Combined analysis results of a node or an element
                (forces, displacements, deformations etc.)
            """

            u = 0.0
            for idx1 in range(len(eigenvalues)):
                for idx2 in range(len(eigenvalues)):
                    di = damping[idx1]
                    dj = damping[idx2]
                    bij = eigenvalues[idx1] / eigenvalues[idx2]
                    rho = ((8.0 * ((di * dj) ** 0.5) * (di + bij * dj) * (bij ** (3.0 / 2.0))) /
                           ((1.0 - bij ** 2.0) ** 2.0 + 4.0 * di * dj * bij * (1.0 + bij ** 2.0) +
                            4.0 * (di ** 2.0 + dj ** 2.0) * bij ** 2.0))
                    u += mu[idx1] * mu[idx2] * rho
            u = u ** 0.5
            return u

        # Definition of srss function
        # ------------------------------------------------------------------------
        def srss(mu, modes):
            """
            ------------------------
            MODAL COMBINATION - SRSS
            ------------------------
            This method is used to carry out Square-Root-of-Sum-of-Squares (srss) combination of modes.
            It is intended to be used internally only.

            Parameters
            ----------
            mu : numpy.ndarray
                Analysis results of a node or an element for each mode
                (forces, displacements, deformations etc.)
            modes : int
                number of modes to combine (first modes)

            Returns
            -------
            u : numpy.ndarray
                Combined analysis results of a node or an element
                (forces, displacements, deformations etc.)
            """

            u = 0.0
            for idx in range(modes):
                u += mu[idx] ** 2
            u = u ** 0.5
            return u

        # Initialize variables to save information
        # ------------------------------------------------------------------------
        modal_forces = {}  # Initialize the dictionary for modal forces of elements
        for ele in ops.getEleTags():
            modal_forces[ele] = []

        modal_reactions = {}  # Initialize the dictionary for modal reaction forces of nodes
        modal_disps = {}  # Initialize the dictionary for modal displacement of nodes
        for node in ops.getNodeTags():
            modal_disps[node] = []
            modal_reactions[node] = []

        modal_bent_drifts = {}  # Initialize the dictionary for modal bent drifts
        for pier_tag in self.EleIDsPier:
            for ele in self.EleIDsPier[pier_tag]['elements']:
                modal_bent_drifts[ele] = []

        # Definition of target spectrum for RSA
        # ------------------------------------------------------------------------
        periods = target_spectrum[:, 0]
        sa = target_spectrum[:, 1]

        # Time series tag
        self.EndTsTag += 1

        # the response spectrum function
        ops.timeSeries("Path", self.EndTsTag, "-time", *periods, "-values", *sa, "-factor", 9.81)

        # Set analysis parameters for response spectrum analysis
        # ------------------------------------------------------------------------
        # Wipe any previous analysis object
        ops.wipeAnalysis()

        # Convergence Test -- determines when convergence has been achieved.
        ops.test('NormUnbalance', 0.0001, 10)

        # SolutionAlgorithm -- determines the sequence of steps taken to solve the non-linear equation at the current
        # time step
        ops.algorithm("Linear")

        # DOF_Numberer -- determines the mapping between equation numbers and degrees-of-freedom
        ops.numberer('RCM')

        # SystemOfEqn/Solver -- within the solution algorithm, it specifies how to store and solve the system of
        # equations in the analysis
        ops.system('UmfPack')

        # Constraints handler: determines how the constraint equations are enforced in the analysis -- how it handles
        # the boundary conditions/imposed displacements
        ops.constraints('Transformation')

        # Integrator -- determines the predictive step for time t+dt
        ops.integrator('LoadControl', 0.0)

        # AnalysisType -- defines what type of analysis is to be performed ('Static', 'Transient' etc.)
        ops.analysis('Static')

        # Compute the modal properties
        # ------------------------------------------------------------------------
        lambdas = ops.eigen('-genBandArpack', num_modes)  # eigenvalue analysis
        ops.modalProperties("-unorm")  # modal properties
        dmp = [xi] * num_modes  # currently, we use the same damping for each mode (xi)

        # Perform the analysis
        # ------------------------------------------------------------------------
        # Perform a response spectrum analysis mode-by-mode, and grab results during the loop

        for i in range(len(lambdas)):
            ops.responseSpectrum(self.EndTsTag, analysis_direction, '-mode', i + 1)
            for ele in ops.getEleTags():
                forces = np.array(ops.eleResponse(ele, 'force'))
                modal_forces[ele].append(forces)
            for node in ops.getNodeTags():
                disps = np.array(ops.nodeDisp(node))
                modal_disps[node].append(disps)
                ops.reactions('-dynamic', '-rayleigh')  # Must call this command before using nodeReaction() command.
                reactions = np.array(ops.nodeReaction(node))
                modal_reactions[node].append(reactions)
            for ele in modal_bent_drifts:
                node_i, node_j = ops.eleNodes(ele)
                drifts = modal_disps[node_i][i][0:2] - modal_disps[node_j][i][0:2]
                drift = (drifts[0] ** 2 + drifts[1] ** 2) ** 0.5
                modal_bent_drifts[ele].append(drift)

        # Perform modal combinations
        # ------------------------------------------------------------------------
        ele_forces = {}
        for ele in ops.getEleTags():
            if method == 'cqc':
                forces = cqc(modal_forces[ele], lambdas, dmp)
            elif method == 'srss':
                forces = srss(modal_forces[ele], num_modes)
            ele_forces[ele] = forces

        bent_drifts = {}
        for ele in modal_bent_drifts:
            if method == 'cqc':
                drifts = cqc(modal_bent_drifts[ele], lambdas, dmp)
            elif method == 'srss':
                drifts = srss(modal_bent_drifts[ele], num_modes)
            bent_drifts[ele] = drifts

        nodal_disps = {}
        nodal_reactions = {}
        for node in ops.getNodeTags():
            if method == 'cqc':
                disps = cqc(modal_disps[node], lambdas, dmp)
                reactions = cqc(modal_reactions[node], lambdas, dmp)
            elif method == 'srss':
                disps = srss(modal_disps[node], num_modes)
                reactions = srss(modal_reactions[node], num_modes)
            nodal_disps[node] = disps
            nodal_reactions[node] = reactions

        base_reactions = 0
        for node in nodal_reactions.keys():
            forces = nodal_reactions[node]
            base_reactions += forces

        return ele_forces, nodal_disps, base_reactions, bent_drifts

    def do_nspa(self, scheme='MPP', max_displacement=1.0, ctrl_node=None, ctrl_dof=2, num_steps=1000, pflag=1,
                bilinearization='EC', Modes=None):
        """
        -----------------------------------------
        NONLINEAR STATIC PUSHOVER ANALYSIS (NSPA)
        -----------------------------------------
        The method is used to perform NSPA using various loading schemes.
        The loads are applied at the deck level only. This may not be an adequate approach in case of tall piers.

        Parameters
        ----------
        scheme : str, optional (The default is 'MPP')
            The loading scheme to use for _nspa ('MPP', 'FMP', 'UNI')
            'MPP': Mass proportional loading
            'FMP': Fundamental-mode proportional loading
            'UNI': Uniformly distributed loading
        max_displacement : float, optional (The default is 1)
            Displacement value to which control node is pushed.
        ctrl_dof : int, optional (The default is 2)
            Control degrees of freedom
        ctrl_node : int, optional (The default is None)
            Control node to carry out displacement controlled nonlinear static pushover analysis.
            If none, control node will be determined as the node which has the greatest modal displacement in ctrl_dof
            based on the fundamental mode
        num_steps: int, optional (The default is 2000)
            Number of steps used to carry out NSPA
        pflag : int (0, 1)
            Output information on the screen
        bilinearization: str, optional (The default is 'EC')
            Approach to follow for bi-linear idealization of pushover curve ('EC','ASCE','NTC')
            'EC': Eurocode 8 approach
            'ASCE': ASCE 7-10 approach
            'NTC': NTC 2018 approach
        Modes: int, optional (The default is None)
            Number of nodes considered to carry out modal analysis. Modal analysis is carried out
            to find out the fundamental mode in considered control DOF required for FMP case.

        Returns
        -------

        """

        print("#########################################################################")
        print("           Performing Nonlinear Static Pushover Analysis (NSPA)...       ")
        print("#########################################################################")

        # Rebuild the model
        self._build()

        #  Perform gravity analysis
        self.do_gravity(pflag=0)

        # Set the push nodes for analysis
        push_nodes = []  # Nodes to push
        for span in self.D1Nodes:  # deck nodes
            push_nodes.extend(self.D1Nodes[span])
            push_nodes.extend(self.D2Nodes[span])
        push_nodes = list(set(push_nodes))  # Get unique nodes
        push_nodes.sort()
        # push_nodes = []  # Nodes to push
        # for i in range(len(self.S1Nodes)):
        #     push_nodes.append(self.S1Nodes[i])
        #     push_nodes.append(self.S2Nodes[i])
        # push_nodes = list(set(push_nodes))  # Get unique nodes
        # Get Mass matrix, and masses corresponding to each node
        # if numberer is plain DOFs are in order with ops.getNodeTags()
        ops.wipeAnalysis()
        ops.system('FullGeneral')
        ops.analysis('Transient')
        ops.numberer('Plain')
        ops.algorithm('Linear')
        # Extract the Mass Matrix
        ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
        ops.analyze(1, 0.0)
        # Number of equations in the model
        N = ops.systemSize()  # Has to be done after analyze
        M_matrix = ops.printA('-ret')  # Or use ops.printA('-file','M.out')
        M_matrix = np.array(M_matrix)  # Convert the list to an array
        M_matrix.shape = (N, N)  # Make the array an NxN matrix
        M_diagonal = M_matrix.diagonal()  # Get the uncoupled nodal masses per DOF (no coupling)
        print(end_text)
        print('Extracted the mass matrix, ignore the previous warnings...')
        nodes_dof_mass = {}  # Save here the nodes, and the masses in their associated dofs from global mass matrix
        DOFs = []
        k = 0  # index for global DOFs
        for node in ops.getNodeTags():
            nodes_dof_mass[node] = {}
            for i in range(6):  # Loop through number of DOFs/node
                temp = ops.nodeDOFs(node)[i]  # Get the global DOF id (-1 if restrained)
                if temp not in DOFs and temp >= 0:  # Check if the DOF is not restrained
                    DOFs.append(temp)
                    nodes_dof_mass[node][i + 1] = M_diagonal[k]
                    k += 1
                else:  # if the dof is constrained, assign 0
                    nodes_dof_mass[node][i + 1] = 0

        # Perform modal analysis to identify the first mode
        if not Modes:
            Modes = int(sum(M_diagonal > 0) / 3)
        self.do_modal(num_eigen=Modes, pflag=0)
        Ms = self.modal_properties['Modal Mass Ratios'][:, ctrl_dof - 1]
        inds = Ms.argsort()[::-1]
        Mode1 = int(inds[0] + 1)

        masses = np.ones(len(push_nodes))
        modal_disp = []
        h1loads = np.ones(len(push_nodes))
        h2loads = np.ones(len(push_nodes))
        for i, node in enumerate(push_nodes):
            masses[i] = nodes_dof_mass[node][ctrl_dof]
            modal_disp.append(abs(ops.nodeEigenvector(node, Mode1, ctrl_dof)))
            h1loads[i] = nodes_dof_mass[node][1] * ops.nodeEigenvector(node, Mode1, 1)
            h2loads[i] = nodes_dof_mass[node][2] * ops.nodeEigenvector(node, Mode1, 2)

        if not ctrl_node:
            ctrl_node = push_nodes[modal_disp.index(max(modal_disp))]

        # Initialize the force matrix
        forces = np.zeros((len(push_nodes), 6))

        # PUSHOVER ANALYSIS WITH UNIFORM LOAD DISTRIBUTION
        if scheme == 'UNI':
            # Normalize forces with respect to the mass distribution
            forces[:, ctrl_dof - 1] = np.ones(len(push_nodes)) / len(push_nodes)

        # MASS PROPORTIONAL PUSHOVER ANALYSIS
        if scheme == 'MPP':
            # Normalize forces with respect to the mass distribution
            forces[:, ctrl_dof - 1] = masses / np.sum(masses)

        # FIRST MODE PROPORTIONAL PUSHOVER ANALYSIS
        elif scheme == 'FMP':
            # Normalize the loads. This will make the load factor equal to the total base shear force.
            hloads = (sum(h1loads) ** 2 + sum(h2loads) ** 2) ** 0.5
            # Reverse the push direction against negative loading
            if ctrl_dof == 2 and sum(h2loads) < 0:
                h2loads = - h2loads
                h1loads = - h1loads
            elif ctrl_dof == 1 and sum(h1loads) < 0:
                h2loads = - h2loads
                h1loads = - h1loads

            forces[:, 0] = h1loads / hloads
            forces[:, 1] = h2loads / hloads

        # Define analysis parameters
        self._assign_analysis_parameters()

        # CREATE TIME SERIES AND APPLY LOADS
        self.EndPtag += 1
        self.EndTsTag += 1
        ops.timeSeries('Linear', self.EndTsTag)  # Define the timeSeries for the load pattern
        ops.pattern('Plain', self.EndPtag, self.EndTsTag)  # Define load pattern -- generalized
        for i, node_tag in enumerate(push_nodes):
            load_values = forces[i, :].tolist()
            ops.load(node_tag, *load_values)

        # set animation recorders
        if self.animate == 1:
            create_dir(self.animation_dir)
            self._animation_recorders()

        # Perform npsa
        [base_shear, ctrl_node_displacement] = self._nspa(max_displacement, ctrl_node, ctrl_dof, num_steps, pflag)

        # Plot and save pushover curve
        np.savetxt(os.path.join(self.out_dir, 'NSPA_Summary.txt'),
                   np.column_stack((ctrl_node_displacement, base_shear)))
        self._bilin_pushover(self.out_dir, base_shear, ctrl_node_displacement, self.M_total[1], bilinearization,
                             scheme, ctrl_node)

        # Animate
        if self.animate == 1:
            ops.wipe()
            self._animate_nspa(scheme, base_shear, ctrl_node_displacement, ctrl_node)

    def set_damping(self, option=3, modes=1, xi=0.02, damping_periods=None):
        """
        --------------------------
        DEFINE NEW DAMPING OPTIONS
        --------------------------
        This method is used to set the new damping options.

        Parameters
        ----------
        option : int, optional (The default is 3).
            The parameter defines the damping type being used (1,2,3,4,5,6).
            1: Stiffness proportional damping with current stiffness matrix
            2: Stiffness proportional damping with initial stiffness matrix
            3: Stiffness proportional damping with last committed stiffness matrix
            4: Rayleigh damping with current stiffness matrix
            5: Rayleigh damping with initial stiffness matrix
            6: Rayleigh damping with last committed stiffness matrix
        modes : int or list, optional (The default is 1).
            Considered mode (int) or modes (list) considered for stiffness proportional (1,2,3)
            or rayleigh damping (4,5,6), respectively.
        xi : float or list, optional (The default is 0.02).
            Damping ratio (float) or ratios (list) used for stiffness proportional (1,2,3) or
            rayleigh damping (4,5,6), respectively.
        damping_periods : float or list, optional (The default is None).
            User may define the periods to use directly instead of using periods corresponding to the specified modes.
            If this parameter is not None, the damping will be defined based on damping_periods
            rather than periods obtained from "modes" parameter.

        Returns
        -------

        """

        self.damping_option = option
        self.damping_modes = modes
        self.damping_xi = xi
        self.damping_periods = damping_periods

    def _get_damping_parameters(self):
        """
         ----------------------------
         DETERMINE DAMPING PARAMETERS
         ----------------------------
         This method is intended to be used internally only.

         Parameters
         ----------

         Returns
         -------

         """

        # flags to determine which stiffness to use
        k_comm_flag = 0.0
        k_init_flag = 0.0
        k_curr_flag = 0.0

        # Use current stiffness matrix to calculate damping matrix
        if self.damping_option in [1, 4]:
            k_curr_flag = 1.0
        # Use initial stiffness matrix to calculate damping matrix
        elif self.damping_option in [2, 5]:
            k_init_flag = 1.0
        # Use initial stiffness matrix to calculate damping matrix
        elif self.damping_option in [3, 6]:
            k_comm_flag = 1.0

        # Stiffness proportional damping
        if isinstance(self.damping_periods, float) or self.damping_option in [1, 2, 3]:
            if self.damping_periods:
                wi = 2 * np.pi / self.damping_periods
            else:
                wi = 2 * np.pi / self.modal_properties['Periods'][self.damping_modes - 1]
            a0 = 0
            a1 = 2.0 * self.damping_xi / wi

        # Rayleigh damping
        elif isinstance(self.damping_periods, list) or self.damping_option in [4, 5, 6]:
            if self.damping_periods:
                wi = 2 * np.pi / self.damping_periods[0]
                wj = 2 * np.pi / self.damping_periods[1]

            elif self.damping_option in [4, 5, 6]:
                # Compute the Rayleigh damping coefficients
                wi = self.modal_properties['Periods'][self.damping_modes[0] - 1]
                wj = self.modal_properties['Periods'][self.damping_modes[1] - 1]

            xii = self.damping_xi[0]  # damping ratio of ith mode
            xij = self.damping_xi[1]  # damping ratio of jth mode
            A = np.array([[1 / wi, wi], [1 / wj, wj]])
            b = np.array([xii, xij])
            a0, a1 = np.linalg.solve(A, 2 * b)

        else:
            raise ValueError('You did not choose correct damping option!')

        # Assign damping to the model
        alphaM = a0
        betaK_curr = a1 * k_curr_flag
        betaK_init = a1 * k_init_flag
        betaK_comm = a1 * k_comm_flag
        return alphaM, betaK_curr, betaK_init, betaK_comm

    def _create_excitation_pattern(self, gm_filenames, gm_components, gm_dir, gm_dt, gm_angle, gm_sf=None):
        """
         ---------------------------------------------
         CREATE EXCITATION LOAD PATTERN TO USE IN NRHA
         ---------------------------------------------
         This method is intended to be used internally only.

         Parameters
         ----------
         gm_filenames : list
            Ground motion filenames corresponding to each component
         gm_components : list
            Ground motion components to use 1 and 2 are the horizontal components, 3 is the vertical component
         gm_dir : str
            Directory where ground motion files are located
         gm_dt : float
            Time step for ground motion time histories
         gm_angle : float
            Incidence angle to use while applying horizontal ground motion components
         gm_sf : list
            Scaling factors corresponding to each ground motion components

         Returns
         -------
        gm_dur : float
            Duration of the ground motion
        gm_dt : float
            Time step of the ground motion
         """

        if gm_sf is None:
            gm_sf = [1.0, 1.0, 1.0]

        npts_h1 = 0
        npts_h2 = 0
        npts_v = 0
        for i in range(len(gm_components)):
            gm_path = os.path.join(gm_dir, gm_filenames[i])  # Assuming GM is in PEER format

            if gm_components[i] == 1:
                if gm_path[-4:] == '.AT2':
                    gm_dt, _, _, _, h1_tmp = read_nga_record(gm_path)
                else:
                    h1_tmp = np.loadtxt(gm_path)
                npts_h1 = len(h1_tmp)
                h1_fact = gm_sf[i]
                h1_tmp = h1_tmp * h1_fact

            if gm_components[i] == 2:
                if gm_path[-4:] == '.AT2':
                    gm_dt, _, _, _, h2_tmp = read_nga_record(gm_path)
                else:
                    h2_tmp = np.loadtxt(gm_path)
                npts_h2 = len(h2_tmp)
                h2_fact = gm_sf[i]
                h2_tmp = h2_tmp * h2_fact

            if gm_components[i] == 3:
                if gm_path[-4:] == '.AT2':
                    gm_dt, _, _, _, v_tmp = read_nga_record(gm_path)
                else:
                    v_tmp = np.loadtxt(gm_path)
                npts_v = len(v_tmp)
                v_fact = gm_sf[i]
                v_tmp = v_tmp * v_fact

        # Assign zero if gm component is not used
        npts = max(npts_h1, npts_h2, npts_v)
        gm_dur = gm_dt * npts
        gm_h1 = np.zeros(npts)
        gm_h2 = np.zeros(npts)
        gm_v = np.zeros(npts)

        try:
            gm_h1[:npts_h1] = h1_tmp
        except NameError:
            pass
        try:
            gm_h2[:npts_h2] = h2_tmp
        except NameError:
            pass
        try:
            gm_v[:npts_v] = v_tmp
        except NameError:
            pass

        # Modify the ground motions according to the angle of incidence
        gm_h1.shape = (1, npts)
        gm_h2.shape = (1, npts)
        gm_v.shape = (1, npts)
        gm_mat = np.vstack((gm_h1, gm_h2))
        theta_mat = np.array([[np.cos(gm_angle * np.pi / 180), np.sin(gm_angle * np.pi / 180)],
                              [-np.sin(gm_angle * np.pi / 180), np.cos(gm_angle * np.pi / 180)]])
        gm_mat = np.vstack((theta_mat @ gm_mat, gm_v))

        # TODO: modify this part for multi support excitation
        sf = g + 0.0  # if signal is acceleration, the records are always defined in g

        for i in range(1, 4):
            # Setting time series
            self.EndTsTag += 1
            self.EndPtag += 1
            gm_i = gm_mat[i - 1, :]
            ops.timeSeries('Path', self.EndTsTag, '-dt', gm_dt, '-values', *list(gm_i), '-factor', sf)
            ops.pattern('UniformExcitation', self.EndPtag, i, '-accel', self.EndTsTag)

        return gm_dur, gm_dt

    def do_nrha(self, gm_dir, gm_filenames, gm_components, gm_dt=None, gm_sf=None, gm_angle=0, dc=10, t_free=0, dt_factor=1):
        """
        ------------------------------------------
        NONLINEAR RESPONSE HISTORY ANALYSIS (NRHA)
        ------------------------------------------
        The method is used to perform NRHA of the structural model.
        The signals are applied as uniform excitation at the moment.

        Parameters
        ----------
        gm_dir : str
            The directory where files to use in nrha are located
        gm_filenames : list
            Ground motion filenames corresponding to each component
        gm_components : list
            Ground motion components to use 1 and 2 are the horizontal components, 3 is the vertical component
        gm_dt : float, optional (The default is None)
            Time step for ground motion time histories
            This parameters is not required if time history files are in PEER format; .AT2 extension
        gm_sf : list, optional (The default is [1.0, 1.0, 1.0]).
            Scaling factors corresponding to each ground motion components
        gm_angle : float, optional (The default is 0)
            Incidence angle to use while applying horizontal ground motion components
        dc : float, optional (The default is 10)
            Drift capacity (%) to define local collapse in NRHA. The analysis stops if pier reaches this drift level
        t_free : float, optional (The default is 0)
            Additional free vibration time
        dt_factor : float, optional (The default is 1)
            Analysis time step is determined as a (dt_factor * gm_dt). For example,
            if dt_factor is 1.0, analysis time step is equal to ground motion time step

        Returns
        -------

        """

        print("#########################################################################")
        print("          Performing Nonlinear Response History Analysis (NRHA)...       ")
        print("#########################################################################")

        # Rebuild the model
        self._build()
        #  Perform gravity analysis
        self.do_gravity(pflag=0)
        # Get damping damping parameters
        alphaM, betaK_curr, betaK_init, betaK_comm = self._get_damping_parameters()
        # Define analysis parameters
        self._assign_analysis_parameters()
        # Assign rayleigh damping
        ops.rayleigh(alphaM, betaK_curr, betaK_init, betaK_comm)

        if gm_filenames is None:
            raise ValueError('Assign ground motion files to use: GMs...')

        # TODO: modify this part for multi support excitation
        # Create the excitation load patterns
        gm_dur, gm_dt = self._create_excitation_pattern(gm_filenames, gm_components, gm_dir, gm_dt, gm_angle, gm_sf)

        # Get analysis duration and time step
        t_final = gm_dur + t_free
        dt_analysis = gm_dt * dt_factor

        # Set recorders to animate the response
        if self.animate == 1:
            create_dir(self.animation_dir)
            self._animation_recorders()

        # Perform structural analysis
        edps_gm, c_index, anlys = self._nrha(dt_analysis, t_final, dc, pflag=1)

        # Create a log file for analysis
        f = open(os.path.join(self.out_dir, "nrha_log.txt"), "a")

        # Print the information on screen
        print(' Pier  | MDrftR [%] | MuDispZ | MuDispY | MuCurvZ | MuCurvY |')
        f.write(' Pier  | MDrftR [%] | MuDispZ | MuDispY | MuCurvZ | MuCurvY |')
        f.write('\n')
        idx = 0
        for i in self.fixed_bent:
            for j in range(len(self.fixed_bent[i])):
                print('%6s | %7s    | %6s  | %6s  | %6s  | %6s  |' %
                      ('B' + str(i) + '/P' + str(j + 1),
                       "{:.3f}".format(edps_gm['drift_ratio'][idx] * 100),
                       "{:.3f}".format(edps_gm['mu_disp_z'][idx]),
                       "{:.3f}".format(edps_gm['mu_disp_y'][idx]),
                       "{:.3f}".format(edps_gm['mu_curv_z'][idx]),
                       "{:.3f}".format(edps_gm['mu_curv_y'][idx])
                       ))
                f.write('%6s | %7s    | %6s  | %6s  | %6s  | %6s  |' %
                        ('B' + str(i) + '/P' + str(j + 1),
                         "{:.3f}".format(edps_gm['drift_ratio'][idx] * 100),
                         "{:.3f}".format(edps_gm['mu_disp_z'][idx]),
                         "{:.3f}".format(edps_gm['mu_disp_y'][idx]),
                         "{:.3f}".format(edps_gm['mu_curv_z'][idx]),
                         "{:.3f}".format(edps_gm['mu_curv_y'][idx])
                         ))
                f.write('\n')
                idx += 1

        print('Abutment | MDispL | MDispT |')
        f.write('\n')
        f.write('Abutment | MDispL | MDispT |')
        f.write('\n')

        for i in range(2):
            print('   %2s    | %6s | %6s |' %
                  ('A' + str(i + 1),
                   "{:.3f}".format(edps_gm['abut_disp_long'][i]),
                   "{:.3f}".format(edps_gm['abut_disp_transv'][i]),
                   ))

            f.write('   %2s    | %6s | %6s |' %
                    ('A' + str(i + 1),
                     "{:.3f}".format(edps_gm['abut_disp_long'][i]),
                     "{:.3f}".format(edps_gm['abut_disp_transv'][i]),
                     ))
            f.write('\n')

        print('  Bearing  |  MDispL |  MDispT |')
        f.write('\n')
        f.write('  Bearing  |  MDispL |  MDispT |')
        f.write('\n')
        for joint in self.EleIDsBearing:
            for i, _ in enumerate(self.EleIDsBearing[joint]):
                print('  %6s   |  %4s  |  %4s  |' %
                      ('J' + str(joint) + '/B' + str(i + 1),
                       "{:.3f}".format(edps_gm['bearing_disp_long'][joint][i]),
                       "{:.3f}".format(edps_gm['bearing_disp_transv'][joint][i]),
                       ))
                f.write('  %6s   |  %4s  |  %4s  |' %
                        ('J' + str(joint) + '/B' + str(i + 1),
                         "{:.3f}".format(edps_gm['bearing_disp_long'][joint][i]),
                         "{:.3f}".format(edps_gm['bearing_disp_transv'][joint][i]),
                         ))
                f.write('\n')
        f.write('\n')
        f.write(anlys)
        f.close()
        # Animate the response
        if self.animate == 1:
            ops.wipe()
            self._animate_nrha()

    def do_msa(self, msa_dir, gm_components, dt_file, h1_names_file=None, h2_names_file=None,
               v_names_file=None, gm_angle=0, dc=10, t_free=0):
        """
        ------------------------------
        MULTIPLE-STRIPE ANALYSIS (MSA)
        ------------------------------
        The method is used to perform MSA.
        The signals are applied as uniform excitation at the moment.

        Parameters
        ----------
        msa_dir : str
            Name of the folder in which sets of the ground motion records to be used are located.
        gm_components : list
            Ground motion components to use 1 and 2 are the horizontal components, 3 is the vertical component
        dt_file : str
            Name of the file in which time step for ground motion time histories are given
        h1_names_file : str
            Name of the file in which file names of first horizontal ground motion record component (1) to use
            in each record set are located.
        h2_names_file : str
            Name of the file in which file names of first second ground motion record component (2) to use
            in each record set are located.
        v_names_file : str
            Name of the file in which file names of vertical ground motion record component (3) to use
            in each record set are located.
        gm_angle : float, optional (The default is 0)
            Incidence angle to use while applying horizontal ground motion components
        dc : float, optional (The default is 10)
            Drift capacity (%) to define local collapse in NRHA. The analysis stops if pier reaches this drift level
        t_free : float, optional (The default is 0)
            Additional free vibration time

        Returns
        -------

        """

        # Rebuild the model
        self._build()
        #  Perform gravity analysis
        self.do_gravity(pflag=0)
        # Get damping damping parameters
        alphaM, betaK_curr, betaK_init, betaK_comm = self._get_damping_parameters()

        # Create output folder
        msa_out_dir = os.path.join(self.out_dir, 'MSA')
        create_dir(msa_out_dir)

        print("#########################################################################")
        print("             Performing Multiple Stripes Analysis (MSA)...               ")
        print("#########################################################################")

        # Read record sets
        gm_sets = []
        for file in os.listdir(msa_dir):
            if not file.startswith('.'):
                gm_sets.append(file)

        for gm_set in gm_sets:  # loop for each ground motion set
            gm_dir = os.path.join(msa_dir, gm_set)
            out_dir = os.path.join(msa_out_dir, gm_set)
            logfile_path = os.path.join(out_dir, 'log.txt')
            create_dir(out_dir)
            # create a log file for each set
            with open(logfile_path, 'w') as log:
                log.write(program_info())

                dt_path = os.path.join(gm_dir, dt_file)
                dts = np.loadtxt(dt_path)
                try:
                    num_gms = len(dts)
                except TypeError:  # If a single record exists in gm set
                    num_gms = 1
                    dts = [float(dts)]
                # Set up the error log to write to
                error_log = ["List of warning and errors encountered:"]

                # Initialize the engineering demand parameters to calculate
                self.edps_msa = {
                    # Check convergence of the results
                    'convergence_index': np.zeros((num_gms, 1)),
                    # Maximum Pier Drift Ratio
                    'drift_ratio': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Curvature Ductility (-z)
                    'mu_curv_z': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Curvature Ductility (-y)
                    'mu_curv_y': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Curvature Ductility (srss)
                    'mu_curv': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Displacement Ductility (-z)
                    'mu_disp_z': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Displacement Ductility (-y)
                    'mu_disp_y': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Displacement Ductility (srss)
                    'mu_disp': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Shear Force (-z)
                    'Vz': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Shear Force (-y)
                    'Vy': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Pier Shear Force (srss)
                    'V': np.zeros((num_gms, len(self.EleIDsPier))),
                    # Maximum Abutment Displacements in Longitudinal Direction
                    'abut_disp_long': np.zeros((num_gms, 2)),
                    # Maximum Abutment Displacements in Active Direction (long)
                    'abut_disp_active': np.zeros((num_gms, 2)),
                    # Maximum Abutment Displacements in Passive Direction (long)
                    'abut_disp_passive': np.zeros((num_gms, 2)),
                    # Maximum Abutment Displacements in Transverse Direction
                    'abut_disp_transv': np.zeros((num_gms, 2)),
                    # Maximum Abutment Displacements (srss)
                    'abut_disp': np.zeros((num_gms, 2)),
                    # Maximum Bearing Displacements in Longitudinal Direction
                    'bearing_disp_long': {joint: np.zeros((num_gms, len(self.EleIDsBearing[joint]))) for joint in
                                          self.EleIDsBearing},
                    # Maximum Bearing Displacements in Transverse Direction
                    'bearing_disp_transv': {joint: np.zeros((num_gms, len(self.EleIDsBearing[joint]))) for joint in
                                            self.EleIDsBearing},
                    # Maximum Bearing Displacements (srss)
                    'bearing_disp': {joint: np.zeros((num_gms, len(self.EleIDsBearing[joint]))) for joint in
                                     self.EleIDsBearing},
                    # Unseating
                    'unseating_disp_long': {joint: np.zeros((num_gms, len(self.EleIDsBearing[joint]))) for joint in
                                            self.EleIDsBearing}
                }

                # Open and load the ground motions
                gm_uniform = []
                if 1 in gm_components:
                    names_h1_path = os.path.join(gm_dir, h1_names_file)
                    with open(names_h1_path) as input_file:
                        gm_h1_names = [line.rstrip() for line in input_file]
                    gm_uniform.append(gm_h1_names)
                if 2 in gm_components:
                    names_h2_path = os.path.join(gm_dir, h2_names_file)
                    with open(names_h2_path) as input_file:
                        gm_h2_names = [line.rstrip() for line in input_file]
                    gm_uniform.append(gm_h2_names)
                if 3 in gm_components:
                    names_v_path = os.path.join(gm_dir, v_names_file)
                    with open(names_v_path) as input_file:
                        gm_v_names = [line.rstrip() for line in input_file]
                    gm_uniform.append(gm_v_names)

                # Loop through each ground motion
                for gm_idx in np.arange(num_gms):
                    # Get the ground motion
                    gm_dt = dts[gm_idx]

                    # Get filenames for components being used
                    gm_filenames = [gm_names[gm_idx] for gm_names in gm_uniform]

                    # Record information
                    gm_text = 'GM set: %s, GM no: %d' % (gm_set, gm_idx + 1)
                    print('Running ' + gm_text + '...')

                    # Try changing time step of analysis if the analysis does not converge
                    dt_values = [0.01, 0.005, 0.001]

                    # Get starting time of the analysis
                    t0 = get_current_time()

                    two_horizontal_components_flag = all(x in gm_components for x in [1, 2])   # if this is true run the case again by switching the gm components
                    # Check convergence, and reduce analysis time step if required.
                    c_index1 = -1  # index to check convergence
                    idx = 0  # counter for dt_factors
                    while c_index1 == -1 and idx < len(dt_values):
                        # Lets try to reduce the time step if convergence is not satisfied, and re-run the analysis
                        if idx == 0:
                            pass
                        else:
                            error_log.append('While running ' + gm_text + f' GM components: {gm_components}' +
                                             ' analysis failed to converge, reducing the analysis time step...')

                        # Rebuild the model
                        self._build()
                        #  Perform gravity analysis
                        self.do_gravity(pflag=0)
                        # Define analysis parameters
                        self._assign_analysis_parameters()
                        # Assign rayleigh damping
                        ops.rayleigh(alphaM, betaK_curr, betaK_init, betaK_comm)
                        # TODO: modify this part for multi support excitation
                        # Create the excitation load patterns
                        gm_dur, gm_dt = self._create_excitation_pattern(gm_filenames, gm_components, gm_dir, gm_dt,
                                                                        gm_angle)
                        # Get analysis duration and time step
                        t_final = gm_dur + t_free
                        dt_analysis = dt_values[idx]
                        idx += 1
                        # Perform structural analysis
                        edp_gm1, c_index1, anlys1 = self._nrha(dt_analysis, t_final, dc, pflag=1)

                    # Switch the components to run
                    if two_horizontal_components_flag:
                        gm_components2 = gm_components.copy()
                        idx1, idx2 = gm_components.index(1), gm_components.index(2)
                        gm_components2[idx1], gm_components2[idx2] = gm_components[idx2], gm_components[idx1]

                        # Check convergence, and reduce analysis time step if required.
                        c_index2 = -1  # index to check convergence
                        idx = 0  # counter for dt_factors
                        while c_index2 == -1 and idx < len(dt_values):
                            # Lets try to reduce the time step if convergence is not satisfied, and re-run the analysis
                            if idx == 0:
                                pass
                            else:
                                error_log.append('While running ' + gm_text + f' GM components: {gm_components2}' +
                                                 ' analysis failed to converge, reducing the analysis time step...')

                            # Rebuild the model
                            self._build()
                            #  Perform gravity analysis
                            self.do_gravity(pflag=0)
                            # Define analysis parameters
                            self._assign_analysis_parameters()
                            # Assign rayleigh damping
                            ops.rayleigh(alphaM, betaK_curr, betaK_init, betaK_comm)
                            # TODO: modify this part for multi support excitation
                            # Create the excitation load patterns
                            gm_dur, gm_dt = self._create_excitation_pattern(gm_filenames, gm_components2, gm_dir, gm_dt,
                                                                            gm_angle)
                            # Get analysis duration and time step
                            t_final = gm_dur + t_free
                            dt_analysis = dt_values[idx]
                            idx += 1
                            # Perform structural analysis
                            edp_gm2, c_index2, anlys2 = self._nrha(dt_analysis, t_final, dc, pflag=1)

                        for key in self.edps_msa:
                            if key == 'convergence_index':
                                self.edps_msa[key][gm_idx, 0] = min(c_index1, c_index2)
                            elif key in ['bearing_disp_long', 'bearing_disp_transv', 'bearing_disp', 'unseating_disp_long']:
                                for joint in self.EleIDsBearing:
                                    self.edps_msa[key][joint][gm_idx, :] = list(map(max, edp_gm1[key][joint], edp_gm2[key][joint]))
                            else:
                                self.edps_msa[key][gm_idx, :] = list(map(max, edp_gm1[key], edp_gm2[key]))

                        # Save some analysis info to the log file and print on the screen during the analysis
                        time_text = get_run_time(t0)
                        log.write(gm_text + '\n')
                        log.write(f'gm_components: {gm_components}\n')
                        log.write(anlys1 + '\n')
                        log.write(f'gm_components: {gm_components2}\n')
                        log.write(anlys2 + '\n')
                        log.write(time_text + '\n')
                        log.write(end_text + "\n")
                        print(time_text)
                        print(end_text)

                    else:
                        for key in self.edps_msa:
                            if key == 'convergence_index':
                                self.edps_msa['convergence_index'][gm_idx, 0] = c_index1
                            elif key in ['bearing_disp_long', 'bearing_disp_transv', 'bearing_disp', 'unseating_disp_long']:
                                for joint in self.EleIDsBearing:
                                    self.edps_msa[key][joint][gm_idx, :] = edp_gm1[key][joint]
                            else:
                                self.edps_msa[key][gm_idx, :] = edp_gm1[key]

                        # Save some analysis info to the log file and print on the screen during the analysis
                        time_text = get_run_time(t0)
                        log.write(gm_text + '\n')
                        log.write(anlys1 + '\n')
                        log.write(time_text + '\n')
                        log.write(end_text + "\n")
                        print(time_text)
                        print(end_text)

                # Add the errors encountered during the analyses to the log file
                for error in error_log:  # add the error log
                    log.write('%s\n' % error)

            # Save the object
            with open(os.path.join(out_dir, 'edps.pkl'), 'wb') as handle:
                pickle.dump(self.edps_msa, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("#########################################################################")
        print("             Multiple Stripes Analysis (MSA) is Completed!               ")
        print("#########################################################################")
