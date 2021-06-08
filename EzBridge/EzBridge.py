"""
EzBridge: Program to model and analyze Ordinary RC Bridges
Units: kN, m, sec

Author: Volkan Ozsarac, Earthquake Engineering PhD Candidate
Affiliation: University School for Advanced Studies IUSS Pavia
e-mail: volkanozsarac@iusspavia.it
"""

#  ----------------------------------------------------------------------------
#  Import Python Libraries
#  ----------------------------------------------------------------------------
import os
import sys
import pickle
import numpy as np
import openseespy.opensees as ops
import pandas as pd

from . import Analysis
from .Utility import program_info, create_outdir, distance, ReadRecord, def_units
from .Utility import Get_T0, RunTime, sdof_ltha
from .Modelling import Builder
from . import Recorders
from . import Rendering
from .BridgeSummary import BridgeSummary, PierInfo


class Main(Builder, BridgeSummary, PierInfo):

    def __init__(self, model, output_dir='Outputs'):
        """
        -------------------------------
        OBJECT INITIATION
        -------------------------------
        """

        # PRINT BASIC PROGRAM INFO
        print(program_info())

        # INITIALIZE THE CLASS OBJECTS USING INHERITANCE
        Builder.__init__(self)
        BridgeSummary.__init__(self)
        PierInfo.__init__(self)

        # MODEL INPUT
        self.model = model

        # CREATE DIRECTORIES
        self.out_dir = output_dir
        create_outdir(self.out_dir)
        self.animation_dir = os.path.join(output_dir, 'Animation')
        self.nrha_dir = os.path.join(output_dir, 'NRHA')

        # DEFAULT ANIMATION PARAMETERS
        self.animate = 0
        self.Movie = 0
        self.FrameStep = 5
        self.scale = 50
        self.fps = 50

        # DEFAULT ANALYSIS PARAMETERS
        self.constraintType = 'Penalty'
        self.numbererType = 'RCM'
        self.systemType = 'UmfPack'
        self.alphaS = 1e16
        self.alphaM = 1e16

        global end_text
        # The following end_text is used for separation of some text outputs
        end_text = '-------------------------------------------------------------------------'

    def generate_model(self):
        """
        -------------------------------
        MODEL GENERATOR
        -------------------------------
        """

        # Get bent axial forces with elastic sections
        self._get_axial_pier()
        # Build the model
        self._build()
        self._get_CircPier_info()

    def get_summary(self):
        self._get_summary()

    def animation_config(self, animate, Movie, FrameStep, scale, fps):
        """
        -------------------------------
        SET ANIMATION PARAMETERS
        -------------------------------
        Args:
            animate: flat to animate the response
            Movie: flag to save animation
            FrameStep: frame step size to use for plotting
            scale: scaling factor to use for animation
            fps: frames per second to use in animation

        Returns:

        """

        self.animate = animate
        self.Movie = Movie
        self.FrameStep = FrameStep
        self.scale = scale
        self.fps = fps

    def analysis_config(self, constraintType, numbererType, systemType, alphaS=1e16, alphaM=1e16):
        """
        -----------------------------------
        SET SOME ANALYSIS PARAMETERS
        -----------------------------------
        Args:
            constraintType: Set the constraint type for analysis ('Transformation' or 'Penalty')
            numbererType: 'RCM' Set the the numberer type for analysis (mapping between equation numbers and DOFs)
            systemType: 'UmfPack'  Set the system type for analysis (Linear Equation Solvers)
            alphaS: penalty factor on single point constraints
            alphaM	penalty factor on multi-point constraints
        Returns:

        """
        # Set analysis parameters
        self.constraintType = constraintType
        self.numbererType = numbererType
        self.systemType = systemType
        self.alphaS = alphaS
        self.alphaM = alphaM

    def set_recorders(self):
        """
        -------------------------------
        ADD THE RECORDERS
        -------------------------------
        Notes
        -----
            Using this method will activate the recorders defined
            inside EzGM/Recorders.py
            
        """
        Recorders.user(self)

    def plot_model(self, show_node_tags='no',
                   show_element_tags='no',
                   show_node='no'):
        """
        --------------------------
        MODEL PLOTTING
        --------------------------
        """
        Rendering.plot_model(self, show_node_tags,
                             show_element_tags, show_node)

    def plot_modeshape(self, modeNumber=1, scale=20):
        """
        --------------------------
        MODE SHAPE PLOTTING
        --------------------------
        """
        Rendering.plot_modeshape(self, modeNumber, scale)

    def plot_sections(self, save = 0):
        """
        ---------------------------
        PIER CROSS SECTION PLOTTING
        ---------------------------
        """
        Rendering.plot_sec(self, save)

    def plot_deformedshape(self, ax=None, scale=5):
        """
        ---------------------------
        DEFORMED SHAPE PLOTTING
        ---------------------------
        """
        Rendering.plot_deformedshape(self, ax, scale)

    def _get_axial_pier(self):
        """
        --------------------------------
        AXIAL LOAD CALCULATION FOR PIERS
        --------------------------------
        """
        eleType_pier = self.model['Bent']['EleType']
        AbutType = self.model['Abutment_BackFill']['Type']

        self.model['Bent']['EleType'] = 0  # use elastic beam column elements
        self.model['Abutment_BackFill']['Type'] = 'None'  # use fixed abutment to get axial loads
        self._build()  # build the model
        # F_total = self.gravity(pflag=0, load_type = 0)  # perform gravity analysis
        F_total = self.gravity(pflag=0, load_type=1)  # perform gravity analysis

        self.BentAxialForces = []
        for eleList in self.EleIDsBent:
            Forces = []
            for eleTag in eleList:
                Forces.append(ops.eleForce(eleTag, 3))
            self.BentAxialForces.append(Forces)

        self.AB1AxialForces = 0
        for eleTag in self.EleIDsBearing[0]:
            self.AB1AxialForces += ops.eleForce(eleTag, 3)

        self.AB2AxialForces = 0
        for eleTag in self.EleIDsBearing[-1]:
            self.AB2AxialForces += ops.eleForce(eleTag, 3)

        self.model['Bent']['EleType'] = eleType_pier  # return back to the actual inputs
        self.model['Abutment_BackFill']['Type'] = AbutType

        self.M_total = F_total / 9.81  # total mass assigned to the structure

    @staticmethod
    def wipe_model():
        """
        -------------------------------
        MODEL CLEANER
        -------------------------------
        """
        ops.wipe()
        ops.wipeAnalysis()

    def _config(self):
        """
        -----------------------------------
        CONFIGURES SOME ANALYSIS PARAMETERS
        -----------------------------------
        """

        ops.wipeAnalysis()
        if self.constraintType == 'Penalty' or 'Lagrange':
            ops.constraints(self.constraintType, self.alphaS, self.alphaM)
        else:
            ops.constraints(self.constraintType)
        ops.numberer(self.numbererType)
        ops.system(self.systemType)

    @staticmethod
    def _eigen(numEigen, pflag=0):
        """
        --------------------------
        EIGENVALUE ANALYSIS
        --------------------------
        Args:
            numEigen: number of eigenvalues
        Returns:
            Lambda: Eigenvalues
        """
        if pflag == 1:
            print('Performing eigenvalue analysis.')
        ops.wipeAnalysis()
        listSolvers = ['-genBandArpack', '-fullGenLapack', '-symmBandLapack']
        ok = 1
        for s in listSolvers:
            if pflag == 1:
                print("Using %s as solver..." % s[1:])
            try:
                eigenValues = ops.eigen(s, numEigen)
                catchOK = 0
                ok = 0
            except:
                catchOK = 1

            if catchOK == 0:
                for i in range(numEigen):
                    if eigenValues[i] < 0:
                        ok = 1
                if ok == 0:
                    if pflag == 1:
                        print('Eigenvalue analysis is completed.')
                    break
        if ok != 0:
            print("Error on eigenvalue something is wrong...")
            sys.exit()
        else:
            Lambda = np.asarray(eigenValues)
            return Lambda

    def gravity(self, load_type=1, pflag=1):
        """
        ------------------------------------------------------------------------------------------------------------
        GRAVITY ANALYSIS
        ------------------------------------------------------------------------------------------------------------
        Args:
            load_type: distributed (1) or point load (0)
            pflag: flag to print information (1 or 0)
        """

        # CREATE TIME SERIES
        ops.timeSeries("Constant", 1)

        # CREATE A PLAIN LOAD PATTERN
        ops.pattern('Plain', 1, 1)

        # APPLY DECK LOADS
        for i in range(len(self.EleIDsDeck)):
            EleNodes = ops.eleNodes(self.EleIDsDeck[i])
            # Get the vectors to assign uniformly distributed gravity loads
            Coord1 = ops.nodeCoord(EleNodes[0])
            Coord2 = ops.nodeCoord(EleNodes[1])
            # Convert uniformly distributed gravity loads to point loads
            Lele = distance(Coord1, Coord2)
            Pload = Lele * self.EleLoadsDeck[i] / 2

            if load_type == 0:  # assign loads as point load
                ops.load(EleNodes[0], 0, 0, -Pload, 0, 0, 0)
                ops.load(EleNodes[1], 0, 0, -Pload, 0, 0, 0)
            elif load_type == 1:  # assign loads as uniformly distributed load
                ops.eleLoad('-ele', self.EleIDsDeck[i], '-type', '-beamUniform',
                            -self.EleLoadsDeck[i] * self.Vy_deck[i][2],
                            -self.EleLoadsDeck[i] * self.Vz_deck[i][2], -self.EleLoadsDeck[i] * self.Vx_deck[i][2])

        # APPLY PIER LOADS
        for i in range(self.num_bents):
            for EleID in self.EleIDsBent[i]:
                EleNodes = ops.eleNodes(EleID)
                Coord1 = ops.nodeCoord(EleNodes[0])
                Coord2 = ops.nodeCoord(EleNodes[1])
                Lele = distance(Coord1, Coord2)
                Pload = Lele * self.EleLoadsBent[i] / 2

                if load_type == 0:  # assign loads as point load
                    ops.load(EleNodes[0], 0, 0, -Pload, 0, 0, 0)
                    ops.load(EleNodes[1], 0, 0, -Pload, 0, 0, 0)
                elif load_type == 1:  # assign loads as uniformly distributed load
                    ops.eleLoad('-ele', EleID, '-type', '-beamUniform', 0.0, 0.0, -self.EleLoadsBent[i])

        # APPLY BENTCAP LOADS
        if all(num == 1 for num in self.model['Bearing']['N']) and self.num_piers == 1:

            for i in range(self.num_bents):
                ops.load(self.BcapNodes[i][0], 0, 0, -self.PointLoadsBcap[i], 0, 0, 0)
        else:
            for i in range(len(self.EleIDsBcap)):
                EleNodes = ops.eleNodes(self.EleIDsBcap[i])
                Coord1 = ops.nodeCoord(EleNodes[0])
                Coord2 = ops.nodeCoord(EleNodes[1])
                Lele = distance(Coord1, Coord2)
                Pload = Lele * self.EleLoadsBcap[i] / 2

                if load_type == 0:  # assign loads as point load
                    ops.load(EleNodes[0], 0, 0, -Pload, 0, 0, 0)
                    ops.load(EleNodes[1], 0, 0, -Pload, 0, 0, 0)
                elif load_type == 1:  # assign loads as uniformly distributed load
                    ops.eleLoad('-ele', self.EleIDsBcap[i], '-type', '-beamUniform', 0.0, -self.EleLoadsBcap[i], 0.0)

        for i in range(len(self.EleIDsLink)):
            EleNodes = ops.eleNodes(self.EleIDsLink[i])
            Coord1 = ops.nodeCoord(EleNodes[0])
            Coord2 = ops.nodeCoord(EleNodes[1])
            Lele = distance(Coord1, Coord2)
            Pload = Lele * self.EleLoadsLink[i] / 2

            if load_type == 0:  # assign loads as point load
                ops.load(EleNodes[0], 0, 0, -Pload, 0, 0, 0)
                ops.load(EleNodes[1], 0, 0, -Pload, 0, 0, 0)
            elif load_type == 1:  # assign loads as uniformly distributed load
                ops.eleLoad('-ele', self.EleIDsLink[i], '-type', '-beamUniform', 0.0, -self.EleLoadsLink[i], 0.0)

        # Apply the ele loads for piles
        if self.model['Bent_Foundation']['Type'] == 'Pile-Shaft':
            for i in range(self.num_bents):
                EleLoad = self.EleLoadsPile[i]
                for ele in self.EleIDsPile[i]:
                    EleNodes = ops.eleNodes(ele)
                    Coord1 = ops.nodeCoord(EleNodes[0]);
                    Coord2 = ops.nodeCoord(EleNodes[1])
                    Lele = ((Coord2[0] - Coord1[0]) ** 2 + (Coord2[1] - Coord1[1]) ** 2 + (
                                Coord2[2] - Coord1[2]) ** 2) ** 0.5
                    Pload = Lele * EleLoad / 2
                    if load_type == 0:  # assign loads as point load
                        ops.load(EleNodes[0], 0, 0, -Pload, 0, 0, 0);
                        ops.load(EleNodes[1], 0, 0, -Pload, 0, 0, 0)
                    elif load_type == 1:  # assign loads as uniformly distributed load
                        ops.eleLoad('-ele', ele, '-type', '-beamUniform', 0.0, 0.0, -EleLoad)

        elif self.model['Bent_Foundation']['Type'] == 'Group Pile':
            for i in range(self.num_bents):
                EleLoad = self.EleLoadsPile[i]
                for ele in self.EleIDsPile[i]:
                    EleNodes = ops.eleNodes(ele)
                    Coord1 = ops.nodeCoord(EleNodes[0]);
                    Coord2 = ops.nodeCoord(EleNodes[1])
                    Lele = ((Coord2[0] - Coord1[0]) ** 2 + (Coord2[1] - Coord1[1]) ** 2 + (
                                Coord2[2] - Coord1[2]) ** 2) ** 0.5
                    Pload = Lele * EleLoad / 2
                    if load_type == 0:  # assign loads as point load
                        ops.load(EleNodes[0], 0, 0, -Pload, 0, 0, 0);
                        ops.load(EleNodes[1], 0, 0, -Pload, 0, 0, 0)
                    elif load_type == 1:  # assign loads as uniformly distributed load
                        ops.eleLoad('-ele', ele, '-type', '-beamUniform', 0.0, 0.0, -EleLoad)

                ops.load(self.PcapNodes[i], 0, 0, -self.PcapWeight, 0, 0, 0)

        # SET ANALYSIS PARAMETERS
        self._config()
        ops.test('NormDispIncr', 1e-8, 500)
        ops.algorithm('Newton')
        # ops.test('NormDispIncr', 1e-4, 100)
        # ops.algorithm('KrylovNewton')
        # ops.test('NormDispIncr', 1e-4, 500)
        # ops.test('EnergyIncr', 1e-3, 500)
        # ops.test('NormUnbalance', 1e-4, 100)
        # ops.algorithm('ModifiedNewton', '-initial')
        # ops.algorithm('Linear') # if system is linear sure.
        nG = 1000
        ops.integrator('LoadControl', 1 / nG)
        ops.analysis('Static')

        # DO THE ANALYSIS
        ops.analyze(nG)

        # maintain constant gravity loads and reset time to zero
        ops.loadConst('-time', 0.0)

        # Get the reaction forces in vertical direction, this is for verification purposes
        ops.reactions('-dynamic', '-rayleigh')
        F_Abut1 = np.zeros([1, 6])
        F_Abut2 = np.zeros([1, 6])
        F_total = np.zeros([1, 6])

        for node in self.fixed_AB1Nodes:
            F_Abut1 += np.array(ops.nodeReaction(node))
        text1 = ('%4s   | %6s | %6s | %6s | %6s | %6s | %6s |' % ('A1',
                                                                  "{:.0f}".format(F_Abut1[0, 0]),
                                                                  "{:.0f}".format(F_Abut1[0, 1]),
                                                                  "{:.0f}".format(F_Abut1[0, 2]),
                                                                  "{:.0f}".format(F_Abut1[0, 3]),
                                                                  "{:.0f}".format(F_Abut1[0, 4]),
                                                                  "{:.0f}".format(F_Abut1[0, 5])))
        text2 = []
        for i in range(self.num_bents):
            F_bent = np.zeros([1, 6])
            if self.model['Bent_Foundation']['Type'] == 'Fixed':
                for j in range(len(self.fixed_BentNodes[i])):
                    F_pier = np.zeros([1, 6])
                    node = self.fixed_BentNodes[i][j]
                    F_pier += np.array(ops.nodeReaction(node))
                    text2.append('%6s | %6s | %6s | %6s | %6s | %6s | %6s |' %
                                 ('B' + str(i + 1) + '/P' + str(j + 1), "{:.0f}".format(F_pier[0, 0]),
                                  "{:.0f}".format(F_pier[0, 1]),
                                  "{:.0f}".format(F_pier[0, 2]), "{:.0f}".format(F_pier[0, 3]),
                                  "{:.0f}".format(F_pier[0, 4]), "{:.0f}".format(F_pier[0, 5])))
                    F_bent = F_bent + F_pier

            elif self.model['Bent_Foundation']['Type'] == 'Springs':
                node = self.fixed_BentNodes[i]
                F_bent += np.array(ops.nodeReaction(node))
                text2.append('%4s   | %6s | %6s | %6s | %6s | %6s | %6s |' %
                             ('B' + str(i + 1), "{:.0f}".format(F_bent[0, 0]),
                              "{:.0f}".format(F_bent[0, 1]),
                              "{:.0f}".format(F_bent[0, 2]), "{:.0f}".format(F_bent[0, 3]),
                              "{:.0f}".format(F_bent[0, 4]), "{:.0f}".format(F_bent[0, 5])))

            else:
                for j in range(len(self.fixed_BentNodes[i])):
                    for node in self.fixed_BentNodes[i][j]:
                        F_bent += np.array(ops.nodeReaction(node))
                text2.append('%4s   | %6s | %6s | %6s | %6s | %6s | %6s |' %
                             ('B' + str(i + 1), "{:.0f}".format(F_bent[0, 0]),
                              "{:.0f}".format(F_bent[0, 1]),
                              "{:.0f}".format(F_bent[0, 2]), "{:.0f}".format(F_bent[0, 3]),
                              "{:.0f}".format(F_bent[0, 4]), "{:.0f}".format(F_bent[0, 5])))

            F_total += F_bent

        text2 = '\n'.join(text2)

        for node in self.fixed_AB2Nodes:
            F_Abut2 += np.array(ops.nodeReaction(node))

        text3 = ('%4s   | %6s | %6s | %6s | %6s | %6s | %6s |' % ('A2',
                                                                  "{:.0f}".format(F_Abut2[0, 0]),
                                                                  "{:.0f}".format(F_Abut2[0, 1]),
                                                                  "{:.0f}".format(F_Abut2[0, 2]),
                                                                  "{:.0f}".format(F_Abut2[0, 3]),
                                                                  "{:.0f}".format(F_Abut2[0, 4]),
                                                                  "{:.0f}".format(F_Abut2[0, 5])))

        F_total = F_total + F_Abut1 + F_Abut2

        text4 = ('%5s  | %6s | %6s | %6s | %6s | %6s | %6s |' % ('SUM',
                                                                 "{:.0f}".format(F_total[0, 0]),
                                                                 "{:.0f}".format(F_total[0, 1]),
                                                                 "{:.0f}".format(F_total[0, 2]),
                                                                 "{:.0f}".format(F_total[0, 3]),
                                                                 "{:.0f}".format(F_total[0, 4]),
                                                                 "{:.0f}".format(F_total[0, 5])))

        if pflag == 1:
            print("#########################################################################")
            print("                      Performing Graviy Analysis...                      ")
            print("#########################################################################")
            print('  Loc. | Fx[kN] | Fy[kN] | Fz[kN] | Mx[kN] | My[kN] | Mz[kN] |')
            print(text1)
            print(text2)
            print(text3)
            print(text4)
            print(end_text)

        return F_total[0, 2]  # total vertical reaction force --> total weight

    def modal(self, numEigen=1, pflag=1):
        """
        ----------------------------------------------------------------------------
        MODAL ANALYSIS
        ----------------------------------------------------------------------------
        Notes
        -----
            The script makes use of the built-in function created by Massimo Petracca.
            The function is already included inside OpenSees framework.
            
        Parameters
        ----------
            numEigen : int, optional (The default is 1)
                Number of eigenvalues to calculate.
            pflag    : int (1 or 0)
                flag to print output information on screen

        Returns
        -------
            None

        """
        print("#########################################################################")
        print("                         Performing Modal Analysis...                    ")
        print("#########################################################################")

        # compute the modal properties
        self._eigen(numEigen, pflag)
        outname = os.path.join(self.out_dir, "Modal_Properties.out")
        args = ["-print", "-file", outname, "-unorm"]
        if pflag == 0:
            args.remove("-print")
        ops.modalProperties(*args)
        # Read back stuff from the modal properties.txt it could be necessary
        self.modal_properties = []
        with open(outname, "r") as file:
            for line in file:
                self.modal_properties.append(line)

    def modal_simple(self, numEigen=1, pflag=1):
        """
        ----------------------------------------------------------------------------
        MODAL ANALYSIS
        ----------------------------------------------------------------------------
        Notes
        -----
            The script is created by VO, it uses the mass matrix 
            obtained via -printA option available in opensees.
            Total (activated) mass is obtained by summing the masses assigned to the
            unrestrained degrees of freedoms (DOFs). Thus, it should not be confused
            with total mass assigned to all DOFs. Influence vectors for rotational
            excitation are not correct at the moment, this addition remains as future work.
            Which reference point to use is not clear for rotational excitations.
            SAP2000 and Seismostruct use different reference points.

        Parameters
        ----------
            numEigen : int, optional (The default is 1)
                Number of eigenvalues to calculate.
            pflag    : int (1 or 0)
                flag to print output information on screen

        Returns
        -------
            T        : numpy.ndarray
                Period array for the first numEigen modes.
            M_ratios  : dictionary
                Effective modal mass participation ratios for the first numEigen modes.
            M_factors : dictionary
                Modal participation factors for the first numEigen modes.
            M_totals    : dictionary
                Total activated masses.

        """
        print("#########################################################################")
        print("                         Performing Modal Analysis...                    ")
        print("#########################################################################")
        ops.wipeAnalysis()
        ops.numberer("Plain")
        ops.system('FullGeneral')
        ops.analysis('Transient')

        # Extract the Mass Matrix
        # Note that this is not the global mass matrix, but unrestrained part (Muu)
        ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
        ops.analyze(1, 0.0)
        # Number of equations in the model
        N = ops.systemSize()  # Has to be done after analyze
        M_matrix = ops.printA('-ret')  # Or use ops.printA('-file','M.out')
        M_matrix = np.array(M_matrix)  # Convert the list to an array
        M_matrix.shape = (N, N)  # Make the array an NxN matrix
        print(end_text)
        print('Extracting the mass matrix, ignore the warnings...')

        # Determine maximum number of DOFs/node used in the system
        NDF = 0
        for node in ops.getNodeTags():
            temp = len(ops.nodeDOFs(node))
            if temp > NDF:
                NDF = temp

        DOFs = []  # List containing indices of unrestrained DOFs
        used = {}  # Dictionary with nodes and associated unrestrained DOFs
        l_dict = {}  # Dictionary containing influence vectors
        M_ratios = {}  # Dictionary containing effective modal masses ratios
        M_factors = {}  # Dictionary containing modal participation factors
        for i in range(1, NDF + 1):
            l_dict[i] = np.zeros([N, 1])
            M_ratios[i] = np.zeros(numEigen)
            M_factors[i] = np.zeros(numEigen)

        # Create the influence vectors, and get the unrestrained DOFs assigned to the nodes
        # TODO -1: The influence vectors are not correct in case of rotational excitations
        # One typical approach is to use center of mass on plane
        idx = 0  # Counter for unrestrained DOFs
        for node in ops.getNodeTags():  # Start iterating over each node
            used[node] = []  # Unrestrained local DOF ids
            ndof = len(ops.nodeDOFs(node))  # Total number of DOFs assigned
            for j in range(ndof):  # Iterate over each DOF
                temp = ops.nodeDOFs(node)[j]  # Get the global DOF id (-1 if restrained)
                if temp not in DOFs and temp >= 0:  # Check if this DOF is unrestrained and is not known before
                    DOFs.append(temp)  # Save the global id of DOF
                    used[node].append(j + 1)  # Save the local id of DOF
                    l_dict[j + 1][idx, 0] = 1  # Influence vectors for horizontal and vertical excitations
                    idx += 1  # Increase the counter

        # This does not seem necessary when numberer is "Plain"
        # But lets reorganize the mass matrix anyway
        M_matrix = M_matrix[DOFs, :][:, DOFs]

        # Calculate the total masses assigned to the unrestrained DOFs
        M_totals = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for i in range(1, NDF + 1):
            M_totals[i] = (l_dict[i].T @ M_matrix @ l_dict[i])[0, 0]

        # Perform eigenvalue analysis
        Lambda = self._eigen(numEigen, pflag=1)
        Omega = Lambda ** 0.5
        T = 2 * np.pi / Omega
        frq = 1 / T

        # Note: influence factors for rotational excitation is wrong!
        # Obtain modal properties
        for mode in range(1, numEigen + 1):
            idx = 0
            phi = np.zeros([N, 1])  # Eigen vector
            for node in used:
                for dof in used[node]:
                    phi[idx, 0] = ops.nodeEigenvector(node, mode, dof)
                    idx += 1

            phi = phi / (phi.T @ M_matrix @ phi) ** 0.5  # Normalize the eigen vector by modal mass
            Mn = phi.T @ M_matrix @ phi  # Modal mass (should always be equal to 1)

            for j in range(1, NDF + 1):
                if M_totals[j] != 0:  # Check if any mass is assigned
                    Ln = phi.T @ M_matrix @ l_dict[j]  # Modal excitation factor
                    Mnstar = (Ln ** 2 / Mn)[0, 0]  # Effective modal mass
                    M_factors[j][mode - 1] = Ln / Mn  # Modal participation factor
                    M_ratios[j][mode - 1] = (Mnstar / M_totals[j] * 100)  # Effective modal mass participation ratio [%]

        for j in range(1, 7):
            try:
                M_ratios[j]
            except:
                M_ratios[j] = np.zeros(numEigen)
                M_factors[j] = np.zeros(numEigen)

        # TODO-1: Results are not correct for rotational excitation cases, for now ignore those.
        del M_ratios[6], M_ratios[5], M_ratios[4]
        del M_factors[6], M_factors[5], M_factors[4]

        # Calculate cumulative modal mass participation ratio
        sM1 = np.cumsum(M_ratios[1])
        sM2 = np.cumsum(M_ratios[2])
        sM3 = np.cumsum(M_ratios[3])

        # Print modal analysis results
        if pflag == 1:
            arguments = ['Modal Periods and Frequencies', '%4s|%8s|%10s|%12s|%12s'
                         % ('Mode', 'T [sec]', 'f [Hz]', '\u03C9 [rad/sec]', '\u03BB [rad\u00b2/sec\u00b2]')]
            for mode in range(numEigen):
                arguments.append('%4s|%8s|%10s|%12s|%12s'
                                 % ("{:.0f}".format(mode + 1), "{:.4f}".format(T[mode]), "{:.3f}".format(frq[mode]),
                                    "{:.2f}".format(Omega[mode]), "{:.2f}".format(Lambda[mode])))
            arguments.append('Total Activated Masses')
            arguments.append('%8s|%8s|%8s'
                             % ('M\u2081', 'M\u2082', 'M\u2083'))
            arguments.append('%8s|%8s|%8s'
                             % (
                                 "{:.2f}".format(M_totals[1]), "{:.2f}".format(M_totals[2]),
                                 "{:.2f}".format(M_totals[3])))
            arguments.append('Modal Mass Participation Factors')
            arguments.append('%4s|%7s|%7s|%7s'
                             % ('Mode', '\u0393\u2081', '\u0393\u2082', '\u0393\u2083'))
            for mode in range(numEigen):
                arguments.append('%4s|%7s|%7s|%7s' % ("{:.0f}".format(mode + 1),
                                                      "{:.3f}".format(M_factors[1][mode]),
                                                      "{:.3f}".format(M_factors[2][mode]),
                                                      "{:.3f}".format(M_factors[3][mode])))
            arguments.append('Effective Modal Mass Participation Ratios [%]')
            arguments.append('%4s|%7s|%7s|%7s'
                             % ('Mode', 'U\u2081', 'U\u2082', 'U\u2083'))
            for mode in range(numEigen):
                arguments.append('%4s|%7s|%7s|%7s' % ("{:.0f}".format(mode + 1),
                                                      "{:.3f}".format(M_ratios[1][mode]),
                                                      "{:.3f}".format(M_ratios[2][mode]),
                                                      "{:.3f}".format(M_ratios[3][mode])))
            arguments.append('Cumulative Effective Modal Mass Participation Ratios [%]')
            arguments.append('%4s|%7s|%7s|%7s'
                             % ('Mode', '\u2211U\u2081', '\u2211U\u2082', '\u2211U\u2083'))
            for mode in range(numEigen):
                arguments.append('%4s|%7s|%7s|%7s' % ("{:.0f}".format(mode + 1),
                                                      "{:.3f}".format(sM1[mode]), "{:.3f}".format(sM2[mode]),
                                                      "{:.3f}".format(sM3[mode])))

                # To the screen
            arguments = '\n'.join(arguments)
            print(arguments)

            # To the .csv file
            outname = os.path.join(self.out_dir, "Modal_Properties.csv")
            with open(outname, 'w', encoding='utf-32') as f:
                f.write(arguments)

        print(end_text)
        return T, M_ratios, M_factors, M_totals

    def rsa(self, path_spec, num_modes, damping, direction):
        """
        ----------------------------------------------------------------------------
        RESPONSE SPECTRUM ANALYSIS
        ----------------------------------------------------------------------------
        Notes
        -----
            The method makes use of the built-in function created by Massimo Petracca.
            The responseSpectrum function is already included inside OpenSees framework.

        Parameters
        ----------
        path_spec : str
            target spectrum file (.txt), first column is for periods, whereas
            second column specifies spectral acceleration values.
        num_modes : int
            number of modes to use.
        damping : float
            damping ratio (e.g. 0.02), assummed to be same for all modes.
        direction : int
            analysis direction.

        Returns
        -------
            EleForces_RSA dictionary which contains the element forces obtained
            from response spectrum analysis is created.

        """

        # Remove any existing analysis configuration
        ops.wipeAnalysis()

        # CQC combination
        def CQC(mu, lambdas, dmp, scalf):
            u = 0.0
            ne = len(lambdas)
            for i in range(ne):
                for j in range(ne):
                    di = dmp[i]
                    dj = dmp[j]
                    bij = lambdas[i] / lambdas[j]
                    rho = ((8.0 * np.sqrt(di * dj) * (di + bij * dj) * (bij ** (3.0 / 2.0))) /
                           ((1.0 - bij ** 2.0) ** 2.0 + 4.0 * di * dj * bij * (1.0 + bij ** 2.0) +
                            4.0 * (di ** 2.0 + dj ** 2.0) * bij ** 2.0))
                    u += scalf[i] * mu[i] * scalf[j] * mu[j] * rho
            return np.sqrt(u)

        # Read target spectrum, first col: periods, second col: Sa
        Periods = np.loadtxt(path_spec)[:, 0]
        Sa = np.loadtxt(path_spec)[:, 1]

        # Time series tag
        tsTag = 2

        # the response spectrum function
        ops.timeSeries("Path", tsTag, "-time", *Periods, "-values", *Sa, "-factor", 9.806)

        # set some analysis parameters, these depend on the model type that you use
        self._config()
        ops.test("NormUnbalance", 0.0001, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.0)
        ops.analysis("Static")

        # run the eigenvalue analysis with "num_modes" modes
        # and obtain the eigenvalues
        eigs = ops.eigen("-genBandArpack", num_modes)

        # compute the modal properties
        ops.modalProperties("-unorm")

        # currently we use same damping for each mode
        dmp = [damping] * len(eigs)
        # we don't want to scale some modes...
        scalf = [1.0] * len(eigs)

        # Maximum number of DOFs/node used in the system
        NDF = 6

        # Ele Force dictionary
        EleForces = {}

        for ele in ops.getEleTags():
            if str(ele)[-3:] != self.RigidTag:  # Do not include rigid elements
                EleForces[ele] = np.zeros((num_modes, 2 * NDF))

        for mode in range(len(eigs)):
            ops.responseSpectrum(tsTag, direction, '-mode', mode + 1)

            # Element Forces
            for ele in EleForces.keys():
                forces = ops.eleForce(ele)
                EleForces[ele][mode] = forces

        # post process the results doing the CQC modal combination
        CQCForces = {}
        for ele in EleForces.keys():
            forces = EleForces[ele]
            CQCForces[ele] = CQC(forces, eigs, dmp, scalf)

        self.EleForces_RSA = CQCForces

    def nspa(self, ctrlNode, PushOption=1, scheme='Uniform', ctrlDOF=1,
             mu=1, dref=1, nSteps=5000, numCycles=2, IOflag=1,
             mode_nspa=1, Bilin_approach='EC'):
        """
        ----------------------------------------------------------------------------
        Nonlinear Static Pushover Analysis Parameters (NSPA)
        ----------------------------------------------------------------------------
        PushOption = 1              1: Non-Cyclic NSPA, 2: Cyclic NSPA
        ctrlNode = 3004             Control Node
        ctrlDOF = 1                 Control Degrees of Freedom
        mu = 1                      Ductility, with respect to dref
        dref = 0.8*m                Reference displacement point(e.g. yield displacement), note that final
                                    displacement is mu*dref
        nSteps = 2000               Number of steps used to carry out NSPA
        numCycles = 2               Number of cycles, required for Cyclic NSPA
        IOflag = 1                  Output information on the screen (0,1,2,3)
        scheme = 'Modal'            Set the loading scheme to use for pushover analysis ('Uniform', 'Modal')
        mode_nspa = 1               Set the mode number used for mode proportional loading scheme
        Bilin_approach = 'EC'       Approach to follow for bilinear idealization ('EC','ASCE','NTC')

        """

        if PushOption == 1:
            print("#########################################################################")
            print("     Performing Nonlinear Non-Cyclic Static Pushover Analysis (NSPA)...  ")
            print("#########################################################################")
        if PushOption == 2:
            print("#########################################################################")
            print("       Performing Nonlinear Cyclic Static Pushover Analysis (NSPA)...    ")
            print("#########################################################################")

        # Uniformly distributed loading along the deck
        if scheme == 'Uniform':
            PushNodes = []  # Nodes to push
            for i in range(len(self.D1Nodes)):
                PushNodes.append(self.D1Nodes[i])
                PushNodes.append(self.D2Nodes[i])
            if any(self.DeckIntNodes):  # Add the internal deck nodes if deck elements are discretized
                for i in range(len(self.DeckIntNodes)):
                    for node in self.DeckIntNodes[i]:
                        if not node in PushNodes:
                            PushNodes.append(node)

            # Determine normalized uniformly distributed load value.
            # Since the loads are normalized the load factor is equal to the total base shear force.
            Hload = 1 / len(PushNodes)
            if ctrlDOF == 1:
                loadValues = [Hload, 0, 0, 0, 0, 0]
            if ctrlDOF == 2:
                loadValues = [0, Hload, 0, 0, 0, 0]
            # Create the load pattern and apply loads
            ops.timeSeries('Linear', 2)  # Define the timeSeries for the load pattern
            ops.pattern('Plain', 2, 2)  # Define load pattern -- generalized
            for nodeTag in PushNodes:
                ops.load(nodeTag, *loadValues)

        elif scheme == 'Modal':  # Modal pushover (phi*M)
            ops.system('FullGeneral')
            ops.analysis('Transient')
            # Extract the Mass Matrix
            ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
            ops.analyze(1, 0.0)
            # Number of equations in the model
            N = ops.systemSize()  # Has to be done after analyze
            M_matrix = ops.printA('-ret')  # Or use ops.printA('-file','M.out')
            M_matrix = np.array(M_matrix)  # Convert the list to an array
            M_matrix.shape = (N, N)  # Make the array an NxN matrix
            print(end_text)
            print('Extracted the mass matrix, ignore the previous warnings...')
            # Rearrange the mass matrix in accordance with nodelist order from getNodeTags()
            DOFs = []  # These are the idx of all the DOFs used in the extract mass matrix, order is rearranged
            Nodes = {}  # Save here the nodes and their associated dofs used in global mass matrix
            NDF = 6  # NDF is number of DOFs/node
            for node in ops.getNodeTags():
                Nodes[node] = []
                for i in range(NDF):
                    temp = ops.nodeDOFs(node)[i]
                    if temp not in DOFs and temp >= 0:
                        DOFs.append(ops.nodeDOFs(node)[i])
                        Nodes[node].append(i + 1)
            M_matrix = M_matrix[DOFs, :][:, DOFs]

            # Determine the loads to apply for modal pushover
            self._eigen(mode_nspa)
            i = 0  # index for DOFs
            H1loads = []  # Load values in dir-2
            H2loads = []  # Load values in dir-1
            FreeNodes = []  # Nodes with unrestrained DOFs (1 or 2)
            for node in Nodes:
                for dof in Nodes[node]:
                    if dof == 1:
                        H1loads.append(ops.nodeEigenvector(node, mode_nspa, 1) * M_matrix[i, i])
                        if not node in FreeNodes:
                            FreeNodes.append(node)
                    if dof == 2:
                        H2loads.append(ops.nodeEigenvector(node, mode_nspa, 2) * M_matrix[i, i])
                        if not node in FreeNodes:
                            FreeNodes.append(node)
                    i += 1

            H1loads = np.array(H1loads)
            H2loads = np.array(H2loads)

            # Normalize the loads. This will make the load factor equal to the total base shear force.
            Hloads = (sum(H1loads) ** 2 + sum(H2loads) ** 2) ** 0.5

            # Reverse the push direction against negative loading
            if sum(H2loads) ** 2 > sum(H1loads) ** 2 and sum(H2loads) < 0:
                H2loads = - H2loads
                H1loads = - H1loads
            elif sum(H1loads) ** 2 > sum(H2loads) ** 2 and sum(H1loads) < 0:
                H2loads = - H2loads
                H1loads = - H1loads

            H1loads = np.asarray(H1loads) / Hloads
            H2loads = np.asarray(H2loads) / Hloads
            # Create the load pattern and apply loads
            ops.timeSeries('Linear', 2)  # Define the timeSeries for the load pattern
            ops.pattern('Plain', 2, 2)  # Define load pattern -- generalized
            i = 0  # Index for nodes
            PushNodes = []  # Nodes that are going to be pushed
            for node in FreeNodes:
                if H1loads[i] != 0 and H2loads[i] != 0:
                    PushNodes.append(node)
                    loadValues = [H1loads[i], H2loads[i], 0, 0, 0, 0]
                    ops.load(node, *loadValues)
                i += 1

        # set some analysis parameters, these depend on the model type that you use
        self._config()

        if self.animate == 1:
            create_outdir(self.animation_dir)
            Recorders.animation(self.animation_dir)

        # Run The Pushover Analysis
        if PushOption == 1:
            [LoadFactor, DispCtrlNode] = Analysis.SinglePush2(self, dref, mu, ctrlNode, ctrlDOF, nSteps, IOflag)
        elif PushOption == 2:
            [LoadFactor, DispCtrlNode] = Analysis.CyclicPush(dref, mu, numCycles, ctrlNode, ctrlDOF, nSteps, IOflag)
        # Analysis ends here

        #  Post Processing
        np.savetxt(os.path.join(self.out_dir, 'NSPA_Summary.txt'), np.column_stack((DispCtrlNode, LoadFactor)))
        if PushOption == 1:
            Rendering.bilin_pushover(self.out_dir, LoadFactor, DispCtrlNode, self.M_total, Bilin_approach, scheme,
                                     ctrlNode)
        if self.animate == 1:
            Rendering.animate_nspa(self, scheme, LoadFactor, DispCtrlNode, ctrlNode)

    def nrha(self, excitation='Uniform', signal='-accel', GMs=None, GMangle=0,
             GM_components=None, GM_factors=None, GMdt=None,
             pFlag=1, damping='Stiffness', Modes=1, xi=0.02,
             xi_modal=None, Dc=10, tFree=0.0, DtFactor=1.0):
        """
        ----------------------------------------------
        Nonlinear Response History Analysis Parameters
        ----------------------------------------------
        excitation='Uniform'                Excitation type ('Uniform' or 'Multi-Support')
        signal = '-accel'                   Type of time series (-accel, -vel, -disp) to apply. Then the input file must
                                            be defined accordingly. Note that in case of Uniform excitation pattern
                                            signal can only be acceleration.
        pFlag = 1                           Flag to print info during analyses (0,1,2,3)
        GMangle = 0                         Ground motion incidence angle, going to be applied on gm components 1 and 2
        damping = 'Rayleigh'                Specify the damping type to use ('Rayleigh','Stiffness','Mass','Modal')
        xi_modal = [0.05, 0.02]             Damping values to assign for the first n modes (Modal damping)
        xi = 0.02                           Damping value (Rayleigh damping)
        Modes = [1,3]                       List of modes to calculate damping coefficients, necessary for
                                            damping = 'Rayleigh'. For 'Stiffness','Mass' this is an integer.
        Dc = 10.0                           Drift capacity (%) to define local collapse in NRHA - Analysis stops
                                            if pier reaches this drift level
        tFree = 0.0                         Additional free vibration time
        DtFactor = 1.0		                Analysis time step is determined as a (DtFactor * GMdt). For example,
                                            if DtFactor = 1.0, analysis time step is equal to ground motion time step

        -----------------------------------------------
        Ground motion INPUT parameters
        -----------------------------------------------
        Records must be placed into GMfiles/NRHA folder
        Ground-motion components (component numbers corresponds to the assumed coordinate system, x(1), y(2), z(3))
        GM_components = [1, 2, 3]

        Ground-motion scaling factor corresponding to each component and gm, g is applied in time series
        GM_factors = [1.0, 1.0, 1.0]

        input file name for each corresponding ground motion component
        GMs = ['RSN1158_KOCAELI_DZC270.txt', 'RSN1158_KOCAELI_DZC180.txt', 'RSN1158_KOCAELI_DZC-UP.txt']
        GMdt = 0.005  Time step of the record, necessary if gm file is not given PEER format

        input file name for each corresponding ground motion component / PEER FORMAT
        GMs = ['RSN1158_KOCAELI_DZC270.AT2', 'RSN1158_KOCAELI_DZC180.AT2', 'RSN1158_KOCAELI_DZC-UP.AT2']

        """

        if GM_components is None:
            GM_components = [1, 2, 3]
        if GM_factors is None:
            GM_factors = [1.0, 1.0, 1.0]

        def get_time_series(GM_components, GMs, GM_direct, GMdt):
            # Get gm components to use
            nPts1 = 0
            nPts2 = 0
            nPtsV = 0
            for i in range(len(GM_components)):
                gm_path = os.path.join(GM_direct, GMs[i])  # Assuming GM is in PEER format

                if GM_components[i] == 1:
                    if gm_path[-4:] == '.AT2':
                        GMdt, _, _, _, gmH1_tmp = ReadRecord(gm_path)
                    else:
                        gmH1_tmp = np.loadtxt(gm_path)
                    nPts1 = len(gmH1_tmp)
                    gmH1_fact = GM_factors[i]
                    gmH1_tmp = gmH1_tmp * gmH1_fact

                if GM_components[i] == 2:
                    if gm_path[-4:] == '.AT2':
                        GMdt, _, _, _, gmH2_tmp = ReadRecord(gm_path)
                    else:
                        gmH2_tmp = np.loadtxt(gm_path)
                    nPts2 = len(gmH2_tmp)
                    gmH2_fact = GM_factors[i]
                    gmH2_tmp = gmH2_tmp * gmH2_fact

                if GM_components[i] == 3:
                    if gm_path[-4:] == '.AT2':
                        GMdt, _, _, _, gmV_tmp = ReadRecord(gm_path)
                    else:
                        gmV_tmp = np.loadtxt(gm_path)
                    nPtsV = len(gmV_tmp)
                    gmV_fact = GM_factors[i]
                    gmV_tmp = gmV_tmp * gmV_fact

            # Assign zero if gm component is not used
            nPts = max(nPts1, nPts2, nPtsV)
            GM_dur = GMdt * nPts
            gmH1 = np.zeros(nPts)
            gmH2 = np.zeros(nPts)
            gmV = np.zeros(nPts)
            try:
                gmH1[:nPts1] = gmH1_tmp
            except:
                pass
            try:
                gmH2[:nPts2] = gmH2_tmp
            except:
                pass
            try:
                gmV[:nPtsV] = gmV_tmp
            except:
                pass

            # Modify the ground motions according to the angle of incidence
            gmH1.shape = (1, nPts)
            gmH2.shape = (1, nPts)
            gmV.shape = (1, nPts)
            gm_mat = np.vstack((gmH1, gmH2))
            theta_mat = np.array([[np.cos(GMangle * np.pi / 180), np.sin(GMangle * np.pi / 180)],
                                  [-np.sin(GMangle * np.pi / 180), np.cos(GMangle * np.pi / 180)]])
            gm_mat = np.vstack((theta_mat @ gm_mat, gmV))

            return gm_mat, GM_dur, GMdt

        def Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode=None):
            # Define the Time Series and and the Load Pattern
            for i in range(1, 4):
                # Setting time series
                gm_i = gm_mat[i - 1, :]
                ops.timeSeries('Path', tsTag, '-dt', GMdt, '-values', *list(gm_i), '-factor', 1.0)

                # Creating UniformExcitation load pattern
                if excitation == 'Uniform':
                    ops.pattern('UniformExcitation', pTag, i, '-accel', tsTag)

                elif excitation == 'Multi-Support':
                    # Creating MultipleSupport Excitation load pattern                        
                    ops.pattern('MultipleSupport', pTag)
                    ops.groundMotion(gmTag, 'Plain', signal, tsTag)
                    ops.imposedMotion(SupportNode, i, gmTag)

                tsTag += 1
                pTag += 1
                gmTag += 1

            return tsTag, pTag, gmTag

        def create_th_dict(GM_components, GMdt, GM_direct):
            # Create time history dictionary for multi-support excitation case
            excel_file = os.path.join(GM_direct, 'MultiSupport_Excitation.xlsx')
            xlsx = pd.ExcelFile(excel_file)
            th_dict = {}
            for sheet in xlsx.sheet_names:
                data = pd.read_excel(xlsx, sheet)
                th_dict[sheet] = []
                for k in range(data.shape[0]):
                    GMs = [data['H1'][k], data['H2'][k], data['V'][k]]
                    gm_mat, GM_dur, GMdt = get_time_series(GM_components, GMs, GM_direct, GMdt)
                    th_dict[sheet].append(gm_mat)

            return th_dict, GM_dur, GMdt

        print("#########################################################################")
        print("          Performing Nonlinear Response History Analysis (NRHA)...       ")
        print("#########################################################################")

        #  ----------------------------------------------------------------------------
        #  Eigenvalue Analysis - Damping Definition
        #  ----------------------------------------------------------------------------
        if damping == 'Modal':
            self._eigen(len(xi_modal))
            ops.modalDamping(*xi_modal)

        else:
            Mass_flag = 1
            K_comm_flag = 1
            K_init_flag = 0
            K_curr_flag = 0

            if damping == 'Rayleigh':
                # Compute the Rayleigh damping
                numEigen = int(max(Modes))
                Lambda = self._eigen(numEigen)
                Omega = Lambda ** 0.5
                wi = Omega[Modes[0] - 1]
                wj = Omega[Modes[1] - 1]
                a0 = 2.0 * xi * wi * wj / (wi + wj)
                a1 = 2.0 * xi / (wi + wj)

            elif damping == 'Stiffness':
                Mass_flag = 0
                Lambda = self._eigen(Modes)
                Omega = Lambda ** 0.5
                a0 = 0
                a1 = 2.0 * xi / Omega[Modes - 1]

            elif damping == 'Mass':
                K_comm_flag = 0
                Lambda = self._eigen(Modes)
                Omega = Lambda ** 0.5
                a0 = 2.0 * xi / Omega[Modes - 1]
                a1 = 0

            alphaM = a0 * Mass_flag  # Mass-proportional damping coefficient
            betaK_curr = a1 * K_curr_flag  # tangent-stiffness proportional damping
            betaK_init = a1 * K_init_flag  # initial-stiffness proportional damping
            betaK_comm = a1 * K_comm_flag  # Last committed-stiffness proportional damping
            ops.rayleigh(alphaM, betaK_curr, betaK_init, betaK_comm)

        #  ----------------------------------------------------------------------------
        #  Nonlinear Response History Analysis
        #  ----------------------------------------------------------------------------
        self._config()

        tsTag = 2
        pTag = 2
        gmTag = 1
        GM_direct = os.path.join('GMfiles', 'NRHA')

        if excitation == 'Uniform':  # Uniform excitation case
            if GMs is None:
                print('Assign ground motion files to use: GMs')
                sys.exit()
            gm_mat, GM_dur, GMdt = get_time_series(GM_components, GMs, GM_direct, GMdt)
            tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag)

        elif excitation == 'Multi-Support':  # Multi-support excitation case
            th_dict, GM_dur, GMdt = create_th_dict(GM_components, GMdt, GM_direct)
            # Apply excitation on bent nodes
            for n in range(self.num_bents):
                if self.model['Bent_Foundation']['Type'] == 'Fixed':
                    gm_mat = th_dict['Bent' + str(n + 1)][0]
                    for SupportNode in self.fixed_BentNodes[n]:
                        tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

                elif self.model['Bent_Foundation']['Type'] == 'Springs':
                    gm_mat = th_dict['Bent' + str(n + 1)][0]
                    SupportNode = self.fixed_BentNodes[n]
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

                elif self.model['Bent_Foundation']['Type'] == 'Pile-Shaft':
                    count = 0
                    for gm_mat in th_dict['Bent' + str(n + 1)]:
                        for k in range((self.model['Bent']['N'])):
                            SupportNode = self.fixed_BentNodes[n][k][count]
                            tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)
                        count += 1

                elif self.model['Bent_Foundation']['Type'] == 'Group Pile':
                    count = 0
                    for gm_mat in th_dict['Bent' + str(n + 1)]:
                        for k in range(len(self.fixed_BentNodes[n])):
                            SupportNode = self.fixed_BentNodes[n][k][count]
                            tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)
                        count += 1

            if self.model['Abutment_BackFill']['Type'] == 'None' and \
                    self.model['Abutment_Foundation']['Type'] == 'Fixed':
                # Apply excitation on abutment1 nodes
                gm_mat = th_dict['Abutment1'][0]
                for SupportNode in self.fixed_AB1Nodes:
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

                # Apply excitation on abutment2 nodes
                gm_mat = th_dict['Abutment2'][0]
                for SupportNode in self.fixed_AB2Nodes:
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

            elif self.model['Abutment_BackFill']['Type'] != 'None':
                # Apply excitation on abutment1 nodes
                gm_mat = th_dict['Abutment1_Backfill'][0]
                for SupportNode in self.fixed_AB1Nodes_backfill:
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

                # Apply excitation on abutment2 nodes
                gm_mat = th_dict['Abutment2_Backfill'][0]
                for SupportNode in self.fixed_AB2Nodes_backfill:
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

            if self.model['Abutment_Foundation']['Type'] == 'Springs':
                # Apply excitation on abutment1 nodes
                gm_mat = th_dict['Abutment1_Foundation'][0]
                for SupportNode in self.fixed_AB1Nodes_found:
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

                # Apply excitation on abutment2 nodes
                gm_mat = th_dict['Abutment2_Foundation'][0]
                for SupportNode in self.fixed_AB2Nodes_found:
                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

            elif self.model['Abutment_Foundation']['Type'] == 'Group Pile':
                count = 0
                for gm_mat in th_dict['Abutment1_Foundation']:
                    for k in range((len(self.fixed_AB1Nodes_found))):
                        SupportNode = self.fixed_AB1Nodes_found[k][count]
                        tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)
                    count += 1

                count = 0
                for gm_mat in th_dict['Abutment2_Foundation']:
                    for k in range((len(self.fixed_AB2Nodes_found))):
                        SupportNode = self.fixed_AB2Nodes_found[k][count]
                        tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)
                    count += 1

        tFinal = GM_dur + tFree  # Duration of analysis
        DtAnalysis = GMdt * DtFactor  # Default value is ground motion time step

        # Perform structural analysis
        if self.animate == 1:
            create_outdir(self.animation_dir)
            Recorders.animation(self.animation_dir)

        Analysis.nrha_single(self, DtAnalysis, tFinal, Dc,
                             os.path.join(self.out_dir, 'NRHA_Summary.txt'), pFlag)

        # Post Processing
        if self.animate == 1:
            Rendering.animate_nrha(self)

    def msa(self, gm_msa, damping='Stiffness', GMangle=0,
            Modes=1, xi=0.02, xi_modal=None, Dc=10, tFree=0,
            excitation='Uniform', signal='-accel', ScaleFactor=1.0):
        """
        -----------------------------------------------
        -- Script to Conduct Multple-Stripe Analysis --
        -----------------------------------------------
        MSA can be carried out using more than one gm components, 
        analysis directions are set by the user (MSAcomponents)
        If only single processor is used, gmFol can be set only for 
        processor ID:0, since others are not used.
        If multi processors are used, each processor will carry out the 
        analyses for specified ground motion sets.
        Therefore, EDP outputs will be written for those ground motion set.

        ----------------------------------------------
        Nonlinear Response History Analysis Parameters
        ----------------------------------------------
        GMangle = 0                         Ground motion incidence angle, going to be applied on gm components 1 and 2
        damping = 'Rayleigh'                Specify the damping type to use ('Rayleigh','Stiffness','Mass','Modal')
        xi_modal = [0.05, 0.02]             Damping values to assign for the first n modes (Modal damping)
        xi = 0.02                           Damping value (Rayleigh damping)
        Modes = [1,3]                       List of modes to calculate damping coefficients, necessary for
                                            damping = 'Rayleigh'. For 'Stiffness','Mass' this is an integer.
        Dc = 10.0                           Drift capacity (%) to define local collapse in NRHA - Analysis stops
                                            if pier reaches this drift level
        tFree = 0.0                         Additional free vibration time
        excitation = 'Uniform'              Excitation pattern applied ('Uniform','Multi-Support')
        signal = '-accel'                   Type of time series (-accel, -vel, -disp) to apply. Then the input file must
                                            be defined accordingly. Note that in case of Uniform excitation pattern
                                            signal can only be acceleration.
        -----------------------------------------------
        MSA input parameters (gm_msa, dictionary)
        -----------------------------------------------        
        'Folder': 'P0',                                set the ground motion records to be used for each processor (GMFiles/MSA/Folder)
        'MSAcomponents': [1,2,3]                       Ground motion components to use in the analysis
        'dts_file': "GMR_dts.txt",                     Time steps of ground motions to run
        'gm_H1_names_file':   "GMR_H1_names.txt",      Names of ground motions to run (dir 1)
        'gm_H2_names_file':   "GMR_H2_names.txt",      Names of ground motions to run (dir 2)
        'gm_V_names_file':    "GMR_V_names.txt",       Names of ground motions to run (dir 3)
        'gm_multi_support':   "GMR_multi_support.txt"  GMR folder names for multi-support excitation case
         Excel File  MultiSupport_Excitation.xlsx containts the ground motion information for multi-support excitation case.

        Returns
        -------
        None.

        """

        def get_time_series(GM_components, GMs, GM_direct, GMdt):
            # Get gm components to use
            nPts1 = 0
            nPts2 = 0
            nPtsV = 0
            for i in range(len(GM_components)):
                gm_path = os.path.join(GM_direct, GMs[i])  # Assuming GM is in PEER format

                if GM_components[i] == 1:
                    if gm_path[-4:] == '.AT2':
                        GMdt, _, _, _, gmH1_tmp = ReadRecord(gm_path)
                    else:
                        gmH1_tmp = np.loadtxt(gm_path)
                    nPts1 = len(gmH1_tmp)

                if GM_components[i] == 2:
                    if gm_path[-4:] == '.AT2':
                        GMdt, _, _, _, gmH2_tmp = ReadRecord(gm_path)
                    else:
                        gmH2_tmp = np.loadtxt(gm_path)
                    nPts2 = len(gmH2_tmp)

                if GM_components[i] == 3:
                    if gm_path[-4:] == '.AT2':
                        GMdt, _, _, _, gmV_tmp = ReadRecord(gm_path)
                    else:
                        gmV_tmp = np.loadtxt(gm_path)
                    nPtsV = len(gmV_tmp)

            # Assign zero if gm component is not used
            nPts = max(nPts1, nPts2, nPtsV)
            GM_dur = GMdt * nPts
            gmH1 = np.zeros(nPts)
            gmH2 = np.zeros(nPts)
            gmV = np.zeros(nPts)
            try:
                gmH1[:nPts1] = gmH1_tmp
            except:
                pass
            try:
                gmH2[:nPts2] = gmH2_tmp
            except:
                pass
            try:
                gmV[:nPtsV] = gmV_tmp
            except:
                pass

            # Modify the ground motions according to the angle of incidence
            gmH1.shape = (1, nPts)
            gmH2.shape = (1, nPts)
            gmV.shape = (1, nPts)
            gm_mat = np.vstack((gmH1, gmH2))
            theta_mat = np.array([[np.cos(GMangle * np.pi / 180), np.sin(GMangle * np.pi / 180)],
                                  [-np.sin(GMangle * np.pi / 180), np.cos(GMangle * np.pi / 180)]])
            gm_mat = np.vstack((theta_mat @ gm_mat, gmV))

            return gm_mat, GM_dur, GMdt

        def Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode=None):
            # Define the Time Series and and the Load Pattern
            for i in range(1, 4):
                # Setting time series
                gm_i = gm_mat[i - 1, :]
                ops.timeSeries('Path', tsTag, '-dt', GMdt, '-values', *gm_i.tolist(), '-factor', ScaleFactor)

                # Creating UniformExcitation load pattern
                if excitation == 'Uniform':
                    ops.pattern('UniformExcitation', pTag, i, '-accel', tsTag)

                elif excitation == 'Multi-Support':
                    # Creating MultipleSupport Excitation load pattern                        
                    ops.pattern('MultipleSupport', pTag)
                    ops.groundMotion(gmTag, 'Plain', signal, tsTag)
                    ops.imposedMotion(SupportNode, i, gmTag)

                tsTag += 1
                pTag += 1
                gmTag += 1

            return tsTag, pTag, gmTag

        def create_th_dict(GM_components, GMdt, GM_direct):
            # Create time history dictionary for multi-support excitation case
            excel_file = os.path.join(GM_direct, 'MultiSupport_Excitation.xlsx')
            xlsx = pd.ExcelFile(excel_file)
            th_dict = {}
            for sheet in xlsx.sheet_names:
                data = pd.read_excel(xlsx, sheet)
                th_dict[sheet] = []
                for k in range(data.shape[0]):
                    GMs = [data['H1'][k], data['H2'][k], data['V'][k]]
                    gm_mat, GM_dur, GMdt = get_time_series(GM_components, GMs, GM_direct, GMdt)
                    th_dict[sheet].append(gm_mat)

            return th_dict, GM_dur, GMdt

        gmMSA = os.path.join('GMfiles', 'MSA', gm_msa['Folder'])
        direc = os.path.join(self.out_dir, 'MSA')
        create_outdir(direc)

        if ops.getNP() == 1:
            print("\
#########################################################################\n\
          Performing Multiple Stripes Analysis (MSA)...              \n\
#########################################################################")
        else:
            self.analysis_config(self.constraintType,
                                 'Parallel' + self.numbererType, 'Mumps')

        gm_sets = []
        for file in os.listdir(gmMSA):
            if not file.startswith('.'): gm_sets.append(file)
        for gm_set in gm_sets:  # loop for each ground motion set

            gm_dir = os.path.join(gmMSA, gm_set)
            out_dir = os.path.join(direc, gm_set)
            logfile_path = os.path.join(out_dir, 'log.txt')
            create_outdir(out_dir)
            log = open(logfile_path, 'w')  # create a log file for each set
            log.write(program_info())

            # Try changing DtFactors if the analysis does not converge
            DtFactors = [1.0]
            for i in range(3):
                DtFactors.append(DtFactors[-1] * 0.5)

            dt_path = os.path.join(gm_dir, gm_msa['dts_file'])
            dts = np.loadtxt(dt_path)
            try:
                num_gms = len(dts)
            except:  # If a single record exists in gm set
                num_gms = 1
                dts = [float(dts)]
            # Set up the error log to write to
            error_log = ["List of warning and errors encountered:"]

            # Initalize EDPs
            edp = {}  # this the edp dictionary where the all related results will be saved

            # 1) Piers
            num_piers = self.num_bents * self.model['Bent']['N']
            # Initialise the array of IM levels
            edp['max_drift'] = np.zeros((num_gms, num_piers))  # Peak Column Drift Ratio
            edp['mu_curv'] = np.zeros((num_gms, num_piers))  # Peak Curvature Ductility
            edp['mu_disp'] = np.zeros((num_gms, num_piers))  # Peak Displacement Ductility

            # 2) Abutments
            edp['abut_disp'] = np.zeros((num_gms, 2))  # Peak Peak Displacement of Central Abutment Nodes

            # 3) Bearings
            # 4) Foundation etc..

            # Open and load the ground motions
            if excitation == 'Uniform':  # Uniform Excitation Case
                GM_uniform = []
                if 1 in gm_msa['MSAcomponents']:
                    names_h1_path = os.path.join(gm_dir, gm_msa['gm_H1_names_file'])
                    with open(names_h1_path) as inputfile:
                        gm_h1_names = [line.rstrip() for line in inputfile]
                    GM_uniform.append(gm_h1_names)
                if 2 in gm_msa['MSAcomponents']:
                    names_h2_path = os.path.join(gm_dir, gm_msa['gm_H2_names_file'])
                    with open(names_h2_path) as inputfile:
                        gm_h2_names = [line.rstrip() for line in inputfile]
                    GM_uniform.append(gm_h2_names)
                if 3 in gm_msa['MSAcomponents']:
                    names_v_path = os.path.join(gm_dir, gm_msa['gm_V_names_file'])
                    with open(names_v_path) as inputfile:
                        gm_v_names = [line.rstrip() for line in inputfile]
                    GM_uniform.append(gm_v_names)

            elif excitation == 'Multi-Support':  # Multi-Support Excitation Case
                folds_path = os.path.join(gm_dir, gm_msa['gm_multi_support'])
                with open(folds_path) as inputfile:
                    gm_multi_names = [line.rstrip() for line in inputfile]

            # Loop through each ground motion
            for iii in np.arange(num_gms):
                # Get the ground motion
                GMdt = dts[iii]

                if excitation == 'Uniform':
                    GMs = [gm_names[iii] for gm_names in GM_uniform]
                    gm_mat, GM_dur, GMdt = get_time_series(gm_msa['MSAcomponents'], GMs, gm_dir, GMdt)

                elif excitation == 'Multi-Support':
                    gm_dir2 = os.path.join(gm_dir, gm_multi_names[iii])
                    th_dict, GM_dur, GMdt = create_th_dict(gm_msa['MSAcomponents'], GMdt, gm_dir2)

                tFinal = GM_dur + tFree  # Duration of analysis
                gm_text = 'GM set: %s, GM no: %d' % (gm_set, iii + 1)

                if ops.getNP() == 1:
                    print('Running ' + gm_text + '...')

                cIndex = -1
                cLoop = 0
                # Lets try to reduce the time step if convergence is not satisfied, and re-run the analysis
                while cIndex == -1 and cLoop < len(DtFactors):
                    if cLoop != 0: error_log.append(
                        'While running ' + gm_text + ' analysis failed to converge, reducing the analysis time step...')
                    DtAnalysis = GMdt * DtFactors[cLoop]
                    cLoop += 1
                    #  ----------------------------------------------------------------------------
                    #  Gravity Analysis
                    #  ----------------------------------------------------------------------------
                    startT = Get_T0()
                    self.wipe_model()
                    self._build()
                    self.gravity(pflag=0)
                    #  ----------------------------------------------------------------------------
                    #  Eigenvalue Analysis - Damping Definition
                    #  ----------------------------------------------------------------------------
                    if damping == 'Modal':
                        self._eigen(len(xi_modal))
                        ops.modalDamping(*xi_modal)

                    else:
                        Mass_flag = 1
                        K_comm_flag = 1
                        K_init_flag = 0
                        K_curr_flag = 0

                        if damping == 'Rayleigh':
                            # Compute the Rayleigh damping
                            numEigen = int(max(Modes))
                            Lambda = self._eigen(numEigen)
                            Omega = Lambda ** 0.5
                            wi = Omega[Modes[0] - 1]
                            wj = Omega[Modes[1] - 1]
                            a0 = 2.0 * xi * wi * wj / (wi + wj)
                            a1 = 2.0 * xi / (wi + wj)

                        elif damping == 'Stiffness':
                            Mass_flag = 0
                            Lambda = self._eigen(Modes)
                            Omega = Lambda ** 0.5
                            a0 = 0
                            a1 = 2.0 * xi / Omega[Modes - 1]

                        elif damping == 'Mass':
                            K_comm_flag = 0
                            Lambda = self._eigen(Modes)
                            Omega = Lambda ** 0.5
                            a0 = 2.0 * xi / Omega[Modes - 1]
                            a1 = 0

                        alphaM = a0 * Mass_flag  # Mass-proportional damping coefficient
                        betaK_curr = a1 * K_curr_flag  # tangent-stiffness proportional damping
                        betaK_init = a1 * K_init_flag  # initial-stiffness proportional damping
                        betaK_comm = a1 * K_comm_flag  # Last committed-stiffness proportional damping
                        ops.rayleigh(alphaM, betaK_curr, betaK_init, betaK_comm)

                    #  ----------------------------------------------------------------------------
                    #  Nonlinear Response History Analysis
                    #  ----------------------------------------------------------------------------  
                    self._config()  # Configure the analysis parameters

                    # Define the Time Series and and the Load Pattern
                    tsTag = 2
                    pTag = 2
                    gmTag = 1
                    if excitation == 'Uniform':
                        tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag)

                    elif excitation == 'Multi-Support':  # Multi-support excitation case
                        # Apply excitation on bent nodes
                        for n in range(self.num_bents):
                            if self.model['Bent_Foundation']['Type'] == 'Fixed':
                                gm_mat = th_dict['Bent' + str(n + 1)][0]
                                for SupportNode in self.fixed_BentNodes[n]:
                                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                             SupportNode)

                            elif self.model['Bent_Foundation']['Type'] == 'Springs':
                                gm_mat = th_dict['Bent' + str(n + 1)][0]
                                SupportNode = self.fixed_BentNodes[n]
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag, SupportNode)

                            elif self.model['Bent_Foundation']['Type'] == 'Pile-Shaft':
                                count = 0
                                for gm_mat in th_dict['Bent' + str(n + 1)]:
                                    for k in range((self.model['Bent']['N'])):
                                        SupportNode = self.fixed_BentNodes[n][k][count]
                                        tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                                 SupportNode)
                                    count += 1

                            elif self.model['Bent_Foundation']['Type'] == 'Group Pile':
                                count = 0
                                for gm_mat in th_dict['Bent' + str(n + 1)]:
                                    for k in range(len(self.fixed_BentNodes[n])):
                                        SupportNode = self.fixed_BentNodes[n][k][count]
                                        tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                                 SupportNode)
                                    count += 1

                        if self.model['Abutment_BackFill']['Type'] == 'None' and \
                                self.model['Abutment_Foundation']['Type'] == 'Fixed':
                            # Apply excitation on abutment1 nodes
                            gm_mat = th_dict['Abutment1'][0]
                            for SupportNode in self.fixed_AB1Nodes:
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                         SupportNode)

                            # Apply excitation on abutment2 nodes
                            gm_mat = th_dict['Abutment2'][0]
                            for SupportNode in self.fixed_AB2Nodes:
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                         SupportNode)

                        elif self.model['Abutment_BackFill']['Type'] != 'None':
                            # Apply excitation on abutment1 nodes
                            gm_mat = th_dict['Abutment1_Backfill'][0]
                            for SupportNode in self.fixed_AB1Nodes_backfill:
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                         SupportNode)

                            # Apply excitation on abutment2 nodes
                            gm_mat = th_dict['Abutment2_Backfill'][0]
                            for SupportNode in self.fixed_AB2Nodes_backfill:
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                         SupportNode)

                        if self.model['Abutment_Foundation']['Type'] == 'Springs':
                            # Apply excitation on abutment1 nodes
                            gm_mat = th_dict['Abutment1_Foundation'][0]
                            for SupportNode in self.fixed_AB1Nodes_found:
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                         SupportNode)

                            # Apply excitation on abutment2 nodes
                            gm_mat = th_dict['Abutment2_Foundation'][0]
                            for SupportNode in self.fixed_AB2Nodes_found:
                                tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag, gmTag,
                                                                         SupportNode)

                        elif self.model['Abutment_Foundation']['Type'] == 'Group Pile':
                            count = 0
                            for gm_mat in th_dict['Abutment1_Foundation']:
                                for k in range((len(self.fixed_AB1Nodes_found))):
                                    SupportNode = self.fixed_AB1Nodes_found[k][count]
                                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag,
                                                                             gmTag, SupportNode)
                                count += 1

                            count = 0
                            for gm_mat in th_dict['Abutment2_Foundation']:
                                for k in range((len(self.fixed_AB2Nodes_found))):
                                    SupportNode = self.fixed_AB2Nodes_found[k][count]
                                    tsTag, pTag, gmTag = Create_Load_Pattern(gm_mat, GMdt, tsTag, pTag,
                                                                             gmTag, SupportNode)
                                count += 1

                    Mdrft, cIndex, mpier, mdrft, mudisp, muK, anlys, abutdisps = \
                        Analysis.nrha_multiple(self, DtAnalysis, tFinal, Dc, '',  0)
                time_text = RunTime(startT)
                log.write(gm_text + '\n')
                log.write(anlys + '\n')
                log.write(time_text + '\n')
                log.write(end_text + "\n")

                if ops.getNP() == 1:
                    print(time_text)
                    print(end_text)

                # Save edp values
                edp['max_drift'][iii, :] = mdrft
                edp['mu_disp'][iii, :] = mudisp
                edp['mu_curv'][iii, :] = muK
                edp['abut_disp'][iii, :] = abutdisps

            for error in error_log:  # add the error log
                log.write('%s\n' % error)
            log.close()

            # Save the results
            np.savetxt(os.path.join(out_dir, 'DriftRat.txt'), edp['max_drift'])
            np.savetxt(os.path.join(out_dir, 'CurvDuct.txt'), edp['mu_curv'])
            np.savetxt(os.path.join(out_dir, 'DispDuct.txt'), edp['mu_disp'])
            np.savetxt(os.path.join(out_dir, 'AbutDisp.txt'), edp['abut_disp'])
            with open(os.path.join(out_dir, 'EDP.pkl'), 'wb') as handle:
                pickle.dump(edp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if ops.getNP() == 1:
            print("\
#########################################################################\n\
                Multiple Stripes Analysis (MSA) is Completed!            \n\
#########################################################################")

    def ida_htf(self, htf, gm_ida, im, gmFol, IDAdir, damping='Stiffness',
                Modes=1, xi=0.02, xi_modal=None, Dc=10, DtFactor=1):
        """
        TODO: Currently, this option is implemented only for uni-directional analysis. Moreover, 
        there is no option for multiple-support excitation, hence only available option is to use
        uniform excitation.
        --------------------------------------------------------------------------------------------------
        -- Script to Conduct Incremental Dynamic Analysis Using Hunt, Trace and Fill Algorithm -----------
        --------------------------------------------------------------------------------------------------
        This is a script that will conduct an Incremental Dynamic Analysis (IDA) of a given structure
        using hunting, tracing and filling (HTF), where the user needs to provide a list of ground motion
        records and specify the increments, steps and maximum number of runs to conduct per record.
        
        The algorithm is inspired by that described in the Vamvatsikos & Cornell [2004] paper in
        Earthquake Spectra, but the main difference here is that the script is entirely python-based. This
        means that the procedure does not need Matlab to work, so the back and forth between Matlab and
        OpenSees during analysis is removed. Moreover, OpenSeespy (pythonic version of OpenSees) can be used
        directly. The number of inputs is also reduced where just the initial intensity, the increment by 
        which to increment the record and the maximum number of runs to be conducted per record are specified.
        
        The algorithm works by conducting an initial "hunting" phase, where the record is scaled up
        relatively quickly to find a collapse - hence, we are hunting out the collapse. This collapsed run
        is then discarded and this run is re-run at an intensity of the last non-collapsing run plus a
        fraction of the difference between the hunted collapse's intensity and the last non-collapsing
        intensity. This fraction can be modified in the code (iml_trace_incr). Once we go
        back to the last non-collapsing run and start slowly incrementing, we are in the "tracing" phase
        of the algorithm. Once collapse has been found from tracing, the remainder of the available runs
        are then used to go back and fill in the biggest gaps in the hunting phase's intensity steps,
        which is known as the "filling" phase.
        
        Collapse Index (ColIndex = 1 or 0) must be determined after running the model. This limit
        is determined by the user
        Likewise EDP's must be recorded, which is d_max (but could be more) after each run. 
        
        More intensity measures can be included. In the current version there is only PGA and SA(T)
        
        Some of the warning outputs:
            "----- WARNING: Collapsed achieved on first increment, reduce increment..."
            This means that the first step following the initial elastic run resuted in a collapse. In
            short, you have well over-shot the runway with the increment step. This is the second run,
            but if the building is collapsing on the very first run (i.e. the user defined intensity),
            well it seems you have bigger problems.
        
            "--- WARNING: First trace for collapse resulted in collapse..."
            Once the collapse has been hunted out, we go back to tracing the collapse more carefully.
            If this first trace results in collapse, either the last non-collapsing run was quite close
            to the collapsing intensity or the
            fraction of the difference is too big. Ideally, we would like to reduce this fraction
            automatically, but it's a bitch to code at the minute. Flagged for follow-up.....
        
            "----- WARNING: Collapse not achieved, increase increment or number of runs..."
            This means the collapse hasn't been hunted out yet, so either the incrementing to reach
            collapse is increased	or the maximum nuber of runs at the current increment is increased
            to allow it to go further. This warning	could be used as a way of gauging when collapse
            occurs, such that the max runs can be specified a priori such that there are enough runs left
            for tracing and filling.
        
            "----- WARNING: No filling, algorithm still tracing for collapse (reduce increment & increase runs)..."
            The algorithm is still tracing for collapse, either because there are not enough runs or
            because the increment used during the hunting was too big with respect to the fraction of
            the difference being used to trace.

        IDA can be carried out using single gm component in one of the global axes, 
        the analysis direction is set by the user (IDAdir)
        If only single processor is used, gmFol can be set only for processor ID:0, since others are not used.
        If multi processors are used, each processor will carry out IDA for specified ground motion sets.
        Therefore, IM and EDP outputs will be written for those ground motion set.
        mpi4py must be installed to run using parallel processors.
        The command to run is: mpiexec -np np python filename.py

        ----------------------------------------------
        Nonlinear Response History Analysis Parameters
        ----------------------------------------------
        damping = 'Rayleigh'                Specify the damping type to use ('Rayleigh','Stiffness','Mass','Modal')
        xi_modal = [0.05, 0.02]             Damping values to assign for the first n modes (Modal damping)
        xi = 0.02                           Damping value (Rayleigh damping)
        Modes = [1,3]                       List of modes to calculate damping coefficients, necessary for
                                            damping = 'Rayleigh'. For 'Stiffness','Mass' this is an integer.
        Dc = 10.0                           Drift capacity (%) to define local collapse in NRHA - Analysis stops
                                            if pier reaches this drift level
        tFree = 0.0                         Additional free vibration time
        DtFactor = 1.0		                Initial analysis time step is determined as a (DtFactor * GMdt). For example,
                                            if DtFactor = 1.0, analysis time step is equal to ground motion time step
        
        -----------------------------------------------
        IDA input parameters
        -----------------------------------------------
        The ground motion records to be used for each processor
        gmFol = os.path.join('IDA','P'+pID) 
        
        IDAdir = 2 # Directions to apply ground motions
        im = {'im': 'SaT', 'Tstar': 1.4, 'xi': 0.05}
        htf = {'num_runs': 20,
                'iml_init': 0.05,            # initial hunt ratio
                'iml_hunt_incr': 0.20,       # hunt increment ratio
                'iml_trace_incr': 0.10,      # trace increment ratio
                'iml_min_trace': 0.0,        # minimum trace incr ratio
                }
        gm_ida = {'gm_names_file':  "GMR_names.txt", # Names of ground motions to run
                  'dts_file':       "GMR_dts.txt",   # Time steps of ground motions to run
                  'durs_file':      "GMR_durs.txt",  # Durations of the ground motions to run
                  }
        

        """

        # set the ground motion records to be used for each processor
        gmIDA = os.path.join('GMfiles', gmFol)

        # Set the top and bottom nodes for piers to calculate peak interstorey drift ratio
        PierStartNodes = []
        PierEndNodes = []
        for i in range(len(self.BentEndNodes)):
            for node in self.BentEndNodes[i]:
                PierEndNodes.append(node)
            for node in self.BentStartNodes[i]:
                PierStartNodes.append(node)

        direc = os.path.join(self.out_dir, 'IDA')
        create_outdir(direc)

        pid = ops.getPID()
        logfile_path = os.path.join(direc, str(pid) + '_log.txt')
        log = open(logfile_path, 'w')
        txtstr = program_info()
        log.write(txtstr)

        if ops.getNP() == 1:
            print("\
#########################################################################\n\
              Performing Incremental Dynamic Analysis (IDA)...           \n\
#########################################################################")
        else:
            self.analysis_config(self.constraintType,
                                 'Parallel' + self.numbererType, 'Mumps')
            log.write("This is processor %s\nPerforming Incremental Dynamic Analysis (IDA)...\n" % str(pid))

        def analyze(gm, IDAdir):
            #  ----------------------------------------------------------------------------
            #  Gravity Analysis
            #  ----------------------------------------------------------------------------
            startT = Get_T0()
            self.wipe_model()
            self._build()
            self.gravity(pflag=0)

            #  ----------------------------------------------------------------------------
            #  Eigenvalue Analysis - Damping Definition
            #  ----------------------------------------------------------------------------
            if damping == 'Modal':
                self._eigen(len(xi_modal))
                ops.modalDamping(*xi_modal)

            else:
                Mass_flag = 1
                K_comm_flag = 1
                K_init_flag = 0
                K_curr_flag = 0

                if damping == 'Rayleigh':
                    # Compute the Rayleigh damping
                    numEigen = int(max(Modes))
                    Lambda = self._eigen(numEigen)
                    Omega = Lambda ** 0.5
                    wi = Omega[Modes[0] - 1]
                    wj = Omega[Modes[1] - 1]
                    a0 = 2.0 * xi * wi * wj / (wi + wj)
                    a1 = 2.0 * xi / (wi + wj)

                elif damping == 'Stiffness':
                    Mass_flag = 0
                    Lambda = self._eigen(Modes)
                    Omega = Lambda ** 0.5
                    a0 = 0
                    a1 = 2.0 * xi / Omega[Modes - 1]

                elif damping == 'Mass':
                    K_comm_flag = 0
                    Lambda = self._eigen(Modes)
                    Omega = Lambda ** 0.5
                    a0 = 2.0 * xi / Omega[Modes - 1]
                    a1 = 0

                alphaM = a0 * Mass_flag  # Mass-proportional damping coefficient
                betaK_curr = a1 * K_curr_flag  # tangent-stiffness proportional damping
                betaK_init = a1 * K_init_flag  # initial-stiffness proportional damping
                betaK_comm = a1 * K_comm_flag  # Last committed-stiffness proportional damping
                ops.rayleigh(alphaM, betaK_curr, betaK_init, betaK_comm)

            #  ----------------------------------------------------------------------------
            #  Nonlinear Response History Analysis
            #  ----------------------------------------------------------------------------  
            self._config()

            # Define the Time Series and and the Load Pattern
            tsTag = 2
            pTag = 2
            g = 9.81
            dt = gm['dt']
            tFinal = gm['dur']
            GMfatt = gm['sf'] * g
            record = gm['file_name']
            ops.timeSeries('Path', tsTag, '-dt', dt, '-filePath', record, '-factor', GMfatt)
            ops.pattern('UniformExcitation', pTag, IDAdir, '-accel', tsTag)

            # !!! right now EDP is pier drift
            tNode = PierStartNodes
            bNode = PierEndNodes
            # Run analysis
            DtAnalysis = dt * DtFactor
            mdrft, cIndex, mpier, anlys = Analysis.nrha_single(self, DtAnalysis, tFinal, Dc, tNode, bNode, '', 0)

            self.wipe_model()

            time_text = RunTime(startT)
            log.write(anlys + '\n')
            log.write(time_text + '\n')
            log.write(end_text + '\n')

            if ops.getNP() == 1:
                print(time_text)
                print(end_text)
            pass

            return mdrft, cIndex

        DtFactor_init = DtFactor
        iml_hunt_incr_init = htf['iml_hunt_incr']
        iml_trace_incr_init = htf['iml_trace_incr']
        fc_flag = 0  # if this is 1, it means that structure collapsed while filling
        temp_factor = 1

        names_path = os.path.join(gmIDA, gm_ida['gm_names_file'])
        dt_path = os.path.join(gmIDA, gm_ida['dts_file'])
        durs_path = os.path.join(gmIDA, gm_ida['durs_file'])

        # Open and load the ground motions
        with open(names_path) as inputfile:
            gm_names = [line.rstrip() for line in inputfile]

        dts = np.loadtxt(dt_path)
        durs = np.loadtxt(durs_path)
        try:
            num_gms = len(dts)
        except:  # IDA for set with single record
            num_gms = 1
            dts = [float(dts)]
            durs = [float(durs)]

        # Set up the error log to write to
        error_log = ["List of warning and errors encountered:"]

        # Initialise the array of IM levels
        iml = np.zeros((num_gms, htf['num_runs'] + 1))
        d_max = np.zeros((num_gms, htf['num_runs'] + 1))

        iml_sorted = np.zeros((num_gms, htf['num_runs'] + 1))
        d_max_sorted = np.zeros((num_gms, htf['num_runs'] + 1))

        # Loop through each ground motion
        for iii in np.arange(num_gms):
            # Get the ground motion

            gm = {
                'file_name': os.path.join(gmIDA, gm_names[iii]),
                'dt': dts[iii],
                'dur': durs[iii],
                'sf': 0.0,
            }
            # !!! right now analyis is carried out for single direction
            # Identify the IM and the iml of the ground motion
            if im['im'] == 'pga':
                iml_curr_gm = np.max(np.abs(np.loadtxt(gm['file_name'])))
            elif im['im'] == 'SaT':
                _, _, _, ac_tot = sdof_ltha(np.loadtxt(gm['file_name']), dts[iii], np.array([im['Tstar']]), im['xi'], 1)
                iml_curr_gm = np.max(np.abs((ac_tot)), axis=0)[0]
            else:
                print("Cannot find the intensity measure: " + im['im'])
                sys.exit()

            # Set up some parameters
            jjj = 1  # Initialise the list of IM used for printing
            hFlag = 1  # Hunting flag (1 for when we're hunting)
            tFlag = 0  # Tracing flag (0 at first)
            fFlag = 0  # Filling flag (0 at first)
            coll_index = 0  # Set the collapse index up initially

            # Start the process
            while jjj <= htf['num_runs']:

                # Use this gm and sf to analyse the structure
                run = 'Record_' + str(iii + 1) + '_Run_' + str(jjj + 1)

                ######### Hunting part            
                if hFlag == 1:
                    gm_text = "Hunting... GM: " + str(iii + 1) + "   Run: " + str(jjj)
                    if ops.getNP() == 1: print(gm_text)
                    log.write(gm_text + "\n")

                    if jjj == 1:
                        # We are running the first run so use im_init
                        iml[iii][jjj] = htf['iml_init']
                    elif jjj > 1:
                        # Now start to ramp it up!
                        iml[iii][jjj] = iml[iii][jjj - 1] + (jjj - 1) * htf['iml_hunt_incr']

                    # Determine the scale factor to be applied to the current gm to reach the iml required
                    gm['sf'] = iml[iii][jjj] / iml_curr_gm

                    # Build and run the OpenSees model
                    mdrft, cIndex = analyze(gm, IDAdir)
                    d_max[iii][jjj] = mdrft
                    coll_index = cIndex

                    # If we get a collapse
                    if coll_index == 1:
                        DtFactor = DtFactor_init  # reset DtFactor to initial choice
                        htf['iml_hunt_incr'] = iml_hunt_incr_init  # reset iml_hunt_incr to initial choice

                        hFlag = 0  # Hunting flag (Stop hunting)
                        tFlag = 1  # Tracing flag (Now we are tracing)
                        j_hunt = jjj  # The value that we hunted to

                        # Find the difference between the hunting collapse and the last stable point before it
                        iml_diff = iml[iii][j_hunt] - iml[iii][j_hunt - 1]

                        # Remove the last entry of iml
                        iml[iii][j_hunt] = 0.0
                        d_max[iii][j_hunt] = 0.0

                        # Check to see if the first hunt cause a collapse, this is very unlikely
                        if j_hunt == 1:
                            error_log.append("WARNING:  Run: " + run + ", File: " + gm_names[
                                iii] + " collapsed on first hunt, reduce the initial intensity...")
                            if ops.getNP() == 1: print(error_log[-1])

                    elif coll_index == -1:  # convergence problem
                        htf['iml_hunt_incr'] = htf['iml_hunt_incr'] * 0.9  # reduce the hunt incr
                        DtFactor = DtFactor * 0.5  # reduce analysis time step
                        # Remove the last entry of iml
                        iml[iii][jjj] = 0.0
                        d_max[iii][jjj] = 0.0
                    elif coll_index == 0:
                        # No collapse occurred so we can increment the counter and proceed hunting
                        jjj += 1

                ######### Tracing part
                elif tFlag == 1:
                    temp_conv = 0
                    gm_text = "Tracing... GM: " + str(iii + 1) + "   Run: " + str(jjj)
                    if ops.getNP() == 1: print(gm_text)
                    log.write(gm_text + '\n')

                    # Set the tracing iml increments, but set lower limit to avoid tracing too finely
                    iml_incr_trace = max(htf['iml_trace_incr'] * iml_diff, htf['iml_min_trace'])

                    # Get new iml and sf
                    iml[iii][jjj] = iml[iii][jjj - 1] + iml_incr_trace
                    gm['sf'] = iml[iii][jjj] / iml_curr_gm

                    # Build and run the OpenSees model
                    mdrft, cIndex = analyze(gm, IDAdir)
                    d_max[iii][jjj] = mdrft
                    coll_index = cIndex

                    # If we get a collapse
                    if coll_index == 1:
                        j_trace = jjj  # The value that we traced to
                        # Check to see if the first trace cause a collapse
                        if j_trace - j_hunt == 0:
                            error_log.append("WARNING:  Run: " + run + ", File: " + gm_names[
                                iii] + " collapsed on first trace, reducing the intensity increment...")
                            if ops.getNP() == 1: print(error_log[-1])
                            htf['iml_trace_incr'] = htf['iml_trace_incr'] * 0.8
                        else:
                            htf['iml_trace_incr'] = iml_trace_incr_init
                            DtFactor = DtFactor_init  # reset DtFactor to initial choice

                            # Set the demand as the max value
                            d_max[iii][jjj] = Dc

                            tFlag = 0  # Hunting flag (Stop tracing)
                            fFlag = 1  # Tracing flag (Now we are filling)
                            jjj += 1

                    elif coll_index == -1:  # There is a convergence problem!
                        if temp_conv < 2:  # first reduce the analysis time step
                            DtFactor = DtFactor * 0.5
                            # Remove the last entry of iml
                            temp_conv += 1
                        else:  # if this does not work twice reduce the trace incr
                            DtFactor = DtFactor_init;
                            temp_conv = 0
                            htf['iml_trace_incr'] = htf['iml_trace_incr'] * 0.8
                        iml[iii][jjj] = 0.0
                        d_max[iii][jjj] = 0.0

                    elif coll_index == 0:
                        # No collapse occurred so we can increment the counter and proceed
                        jjj += 1

                ######### Filling part
                elif fFlag == 1:
                    gm_text = "Filling... GM: " + str(iii + 1) + "   Run: " + str(jjj)
                    if ops.getNP() == 1: print(gm_text)
                    log.write(gm_text + '\n')

                    # Sort out the iml array
                    iml_srt = np.sort(iml[iii])

                    # Find the biggest gap
                    iml_gap = 0.0
                    for kkk in np.arange(htf['num_runs'] - 1):
                        temp = iml_srt[kkk + 1] - iml_srt[kkk]
                        if temp > iml_gap:
                            iml_gap = temp
                            iml_fill = iml_srt[kkk] + iml_gap * 0.5
                    if fc_flag == 1:  # this is very weird, it should not have collapsed
                        temp_factor = temp_factor * 0.9
                        iml_fill = iml_fill * temp_factor

                    # Add the new intensity to the iml array
                    iml[iii][jjj] = iml_fill
                    gm['sf'] = iml[iii][jjj] / iml_curr_gm

                    # Build and run the OpenSees model
                    mdrft, cIndex = analyze(gm, IDAdir)
                    d_max[iii][jjj] = mdrft
                    coll_index = cIndex

                    # If we get a collapse
                    if coll_index == 1:
                        error_log.append("WARNING:  Run: " + run + ", File: " + gm_names[
                            iii] + " collapsed while filling... changing iml_fill")
                        if ops.getNP() == 1: print(error_log[-1])
                        iml[iii][jjj] = 0.0
                        d_max[iii][jjj] = 0.0
                        fc_flag = 1
                    elif coll_index == -1:
                        DtFactor = DtFactor * 0.5  # just try decreasing the analysis time step
                        # Remove the last entry of iml
                        iml[iii][jjj] = 0.0
                        d_max[iii][jjj] = 0.0
                    elif coll_index == 0:
                        # No collapse occurred so we can increment the counter and proceed
                        DtFactor = DtFactor_init  # reset the analysis time step
                        jjj += 1
                        fc_flag = 0
                        temp_factor = 1

            DtFactor = DtFactor_init  # reset DtFactor to initial choice
            # Sort the ouputted arrays  
            iml_sorted[iii] = np.sort(iml[iii])
            d_max_sorted[iii] = [x for _, x in sorted(zip(iml[iii], d_max[iii]))]

        self.iml_IDA = iml_sorted
        self.edp_IDA = d_max_sorted

        if ops.getNP() == 1:
            print("\
#########################################################################\n\
             Incremental Dynamic Analysis (IDA) is Completed!            \n\
#########################################################################")
        else:
            print("Processor %d Completed Incremental Dynamic Analysis (IDA)!" % pid)
            log.write("Completed Incremental Dynamic Analysis (IDA)!\n")

        for error in error_log:
            log.write('%s\n' % error)
        np.savetxt(os.path.join(direc, str(pid) + '_IM.txt'), np.transpose(self.iml_IDA))
        np.savetxt(os.path.join(direc, str(pid) + '_EDP.txt'), np.transpose(self.edp_IDA))
        log.close()


def units(pFlag=0):
    return def_units(pFlag)
