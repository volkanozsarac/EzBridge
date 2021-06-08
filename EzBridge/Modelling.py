import numpy as np
import openseespy.opensees as ops
import sys
from scipy.interpolate import interp1d
import pandas as pd

from .Utility import distance, getRectProp, getCircProp, def_units
from .Utility import get_pyParam_sand, get_pyParam_clay, get_tzParam, get_qzParam

class RC_Circular:

    def __init__(self):
        #  ------------------------------------------------------------------------------------------------------------
        #  DEFINE UNITS
        #  ------------------------------------------------------------------------------------------------------------
        global LunitTXT, FunitTXT, TunitTXT
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, degrees
        global gamma_c, Ubig, Usmall, g

        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = def_units(0)

        # Basic Units
        LunitTXT = 'm'
        FunitTXT = 'kN'
        TunitTXT = 'sec'

        # Angles
        degrees = np.pi / 180.0

        # Concrete unit weight
        gamma_c = 25 * kN / m ** 3

        # Some constants
        g = 9.81 * m / sec ** 2

    @staticmethod
    def _circ_fiber_config(D):
        # Notes
        # The center of the reinforcing bars are placed at the inner radius
        # The core concrete ends at the inner radius (same as reinforcing bars)
        # The reinforcing bars are all the same size
        # The center of the section is at (0,0) in the local axis system
        # Zero degrees is along section y-axis
        yC = 0  # y-axis for center of circular section
        zC = 0  # z-axis for center of circular section
        startAng = 0  # start angle of circular section
        endAng = 360  # end angle of circular section
        ri = 0.0  # diameter of hollow section
        ro = D / 2  # outer diameter of section
        nfCoreR = 10  # number of radial divisions in the core (number of "rings")
        nfCoreT = 10  # number of theta divisions in the core (number of "wedges")
        nfCoverR = 2  # number of radial divisions in the cover
        nfCoverT = 10  # number of theta divisions in the cover

        return yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT

    def _def_BentCirc_Sec(self):
        # Define RC pier sections
        SecTags = []
        SecWs = []
        for i in range(self.num_bents):
            idx = self.model['Bent']['Sections'][i] - 1
            D = self.model['Bent']['D'][idx]  # Section diameter
            Fce = self.model['Bent']['Fce'][idx]  # Concrete compressive strength
            E = 5000 * MPa * (Fce / MPa) ** 0.5  # Concrete Elastic Modulus
            G = E / (2 * (1 + 0.2))  # Concrete Elastic Shear Modulus
            A, Iy, Iz, J = getCircProp(D)  # Mechanical properties
            self.EndSecTag += 1

            # Create Elastic section
            if self.model['Bent']['EleType'] == 0:
                RF = self.model['Bent']['RF']  # Cracked section properties
                ops.section('Elastic', self.EndSecTag, RF * E, A, Iz, Iy, RF * G, J)
                SecTag = self.EndSecTag

            else:
                Fyle = self.model['Bent']['Fyle'][idx]
                Fyhe = self.model['Bent']['Fyhe'][idx]
                cc = self.model['Bent']['cover'][idx]
                numBars = self.model['Bent']['numBars'][idx]
                dl = self.model['Bent']['dl'][idx]
                s = self.model['Bent']['s'][idx]
                dh = self.model['Bent']['dh'][idx]
                TransReinfType = self.model['Bent']['TransReinfType'][idx]
                Hcol = self.model['Bent']['H'][i]
                Lpl1 = (0.08 * Hcol / mm + 0.022 * dl * Fyle) * mm

                if self.model['Bent']['N'] > 1:  # assuming bentcap is rigid enough
                    Lpl2 = (0.08 * Hcol / 2 / mm + 0.044 * dl * Fyle) * mm
                else:
                    Lpl2 = Lpl1
                
                P = min(self.BentAxialForces[i])
                param1 = self._Circ_MPhi(D / mm, cc / mm, numBars, dl / mm, s / mm, dh / mm,
                                         TransReinfType, P, Fce / MPa, Fyle / MPa,
                                         Fyhe / MPa, Lpl1)
                Ke = param1[0]
                Iye = Ke / (E * Iy) * Iy
                Ize = Ke / (E * Iz) * Iz

                if self.model['Bent']['EleType'] == 1:
                    ops.section('Elastic', self.EndSecTag, E, A, Ize, Iye, G, J)
                    SecTag = self.EndSecTag
                    # define elastic shear stiffness
                    self.EndMatTag += 1
                    shearTag = self.EndMatTag
                    ops.uniaxialMaterial('Elastic', shearTag, G * A)
                    # Create the hinge material
                    self.EndMatTag += 1
                    phi1Tag = self.EndMatTag
                    ops.uniaxialMaterial('ModIMKPeakOriented', phi1Tag, *param1)
                    param2 = self._Circ_MPhi(D / mm, cc / mm, numBars, dl / mm, s / mm, dh / mm,
                                             TransReinfType, P, Fce / MPa, Fyle / MPa,
                                             Fyhe / MPa, Lpl2)
                    self.EndMatTag += 1
                    phi2Tag = self.EndMatTag
                    ops.uniaxialMaterial('ModIMKPeakOriented', phi2Tag, *param2)
                    # Create the axial material
                    self.EndMatTag += 1
                    axialTag = self.EndMatTag
                    ops.uniaxialMaterial('Elastic', axialTag, E * A)
                    self.EndSecTag += 1
                    ops.section('Uniaxial', self.EndSecTag, phi2Tag, 'Mz')
                    temp = self.EndSecTag
                    self.EndSecTag += 1
                    ops.section('Aggregator', self.EndSecTag, axialTag, 'P', phi1Tag, 'My', shearTag, 'Vy', shearTag,
                                'Vz', '-section', temp)
                    SecTag = [self.EndSecTag, SecTag]

                elif self.model['Bent']['EleType'] == 2:
                    ops.section('Elastic', self.EndSecTag, E, A, Ize, Iye, G, J)
                    SecTag = self.EndSecTag
                    self._circ_Fiber(Fyle, Fyhe, Fce, D, cc, numBars, dl,
                                     dh, s, TransReinfType)
                    SecTag = [self.EndSecTag, SecTag]

                elif self.model['Bent']['EleType'] in [3, 4]:
                    self._circ_Fiber(Fyle, Fyhe, Fce, D, cc, numBars, dl,
                                     dh, s, TransReinfType)
                    SecTag = self.EndSecTag

            SecTags.append(SecTag)
            SecWs.append(A * gamma_c)

        return SecTags, SecWs

    def _def_PileCirc_Sec(self, case):
        # TODO: Lumped plasticity approach is not implemented
        # Define RC pile sections
        SecTags = []
        SecWs = []

        if case == 'Bent':

            for i in range(self.num_bents):
                idx = self.model['Bent_Foundation']['Sections'][i] - 1
                D = self.model['Bent_Foundation']['D'][idx]  # Section diameter
                Fce = self.model['Bent_Foundation']['Fce'][idx]  # Concrete compressive strength
                E = 5000 * MPa * (Fce / MPa) ** 0.5  # Concrete Elastic Modulus
                G = E / (2 * (1 + 0.2))  # Concrete Elastic Shear Modulus
                A, Iy, Iz, J = getCircProp(D)  # Mechanical properties
                self.EndSecTag += 1

                # Create Elastic section
                if self.model['Bent_Foundation']['EleType'] == 0:
                    RF = self.model['Bent_Foundation']['RF']  # Cracked section properties
                    ops.section('Elastic', self.EndSecTag, RF * E, A, Iz, Iy, RF * G, J)
                    SecTag = self.EndSecTag

                # Create Inelastic fiber section
                else:
                    Fyle = self.model['Bent_Foundation']['Fyle'][idx]
                    Fyhe = self.model['Bent_Foundation']['Fyhe'][idx]
                    cc = self.model['Bent_Foundation']['cover'][idx]
                    numBars = self.model['Bent_Foundation']['numBars'][idx]
                    dl = self.model['Bent_Foundation']['dl'][idx]
                    s = self.model['Bent_Foundation']['s'][idx]
                    dh = self.model['Bent_Foundation']['dh'][idx]
                    TransReinfType = self.model['Bent_Foundation']['TransReinfType'][idx]
                    self._circ_Fiber(Fyle, Fyhe, Fce, D, cc, numBars, dl,
                                     dh, s, TransReinfType)
                    SecTag = self.EndSecTag

                SecTags.append(SecTag)
                SecWs.append(A * gamma_c)

        elif case == 'Abutment':

            for i in range(2):
                idx = self.model['Abutment_Foundation']['Sections'][i] - 1
                D = self.model['Abutment_Foundation']['D'][idx]  # Section diameter
                Fce = self.model['Abutment_Foundation']['Fce'][idx]  # Concrete compressive strength
                E = 5000 * MPa * (Fce / MPa) ** 0.5  # Concrete Elastic Modulus
                G = E / (2 * (1 + 0.2))  # Concrete Elastic Shear Modulus
                A, Iy, Iz, J = getCircProp(D)  # Mechanical properties
                self.EndSecTag += 1

                # Create Elastic section
                if self.model['Abutment_Foundation']['EleType'] == 0:
                    RF = self.model['Abutment_Foundation']['RF']  # Cracked section properties
                    ops.section('Elastic', self.EndSecTag, RF * E, A, Iz, Iy, RF * G, J)
                    SecTag = self.EndSecTag

                # Create Inelastic fiber section
                else:
                    Fyle = self.model['Abutment_Foundation']['Fyle'][idx]
                    Fyhe = self.model['Abutment_Foundation']['Fyhe'][idx]
                    cc = self.model['Abutment_Foundation']['cover'][idx]
                    numBars = self.model['Abutment_Foundation']['numBars'][idx]
                    dl = self.model['Abutment_Foundation']['dl'][idx]
                    s = self.model['Abutment_Foundation']['s'][idx]
                    dh = self.model['Abutment_Foundation']['dh'][idx]
                    TransReinfType = self.model['Abutment_Foundation']['TransReinfType'][idx]
                    self._circ_Fiber(Fyle, Fyhe, Fce, D, cc, numBars, dl,
                                     dh, s, TransReinfType)
                    SecTag = self.EndSecTag

                SecTags.append(SecTag)
                SecWs.append(A * gamma_c)

        return SecTags, SecWs

    def _circ_Fiber(self, Fyle, Fyhe, Fce, D, cc, numBars, dl, dh, s, TransReinfType):
        #  ------------------------------------------------------------------------------------------------------------
        #  CIRCULAR RC FIBER SECTION
        #  ------------------------------------------------------------------------------------------------------------        

        ######### DEFINE MATERIALS #########
        # Based on Mander et al. 1988
        steelID = self.EndMatTag + 1
        self.EndMatTag += 1
        coverID = self.EndMatTag + 1
        self.EndMatTag += 1
        coreID = self.EndMatTag + 1
        self.EndMatTag += 1
        torsionID = self.EndMatTag + 1
        self.EndMatTag += 1
        MinMaxID = self.EndMatTag + 1
        self.EndMatTag += 1
        ShearID = self.EndMatTag + 1
        self.EndMatTag += 1
        numBars = int(numBars)

        # Fyle:              Expected yield strength of longitudinal steel bars
        # Fyhe:              Expected yield strength of transversal steelbars
        # Fce:               Expected nominal compressive strength of the concrete material
        # D:                 Diameter of the circular column section
        # cc:                Clear cover to centroid of stirrups
        # numBars:           Number of longitudinal bars
        # dl:                Nominal diameter of longitudinal rebars
        # dh:                Nominal diameter of transversal rebars
        # s:                 Vertical spacing between the centroid of spirals or hoops
        # TransReinfType:    Type of transversal steel, 'Hoops' or 'Spirals'

        # Use mander model to get confinement factor
        barArea = np.pi * dl ** 2 / 4
        Asl = numBars * barArea
        ds = D - 2 * cc  # Core diameter
        sp = s - dh  # Clear vertical spacing between spiral or hoop bars
        Acc = np.pi * ds ** 2 / 4  # Area of core of section enclosed by the center lines of the perimeter spiral or hoop
        pcc = Asl / Acc  # Ratio of area of longitudinal reinforcement to area of core of section
        # Confinement effectiveness coefficient
        if TransReinfType == 'Hoops':
            ke = (1 - sp / 2 / ds) ** 2 / (1 - pcc)
        elif TransReinfType == 'Spirals':
            ke = (1 - sp / 2 / ds) / (1 - pcc)

        Ash = np.pi * dh ** 2 / 4  # Area of transverse steel
        ps = 4 * Ash / (
                ds * s)  # Ratio of the volume of transverse confining steel to the volume of confined concrete core
        fpl = ke * 0.5 * Fyhe * ps  # Confinement pressure
        Kfc = (-1.254 + 2.254 * (1 + 7.94 * fpl / Fce) ** 0.5 - 2 * fpl / Fce)  # Confinement factor
        # print('Confinement Factor = %.10f' % Kfc)
        # Concrete Properties
        Ec = 5000 * MPa * (Fce / MPa) ** 0.5  # Concrete Elastic Modulus
        Gc = Ec / (2 * (1 + 0.2))  # Concrete Elastic Shear Modulus

        # # recommended parameters by OpenSees webpage, concrete02 and concrete01
        # # unconfined concrete compressive strength Properties
        # fc1U = -Fce                        # UNCONFINED concrete (todeschini parabolic model), maximum stress
        # eps1U = -0.003                    # strain at maximum strength of unconfined concrete
        # fc2U = 0.01*fc1U                   # ultimate stress
        # eps2U = -0.01                     # spalling strain
        # # confined concrete compressive strength Properties
        # fc1C = Kfc*fc1U                   # CONFINED concrete strength
        # eps1C = 2*fc1C/Ec                 # strain at maximum stress 
        # fc2C = 0.2*fc1C                   # ultimate stress
        # eps2C = 5*eps1C                   # strain at ultimate stress
        # # Tensile-Strength Properties
        # # Lambda = 0.1                      # ratio between unloading slope at $eps2 and initial slope $Ec
        # # fctU = -0.14*fc1U      # tensile strength of unconfined concrete
        # # fctC = -0.14*fc1C      # tensile strength of confined concrete
        # # Ets = fctU/0.002                    # tension softening stiffness
        # # Generate Materials, modified kent-park model
        # # ops.uniaxialMaterial('Concrete02', coverID, fc1U, eps1U ,fc2U, eps2U,Lambda,fctU,Ets)    # build cover concrete (unconfined)
        # # ops.uniaxialMaterial('Concrete02', coreID, fc1C, eps1C, fc2C, eps2C,Lambda,fctC,Ets)     # build core concrete (confined)
        # ops.uniaxialMaterial('Concrete01', coverID, fc1U, eps1U ,fc2U, eps2U)    # build cover concrete (unconfined)
        # ops.uniaxialMaterial('Concrete01', coreID, fc1C, eps1C, fc2C, eps2C)     # build core concrete (confined)
        # # print('Concrete01', coreID, fc1C, eps1C, fc2C, eps2C)
        
        # recommended parameters by Mander Model, concrete04
        # unconfined concrete compressive strength Properties
        fc1U = -Fce  # UNCONFINED concrete (todeschini parabolic model), maximum stress
        eps1U = -0.002  # strain at maximum strength of unconfined concrete
        eps2U = -0.005  # spalling strain
        # confined concrete compressive strength Properties
        fc1C = Kfc * fc1U  # CONFINED concrete strength
        eps1C = eps1U * (1 + 5 * (fc1C / fc1U - 1))  # strain at maximum stress
        epssm = 0.10  # max transv. steel strain (usually ~0.10-0.15)*
        eps2C = -(0.004 + 1.4 * ps * Fyhe * epssm / (-fc1C))  # strain at ultimate stress
        # eps2C = -1.5 * (0.004 + 1.4 * ps * Fyhe * epssm / (-fc1C))  # 1.5 is recommended by Kowalsky

        # Tensile-Strength Properties
        fct = 0.56 * MPa * (Fce / MPa) ** 0.5
        # floating point value defining the maximum tensile strength of concrete (optional)
        et = fct / Ec  # floating point value defining ultimate tensile strain of concrete (optional)
        beta = 0.1  # floating point value defining the exponential curve parameter to define the residual stress (as a factor of ft) at etu

        # Generate Materials, uniaxial popovics concrete material
        # If the user defines Ec=5000∗sqrt(|fc|) (in MPa)’ then the envelope curve is identical to proposed by Mander et al. (1988).
        ops.uniaxialMaterial('Concrete04', coverID, fc1U, eps1U, eps2U, Ec, fct, et,
                              beta)  # build cover concrete (unconfined)
        ops.uniaxialMaterial('Concrete04', coreID, fc1C, eps1C, eps2C, Ec, fct, et,
                              beta)  # build core concrete (confined)

        # Torsion Material
        J = np.pi * D ** 4 / 32  # Polar moment of inertia for solid circular section
        ops.uniaxialMaterial('Elastic', torsionID, Gc * J)  # define elastic torsional stiffness
        # Shear Material
        Ag = np.pi * D ** 2 / 4 # Shear Area
        ops.uniaxialMaterial('Elastic', ShearID, Gc * Ag)  # define elastic shear stiffness

        # Reinforcing steel properties
        Es = 200.0 * GPa  # modulus of steel
        Bs = 0.005  # strain-hardening ratio
        R0 = 18  # control the transition from elastic to plastic branches
        cR1 = 0.925  # control the transition from elastic to plastic branches
        cR2 = 0.15  # control the transition from elastic to plastic branches
        minStrain = -0.1  # minimum steel strain in the fibers (steel buckling)
        maxStrain = 0.1  # maximum steel strain in the fibers (steel rupture)
        # Generate Materials
        ops.uniaxialMaterial('Steel02', steelID, Fyle, Es, Bs, R0, cR1, cR2)
        ops.uniaxialMaterial('MinMax', MinMaxID, steelID, '-min', minStrain, '-max', maxStrain)

        ######### DEFINE SECTIONS #########
        yC, zC, startAng, endAng, ri, ro, nfCoreR, nfCoreT, nfCoverR, nfCoverT = self._circ_fiber_config(D)
        # Define the fiber section, neglect shear deformations
        SecTag = self.EndSecTag + 1
        self.EndSecTag += 1
        ops.section('Fiber', SecTag, '-torsion', torsionID)
        rc = ro - cc  # Core radius
        ops.patch('circ', coreID, nfCoreT, nfCoreR, yC, zC, ri, rc, startAng, endAng)  # Define the core patch
        ops.patch('circ', coverID, nfCoverT, nfCoverR, yC, zC, rc, ro, startAng, endAng)  # Define the cover patch
        theta = endAng / numBars  # Determine angle increment between bars
        ops.layer('circ', MinMaxID, numBars, barArea, yC, zC, rc, theta, endAng)  # Define the reinforcing layer
        SecTag = self.EndSecTag + 1
        self.EndSecTag += 1
        ops.section('Aggregator', self.EndSecTag, ShearID, 'Vy', ShearID, 'Vz', '-section', self.EndSecTag - 1)

    @staticmethod
    def _Circ_MPhi(D, cc, nbl, Dbl, s, Dh, stype, P, fpc, fsy, fyhe, Lpl):
        """
        =============================================================================
        Start of User Input Data
        =============================================================================
        # section properties:
        D = 2000  # section diameter (mm)
        cc = 50   # cover center of stirrups (mm)
        
        # reinforcement details:
        nbl = 30  # number of longitudinal bars
        Dbl = 32  # long. bar diameter (mm)
        Dh = 20  # diameter of transverse reinf. (mm)
        stype = 'Spirals'  # 'spirals' or 'hoops'*
        s = 100  # spacing of transverse steel (mm)*
        
        # aplied loads:
        P = 10000  # axial load kN (-) tension (+)compression
        
        # materials
        fpc = 1.3 * 30  concrete compressive strength (MPa)
        fsy = 420*1.2 steel yeld strength (MPa)
        Lpl = 0.4m plastic hinge length
        """

        def raynor(fsy, dels):
            """
        
            Parameters
            ----------
            Steel : str
                Reinforcing Steel Type.
            dels : float
                strain increment.
        
            Returns
            -------
            es : numpy.ndarray
                strains.
            fs : numpy.ndarray
                stresses.
        
            """
            Es = 200000
            esy = fsy / Es
            fsu = 1.3 * fsy
            esh = 0.008
            esu = 0.1
            es = np.linspace(0, esu, int(esu / dels + 1))
            fs = es * 0
            Ey = 0  # slope of pre-hardening part
            C1 = 2  # shape coeff.

            for i in range(len(es)):
                if es[i] < esy:
                    fs[i] = Es * es[i]

                if esy <= es[i] <= esh:
                    fs[i] = fsy + (es[i] - esy) * Ey

                if es[i] > esh:
                    fs[i] = fsu - (fsu - fsy) * (((esu - es[i]) / (esu - esh)) ** C1)

            return es, fs

        def circ_un(fce, dels):
            """
            Details
            -------
            Mander model for unconfined normal weight concrete
            
            Parameters
            ----------
            fce : float
                Concrete compressive strength (MPa).
            dels : float
                strain increment.
        
            Returns
            -------
            ec : numpy.darray
                concrete strain vector.
            fc : numpy.darray
                concrete strength vector (MPa).
        
            """
            Ec = 5000 * (fce ** 0.5)  # Young's modulus of concrete. (MPa)
            espall = 0.0064  # spalling strain
            eco = 0.002  # concrete strain at peak concrete strength
            ec = np.arange(0, espall, dels)
            fc = ec * 0
            Esecu = fce / eco
            ru = Ec / (Ec - Esecu)
            xu = ec / eco

            for i in range(len(ec)):
                if ec[i] < 2 * eco:
                    fc[i] = fce * xu[i] * ru / (ru - 1 + xu[i] ** ru)

                elif 2 * eco <= ec[i] <= espall:
                    fc[i] = fce * (2 * ru / (ru - 1 + 2 ** ru)) * (1 - (ec[i] - 2 * eco) / (espall - 2 * eco))

                elif ec[i] >= espall:
                    break

            return ec, fc

        def circ_conf(Ast, Dh, cc, s, fce, fyhe, D, dels, stype):
            """
            Details
            -------
            Mander model for confined normal weight concrete, 
            solid circular sections
            
            Parameters
            ----------
            Ast : float
                total long. steel area (mm**2).
            Dh : float
                diameter of transverse reinf. (mm).
            cc : float
                cover to longitudinal bars (mm).
            s : float
                spacing of transverse steel (mm).
            fce : float
                Concrete compressive strength (MPa).
            fye : float
                long steel yielding stress (MPa).
            fyhe : float
                transv steel yielding stress (MPa).
            D : float
                section diameter (mm).
            dels : float
                strain increment.
            stype : str
                transv. reinf. type: 'spirals' or 'hoops'.
        
            Returns
            -------
            ec : numpy.darray
                concrete strain vector.
            fc : numpy.darray
                concrete strength vector (MPa).
            """
            esm = 0.1  # max transv. steel strain (usually ~0.10-0.15).
            Ec = 5000 * (fce ** 0.5)  # Young's modulus of concrete. (MPa)
            eco = 0.002  # concrete strain at peak concrete strength
            sp = s - Dh
            Ash = 0.25 * np.pi * (Dh ** 2)
            ds = D - 2 * cc  # core diameter
            ros = 4 * Ash / (ds * s)  # transv. steel area ratio
            Ac = 0.25 * np.pi * (ds ** 2)  # core area
            rocc = Ast / Ac  # long. steel area ratio
            if stype == 'Spirals':
                n = 1
            elif stype == 'Hoops':
                n = 2
            ke = ((1 - sp / (2 * ds)) / (1 - rocc)) ** n
            fpl = 0.5 * ke * ros * fyhe
            lambdac = (-1.254 + 2.254 * np.sqrt(1 + 7.94 * fpl / fce) - 2 * fpl / fce)
            fcc = lambdac * fce
            ecc = eco * (1 + 5 * (lambdac - 1))
            Esec = fcc / ecc
            r = Ec / (Ec - Esec)
            # ecu = 1.5 * (0.004 + 1.4 * ros * fyhe * esm / fcc)  # 1.5 is recommended by Kowalsky
            ecu = 0.004 + 1.4 * ros * fyhe * esm / fcc  # from Priestley
            ec = np.arange(0, ecu, dels)
            x = (ec / ecc)
            fc = fcc * x * r / (r - 1 + x ** r)

            return ec, fc, ros

        # analysis control parameters:
        itermax = 1000  # max number of iterations
        ncl = 50  # # of concrete layers
        tolerance = 0.001  # x fpc x Ag
        dels = 0.0001  # delta strain for default material models

        # =============================================================================
        # End of User Input Data
        # =============================================================================
        # yield strain
        esy = fsy / 200000

        Dsp = D - 2 * cc  # core diameter
        dcore = cc  # distance to the core
        P = P * 1000  # axial load in Newtons
        Ast = nbl * 0.25 * np.pi * (Dbl ** 2)  # total long. steel area mm2

        tcl = D / ncl  # thickness of concrete layers
        yl = tcl * np.arange(1, ncl + 1)  # border distance conc. layer

        es, fs = raynor(fsy, dels)
        ecun, fcun = circ_un(fpc, dels)
        ec, fc, rho_shr = circ_conf(Ast, Dh, cc, s, fpc, fyhe, D, dels, stype)

        ecu = ec[-1]  # maximum strain confined concrete
        esu = es[-1]  # maximum strain steel

        # vector with strains of confined concrete
        ec = np.append(-1e10, ec)
        ec = np.append(ec, ec[-1] + dels)
        ec = np.append(ec, 1e10)
        # vector with stresses of confined concrete  
        fc = np.append(0, fc)
        fc = np.append(fc, 0)
        fc = np.append(fc, 0)

        # vector with strains of unconfined concrete
        ecun = np.append(-1e10, ecun)
        ecun = np.append(ecun, ecun[-1] + dels)
        ecun = np.append(ecun, 1e10)
        # vector with stresses of unconfined concrete
        fcun = np.append(0, fcun)
        fcun = np.append(fcun, 0)
        fcun = np.append(fcun, 0)

        # vector with strains of the steel
        es = np.append(es, es[-1] + dels)
        es = np.append(es, 1e10)
        # vector with stresses of the steel
        fs = np.append(fs, 0)
        fs = np.append(fs, 0)

        esaux = np.zeros(len(es))
        fsaux = 0 * esaux

        for i in range(len(es)):
            esaux[i] = es[len(es) - i - 1]
            fsaux[i] = fs[len(fs) - i - 1]

        # vector with strains of the steel
        es = np.append(-esaux, es[3:])
        # vector with stresses of the steel
        fs = np.append(-fsaux, fs[3:])

        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.fill_between(ec, 0, fc, edgecolor='black', facecolor='green', interpolate=True, alpha=0.5, label='Confined Concrete')
        # ax.fill_between(ecun, 0, fcun, edgecolor='black', facecolor='blue', interpolate=True, alpha=0.5,
        #                 label='Unconfined Concrete')
        # ax.set_ylabel('Stress [MPa]')
        # ax.set_xlabel('Strain')
        # ax.set_xlim([0, 1.05 * ec[-3]])
        # ax.set_ylim([0, 1.05 * np.max(fc)])
        # ax.legend(loc='upper right')
        # ax.grid(True)
        # ax.set_title('Stress-Strain Relation for Concrete')

        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.fill_between(es, 0, fs, edgecolor='black', facecolor='Red', interpolate=True, alpha=0.5, label='Reinforcing Steel')
        # ax.set_ylabel('Stress [MPa]')
        # ax.set_xlabel('Strain')
        # ax.set_xlim([1.05 * es[3], 1.05 * es[-3]])
        # ax.set_ylim([1.05 * fs[3], 1.05 * np.max(fs)])
        # ax.legend(loc='upper left')
        # ax.grid(True)
        # ax.set_title('Stress-Strain Relation for Reinforcing Steel')

        # ============================== CONCRETE LAYERS ============================

        # add layers to consider unconfined concrete
        yl = np.append(np.append(yl, dcore), D - dcore)
        yl.sort()
        # confined concrete layers
        yc = yl - dcore
        yc = yc[np.where((0 < yc) * (yc <= Dsp))[0]]

        # total area of each layer
        Atemp = ((D / 2) ** 2) * np.arccos(1 - 2 * yl / D) - (D / 2 - yl) * ((D * yl - yl ** 2) ** 0.5)
        Atc = Atemp - np.append(0, Atemp[:-1])

        # total area of each conf. layer
        Atemp = ((Dsp / 2) ** 2) * np.arccos(1 - 2 * yc / Dsp) - (Dsp / 2 - yc) * ((Dsp * yc - yc ** 2) ** 0.5)
        Atcc = Atemp - np.append(0, Atemp[:-1])

        conclay = []
        k = 0
        for i in range(len(yl)):
            if yl[i] <= dcore or yl[i] > D - dcore:
                conclay.append([Atc[i], 0])

            if dcore < yl[i] <= D - dcore:
                conclay.append([Atc[i] - Atcc[k], Atcc[k]])
                k += 1

        conclay = np.asarray(conclay)
        yl.shape = (len(yl), 1)
        ycenter = np.append(yl[0] / 2, 0.5 * (yl[:-1] + yl[1:]))
        ycenter.shape = (len(ycenter), 1)

        # [center_layer|A_uncon|A_conf|d_top_layer]
        conclay = np.concatenate((ycenter, conclay, yl), axis=1)

        # ================================    REBARS     =====================================

        Asb = 0.25 * np.pi * (Dbl ** 2)
        r = 0.5 * (D - 2 * cc - Dh - Dbl)
        theta = (2 * np.pi / nbl) * np.arange(0, nbl)
        distld = (0.5 * (D - 2 * r) + r * np.sin(theta) * np.tan(0.5 * theta))
        distld.sort()  # y coordinate of each bar

        # =============================== CORRECTED AREAS ======================================

        # Substract the steel area
        for i in range(nbl):
            aux = np.where(yl > distld[i])[0][0]
            conclay[aux, 2] = conclay[aux, 2] - Asb
            if conclay[aux, 2] < 0:
                print('decrease # of layers')
                sys.exit()

        # ============  Define vector (def) with the deformations in the top concrete ==================

        df = np.arange(1e-4, 20 * ecu, 1e-4)

        if ecu > 0.0018:
            df = np.append(df[df <= 16e-4], np.arange(18e-4, 20 * ecu, 2e-4))

        if ecu > 0.0025:
            df = np.append(df[df <= 20e-4], np.arange(25e-4, 20 * ecu, 5e-4))

        if ecu > 0.006:
            df = np.append(df[df <= 50e-4], np.arange(60e-4, 20 * ecu, 10e-4))

        if ecu > 0.012:
            df = np.append(df[df <= 100e-4], np.arange(120e-4, 20 * ecu, 20e-4))

        npts = len(df)

        if P > 0:
            for k in range(npts):
                f1 = interp1d(ecun, fcun)
                temp1 = np.sum(f1(df[0] * np.ones(len(yl))) * conclay[:, 1])
                f2 = interp1d(ec, fc)
                temp2 = np.sum(f2(df[0] * np.ones(len(yl))) * conclay[:, 2])
                f3 = interp1d(es, fs)
                temp3 = np.sum(Asb * f3(df[0] * np.ones(len(distld))))
                compch = temp1 + temp2 + temp3
                if compch < P:
                    df = df[1:]

        npts = len(df)

        # ===============ITERATIVE PROCESS TO FIND THE MOMENT - CURVATURE RELATION: ==============================

        msg = 0  # stop conditions
        curv = [0]  # curvatures
        mom = [0]  # moments
        ejen = [0]  # neutral axis
        DF = [0]  # force eqilibrium
        vniter = [0]  # iterations
        coverstrain = [0]
        corestrain = [0]
        steelstrain = [0]

        tol = tolerance * 0.25 * np.pi * (D ** 2) * fpc  # tolerance allowed
        x = [D / 2]  # location of N.A.
        for k in range(npts):
            lostmomcontrol = max(mom)
            if mom[k] < (0.8 * lostmomcontrol):
                msg = 4
                break

            F = 10 * tol
            niter = -1
            while abs(F) > tol:
                niter = niter + 1
                eec = (df[k] / x[niter]) * (conclay[:, 0] - (D - x[niter]))  # vector with the strains in the concrete
                ees = (df[k] / x[niter]) * (distld - (D - x[niter]))  # vector with the strains in the steel

                fcunconf = interp1d(ecun, fcun)(eec)  # vector with stresses in the unconfined concr.
                fcconf = interp1d(ec, fc)(eec)  # vector with stresses in the confinded concr.
                fsteel = interp1d(es, fs)(ees)  # vector with stresses in the steel
                FUNCON = fcunconf * conclay[:, 1]
                FCONF = fcconf * conclay[:, 2]
                FST = Asb * fsteel;
                F = np.sum(FUNCON) + np.sum(FCONF) + np.sum(FST) - P
                if F > 0:
                    x.append(x[niter] - 0.05 * x[niter])

                elif F < 0:
                    x.append(x[niter] + 0.05 * x[niter])

                if niter > itermax:
                    msg = 3
                    break

            cores = (df[k] / x[niter]) * abs(x[niter] - dcore)
            if cores >= ecu:
                msg = 1
                break

            if abs(ees[0]) > esu:
                msg = 2
                break

            ejen.append(x[niter])
            DF.append(x[niter])
            vniter.append(niter)
            temp = (sum(FUNCON * conclay[:, 0]) + sum(FCONF * conclay[:, 0]) + sum(FST * distld) - P * (D / 2)) / (
                    10 ** 6)
            mom.append(temp)

            if mom[k + 1] < 0:
                mom[k + 1] = -0.01 * mom[k + 1]

            curv.append(1000 * df[k] / x[niter])
            coverstrain.append(df[k])
            corestrain.append(cores)
            steelstrain.append(ees[0])
            x[0] = x[niter]
            del x[1:]
            if msg != 0:
                break

        # Do bilinearization based on equivalent energy
        Enrgy = np.trapz(curv, mom)
        fycurv = interp1d(steelstrain, curv)(-esy)  # curvature for first yield
        fyM = interp1d(curv, mom)(fycurv)  # moment for first yield
        Enrgy_diff = []
        Mns = []
        for xx in np.arange(1, 2, 0.001):
            Mn = xx * fyM
            Mns.append(Mn)
            eqcurv = (Mn / fyM) * fycurv
            curv_trial = np.append(np.append(0, eqcurv), curv[-1])
            mom_trial = np.append(np.append(0, Mn), mom[-1])
            Enrgy_trial = np.trapz(curv_trial, mom_trial)
            Enrgy_diff.append(abs(Enrgy - Enrgy_trial))

        Mn = Mns[Enrgy_diff.index(min(Enrgy_diff))]
        eqcurv = max((Mn / fyM) * fycurv, fycurv)
        curvbilin = np.append(np.append(0, eqcurv), curv[-1])
        mombilin = np.append(np.append(0, Mn), mom[-1])

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(curvbilin,mombilin,c='red')
        # plt.plot(curv,mom,c='blue',linestyle='--')
        # plt.grid(True)
        # plt.xlabel('\u03A6 [m$^{-1}$]',fontsize=16, fontname='Times New Roman', fontstyle='italic')
        # plt.ylabel('M [kN.m]',fontsize=16, fontname='Times New Roman', fontstyle='italic')
        # plt.title('Moment - Curvature Relation',fontsize=16, fontname='Times New Roman', fontstyle='italic')
        # plt.xlim(ax.get_xlim()[0]*1e-2, ax.get_xlim()[1])
        # plt.ylim(0, ax.get_ylim()[1])
        # plt.xticks([],[])
        # plt.yticks([],[])
        # plt.savefig('MPhi.pdf', format = 'pdf', bbox_inches = 'tight', facecolor = None)

        # --------------------------------------
        # Compute Stiffness Deterioration
        # --------------------------------------
        # Haselton CB, Deierlein G.G. Assessing seismic collapse safety of modern 
        # reinforced concrete moment frame buildings. Stanford University, 2008.
        # Post-capping rotation capacity, eqn. 4.20, p. 103
        Agross = 0.25 * np.pi * (D ** 2)
        nu = P / (fpc * Agross)
        dist = D - cc - Dbl / 2  # Depth to bottom bar
        Lambda = 170.7 * 0.27 ** nu * 0.1 ** (s / dist)

        # --------------------------------------
        # Compute the Capping Curvature
        # --------------------------------------
        # Haselton CB, Deierlein G.G. Assessing seismic collapse safety of modern 
        # reinforced concrete moment frame buildings. Stanford University, 2008.
        # Post-capping rotation capacity, eqn. 4.17, p. 103
        Ash = np.pi * Dh ** 2 / 4
        rho_shr = 4 * Ash / (s * Dsp)
        thetaPc = min(0.76 * 0.031 ** nu * (0.02 + 40 * rho_shr) ** 1.02, 0.1)
        phiU = curvbilin[2] + thetaPc / Lpl

        K0 = mombilin[1] / curvbilin[1]  # Elastic stiffness
        K1 = (mombilin[2] - mombilin[1]) / (curvbilin[2] - curvbilin[1])
        as_Plus = K1 / K0  # Strain hardening ratio for positive loading direction
        as_Neg = K1 / K0  # Strain hardening ratio for negative loading direction
        My_Plus = mombilin[1]  # Effective yield strength for positive loading direction
        My_Neg = -mombilin[1]  # Effective yield strength for negative loading direction
        Lamda_S = Lambda  # Cyclic deterioration parameter for strength deterioration [see definitions in Lignos and Krawinkler (2011)]
        Lamda_C = Lambda  # Cyclic deterioration parameter for post-capping strength deterioration [see definitions in Lignos and Krawinkler (2011)]
        Lamda_A = 0.0  # Cyclic deterioration parameter for acceleration reloading stiffness deterioration [see definitions in Lignos and Krawinkler (2011)]
        Lamda_K = 0.0  # Cyclic deterioration parameter for unloading stiffness deterioration [see definitions in Lignos and Krawinkler (2011)]
        c_S = 1.0  # rate of strength deterioration. The default value is 1.0.
        c_C = 1.0  # rate of post-capping strength deterioration. The default value is 1.0.
        c_A = 0.0  # rate of accelerated reloading deterioration. The default value is 1.0.
        c_K = 0.0  # rate of unloading stiffness deterioration. The default value is 1.0.
        phi_p_Plus = curvbilin[2] - curvbilin[
            1]  # pre-capping rotation for positive loading direction (often noted as plastic rotation capacity)
        phi_p_Neg = curvbilin[2] - curvbilin[
            1]  # pre-capping rotation for negative loading direction (often noted as plastic rotation capacity) (must be defined as a positive value)
        phi_pc_Plus = thetaPc / Lpl  # post-capping rotation for positive loading direction
        phi_pc_Neg = thetaPc / Lpl  # post-capping rotation for negative loading direction (must be defined as a positive value)
        Res_Pos = 0.1  # residual strength ratio for positive loading direction
        Res_Neg = 0.1  # residual strength ratio for negative loading direction (must be defined as a positive value)
        phi_u_Plus = 2 * phiU  # ultimate rotation capacity for positive loading direction
        phi_u_Neg = 2 * phiU  # ultimate rotation capacity for negative loading direction (must be defined as a positive value)
        D_Plus = 1.0  # rate of cyclic deterioration in the positive loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.
        D_Neg = 1.0  # rate of cyclic deterioration in the negative loading direction (this parameter is used to create assymetric hystereti

        param = [K0, as_Plus, as_Neg, My_Plus, My_Neg, Lamda_S, Lamda_C, Lamda_A, Lamda_K, c_S, c_C, c_A, c_K,
                 phi_p_Plus, phi_p_Neg, phi_pc_Plus, phi_pc_Neg, Res_Pos, Res_Neg, phi_u_Plus, phi_u_Neg,
                 D_Plus, D_Neg]

        return param

class Builder(RC_Circular):

    def __init__(self):
        """
        --------------------------
        OBJECT INITIALIZATION
        --------------------------
        """
        # INITIALIZE THE RC_Circular CLASS OBJECT USING INHERITANCE
        RC_Circular.__init__(self)  # This object defines the circular pier sections

        #  ---------------------------
        #  DEFINE UNITS
        #  ---------------------------
        global LunitTXT, FunitTXT, TunitTXT
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, degrees
        global gamma_c, Ubig, Usmall, g

        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = def_units(0)

        # Basic Units
        LunitTXT = 'm'
        FunitTXT = 'kN'
        TunitTXT = 'sec'

        # Angles
        degrees = np.pi / 180.0

        # Concrete unit weight
        gamma_c = 25 * kN / m ** 3

        # Some constants
        g = 9.81 * m / sec ** 2
        
        # Node Tags
        self.D1Tag = '001'  # Identifier for Deck nodes
        self.D2Tag = '002'  # Identifier for Deck nodes
        self.D3Tag = '003'  # Identifier for internal Deck Nodes
        self.PTag = '004'  # Identifier for pier nodes
        self.LTag = '005'  # Identifier for linkslab nodes
        self.BTag = '006'  # Identifier for bearing node
        self.BC1Tag = '007'  # Identifier for bentcap pier nodes
        self.BC2Tag = '008'  # Identifier for bentcap bearing nodes
        self.BC3Tag = '009'  # Identifier for bentcap start and end nodes
        self.Found1Tag = '010'  # Identifier for the fixed nodes of foundation soil springs
        self.Found2Tag = '011'  # Identifier for the embedded nodes of foundation soil springs
        self.Found3Tag = '012'  # Identifier for the pile nodes of foundation soil springs
        self.Found4Tag = '013'  # Identifier for the pile nodes of foundation cap nodes
        self.ATag = '014'  # Identified for abutment nodes

        # Element Tags
        self.DeckTag = '001'  # Identifier for deck elements
        self.LinkTag = '002'  # Identifier for lnikslab elements
        self.BearingTag = '003'  # Identifier for bearing elements
        self.BentCapTag = '004'  # Identifier for bentcap elements
        self.PierTag = '005'  # Identifier for pier elements
        self.SpringEleTag = '006'  # Identifier for zerolength soil springs
        self.PileEleTag = '007'  # Identifier for pile elements (single pile foundation)
        self.GapTag = '008'  # Identifier for gap elements
        self.AbutTag = '009'  # Identified for abutment springs
        self.RigidTag = '099'  # Rigid like elements end with this tag

    def _build(self):
        """
        --------------------------
        MODEL BUILDER
        --------------------------
        """
        
        ops.wipe()  # Remove any existing model
        ops.wipeAnalysis()
        ops.model('basic', '-ndm', 3, '-ndf', 6)  # Define the model builder, ndm=#dimension, ndf=#dofs

        # Define dummies and counters
        self.BigMat = 1
        self.BigSec = 1
        self.BigInt = 1
        self.SmallSec = 2
        self.SmallInt = 2
        self.SmallMat = 2
        self.ZeroMat = 3
        self.bigMat = 4
        self.EndMatTag = 4
        self.EndSecTag = 2
        self.EndIntTag = 2
        self.EndTransfTag = 0
        ops.uniaxialMaterial('Elastic', self.BigMat, 1e14)
        ops.section('Elastic', self.BigSec, 1e14, 1, 1, 1, 1e14, 1)
        ops.beamIntegration('Legendre', self.BigInt, self.BigSec, 2)
        ops.section('Elastic', self.SmallSec, 1e-5, 1, 1, 1, 1e-5, 1)
        ops.beamIntegration('Legendre', self.SmallInt, self.SmallSec, 2)
        ops.uniaxialMaterial('Elastic', self.SmallMat, 10)
        ops.uniaxialMaterial('Elastic', self.ZeroMat, 0)
        ops.uniaxialMaterial('Elastic', self.bigMat, 1e8)

        # Mass type to use in elements
        if self.model['Mass'] == 'Consistent':
            self.mass_type = '-cMass'
        else:
            self.mass_type = '-lMass'

        # Store nodes which are rigidly connected
        self.RigidLinkNodes = []

        # Definition of constraints 
        self.const_opt = 0  # (0: via rigid links, 1: via rigid like elements)

        # Start building the model
        self._deck()  # Define deck elements
        self._linkslab()  # Define linkSlabs or Expansion joints
        self._bent()  # Define bent elements
        self._bearing()  # Define bearing elements
        self._bentcap()  # Define bentcap elements
        self._abutment()  # Define abutment elements
        self._foundation()  # Define foundation elements
        self._constraints()  # Define constraints e.g. Rigid Links
    
    def _deck(self):
        """
        --------------------------
        DECK MODELLING
        --------------------------
        """ 
        # INPUTS
        DXs = self.model['Deck']['Xs']
        DYs = self.model['Deck']['Ys']
        DZs = self.model['Deck']['Zs']
        A = self.model['Deck']['A']
        E = self.model['Deck']['E']
        G = self.model['Deck']['G']
        J = self.model['Deck']['J']
        Iz = self.model['Deck']['Iz']
        Iy = self.model['Deck']['Iy']
        wSDL = self.model['Deck']['wSDL']
        numDeckEle = self.model['Deck']['numEle']
        if self.model['Deck']['Type'] == 'Continuous':
            self.model['LinkSlab']['L'] = 0

        L_link = self.model['LinkSlab']['L']

        # Necessary info to save
        self.num_spans = len(DXs) - 1
        self.D1Nodes = []
        self.D2Nodes = []
        self.skew = []
        self.EleLoadsDeck = []  # Element Loads
        self.EleIDsDeck = []  # Element Tags
        self.DeckIntNodes = []  # Internal Deck Nodes
        self.Vx_deck = []
        self.Vy_deck = []
        self.Vz_deck = []  # Vectors defining the element orientation

        # NODES, node name: idx_tag 
        for i in range(self.num_spans):
            self.skew.append(
                np.arctan((DYs[i + 1] - DYs[i]) / (DXs[i + 1] - DXs[i])))
            skew = self.skew[i]

            if i == 0:
                Coord1 = [DXs[i], DYs[i], DZs[i]]
                Coord2 = [DXs[i + 1] - np.cos(skew) * L_link, DYs[i + 1] - np.sin(skew) * L_link, DZs[i + 1]]
            elif i == self.num_spans - 1:
                Coord1 = [DXs[i] + np.cos(skew) * L_link, DYs[i] + np.sin(skew) * L_link, DZs[i]]
                Coord2 = [DXs[i + 1], DYs[i + 1], DZs[i + 1]]
            else:
                Coord1 = [DXs[i] + np.cos(skew) * L_link, DYs[i] + np.sin(skew) * L_link, DZs[i]]
                Coord2 = [DXs[i + 1] - np.cos(skew) * L_link, DYs[i + 1] - np.sin(skew) * L_link, DZs[i + 1]]

            if self.model['Deck']['Type'] == 'Discontinuous':  # Discontinuous Deck Case
                nodeI = int(str(i + 1) + self.D1Tag)
                nodeJ = int(str(i + 1) + self.D2Tag)
                ops.node(nodeI, *Coord1)
                ops.node(nodeJ, *Coord2)
                self.D1Nodes.append(nodeI)
                self.D2Nodes.append(nodeJ)

            elif self.model['Deck']['Type'] == 'Continuous':  # Continuous Deck Case
                if i == 0:
                    nodeI = int(str(i + 1) + self.D1Tag)
                    nodeJ = int(str(i + 2) + self.D1Tag)
                    ops.node(nodeI, *Coord1)
                    ops.node(nodeJ, *Coord2)
                    self.D1Nodes.append(nodeI)
                    self.D2Nodes.append(nodeJ)
                else:
                    nodeI = self.D2Nodes[-1]
                    nodeJ = int(str(i + 2) + self.D1Tag)
                    ops.node(nodeJ, *Coord2)
                    self.D1Nodes.append(nodeI)
                    self.D2Nodes.append(nodeJ)

        # create the sections
        self.EndSecTag += 1
        self.EndIntTag += 1
        ops.section('Elastic', self.EndSecTag, E, A, Iz, Iy, G, J)
        ops.beamIntegration('Legendre', self.EndIntTag, self.EndSecTag, 2)

        # ELEMENTS, element name: span_idx(100,200, etc.)_tag
        for i in range(self.num_spans):
            wTOT = (wSDL + gamma_c * A)
            idx = str(i + 1)
            eleTag = int(idx + self.DeckTag)
            nodeTag = int(idx + self.D3Tag)
            skew = self.skew[i]

            # Store these vectors, bridge could have in plane curvature as well.
            Coord1 = ops.nodeCoord(self.D1Nodes[i])
            Coord2 = ops.nodeCoord(self.D2Nodes[i])
            Vx = np.asarray([Coord2[0] - Coord1[0], Coord2[1] - Coord1[1], Coord2[2] - Coord1[2]])
            Vx = Vx / np.sqrt(Vx.dot(Vx))
            Vy = np.asarray([np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0])
            Vy = Vy / np.sqrt(Vy.dot(Vy))  # I am neglecting superelevation here! But this can be added as well!
            Vz = np.cross(Vx, Vy)
            Vz = Vz / np.sqrt(Vz.dot(Vz))
            Vz = list(Vz)
            self.EndTransfTag += 1
            ops.geomTransf('Linear', self.EndTransfTag, *Vz)
            eleList, nodeList = DiscretizeMember(self.D1Nodes[i], self.D2Nodes[i], numDeckEle, 'forceBeamColumn',
                                                 self.EndIntTag, self.EndTransfTag, nodeTag, eleTag, wTOT / g,
                                                 self.mass_type)

            for ele in eleList:
                self.EleIDsDeck.append(ele)
                self.EleLoadsDeck.append(wTOT)
                self.Vx_deck.append(Vx)
                self.Vy_deck.append(Vy)
                self.Vz_deck.append(np.asarray(Vz))
            self.DeckIntNodes.append(nodeList)

    def _linkslab(self):
        """
        --------------------------
        LINK SLAB MODELLING
        --------------------------
        TODO: Need to Add Deck Hinge Option (or expansion joint) for Discontinuous Case
        ---> if not self.model['LinkSlab']['cond'][i]
        """
        # INPUTS
        d = self.model['LinkSlab']['d']
        t = self.model['LinkSlab']['t']
        w = self.model['LinkSlab']['w']
        E = self.model['LinkSlab']['E']
        G = self.model['LinkSlab']['G']
        wSDL = self.model['Deck']['wSDL']

        # Necessary info to save
        self.EleLoadsLink = []  # Element Loads
        self.EleIDsLink = []  # Element Tags

        if self.model['Deck']['Type'] == 'Discontinuous':
            # create the sections
            A, Iz, Iy, J = getRectProp(t, w)
            Iy *= 0.2
            Iz *= 0.5
            J *= 0.5
            self.EndSecTag += 1
            self.EndIntTag += 1
            ops.section('Elastic', self.EndSecTag, E, A, Iz, Iy, G, J)
            ops.beamIntegration('Legendre', self.EndIntTag, self.EndSecTag, 2)
            self.EndTransfTag += 1
            ops.geomTransf('Linear', self.EndTransfTag, 0, 0, 1)  # assuming that z axis is always the same with global
            wTOT = (wSDL + gamma_c * A)
            for i in range(self.num_spans - 1):
                if self.model['LinkSlab']['cond'][i]:
                    # NODES, node name: idx_tag
                    Coord1 = ops.nodeCoord(self.D2Nodes[i])
                    Coord1[2] += d
                    Coord2 = ops.nodeCoord(self.D1Nodes[i + 1])
                    Coord2[2] += d
                    nodeI = int(str(2 * i + 1) + self.LTag)
                    nodeJ = int(str(2 * i + 2) + self.LTag)
                    ops.node(nodeI, *Coord1)
                    ops.node(nodeJ, *Coord2)

                    # RIGID LINKS
                    self.RigidLinkNodes.append([self.D2Nodes[i], nodeI])
                    self.RigidLinkNodes.append([self.D1Nodes[i + 1], nodeJ])

                    # ELEMENTS, element name: bent_tag
                    eleTag = int(str(i + 1) + self.LinkTag)
                    ops.element('dispBeamColumn', eleTag, nodeI, nodeJ, self.EndTransfTag, self.EndIntTag, '-mass',
                                wTOT / g)
                    self.EleIDsLink.append(eleTag)
                    self.EleLoadsLink.append(wTOT)

        elif self.model['Deck']['Type'] == 'Continuous':
            # Do not create any linkslab if deck is continuous of course
            pass

    def _bent(self):
        """
        --------------------------
        BENT MODELLING
        --------------------------
        TODO: Discretization option for pier elements (can be imported in case of long piers)
        """

        # INPUTS
        self.num_piers = self.model['Bent']['N']
        H_piers = self.model['Bent']['H']
        dist = self.model['Bent']['Dist']
        DXs = self.model['Deck']['Xs']
        DYs = self.model['Deck']['Ys']
        DZs = self.model['Deck']['Zs']

        # Initialize some parameters
        self.num_bents = self.num_spans - 1
        self.EleLoadsBent = []
        self.EleIDsBent = []
        self.BentStartNodes = []
        self.BentEndNodes = []

        # Element type
        if self.model['Bent']['EleType'] == 0:
            eleType = 'dispBeamColumn'
        else:
            eleType = 'forceBeamColumn'

        # Define sections
        IntTags, BentWs = self._def_bent_Int()

        for i in range(self.num_bents):
            # Define Integration Tag of the bent
            IntTag = IntTags[i]
            # Define Distributed Load of the bent
            wTOT = BentWs[i]
            # Define skew angle of the bent
            skew = (self.skew[i] + self.skew[i + 1]) / 2
            # Define geometric transformation of the bent
            self.EndTransfTag += 1  # local z is in global x
            ops.geomTransf('PDelta', self.EndTransfTag, np.cos(skew), np.sin(skew), 0)

            # Define Nodal Coordinates of the bent
            dist_mid = dist * (self.num_piers - 1) / 2
            zTop = DZs[i + 1] - self.model['Bearing']['dv'][i + 1] - self.model['Bearing']['h'][i + 1] - \
                    self.model['BentCap']['h']
            zBot = zTop - H_piers[i]
            xBot = DXs[i + 1] - np.cos(skew + np.pi / 2) * dist_mid
            yBot = DYs[i + 1] - np.sin(skew + np.pi / 2) * dist_mid
            CoordBot = [xBot, yBot, zBot]
            CoordTop = [CoordBot[0], CoordBot[1], zTop]

            botNodes = []
            topNodes = []
            eleIDs = []

            for j in range(self.num_piers):  # Define piers of bent (assuming they are the same)
                # NODES, node name: bent_idx_(1-2)_tag
                nodeI = int(str(i + 1) + str(j + 1) + '1' + self.PTag)
                nodeJ = int(str(i + 1) + str(j + 1) + '2' + self.PTag)
                # nodeJ = self.D1Nodes[i+1]
                botNodes.append(nodeI)
                topNodes.append(nodeJ)
                ops.node(nodeI, *CoordBot)
                ops.node(nodeJ, *CoordTop)

                # ELEMENTS, element name: bent_idx_tag
                eleTag = int(str(i + 1) + str(j + 1) + self.PierTag)
                ops.element(eleType, eleTag, nodeI, nodeJ, self.EndTransfTag, IntTag, '-mass',
                            wTOT / g, self.mass_type)
                eleIDs.append(eleTag)

                CoordBot = [CoordBot[0] + np.cos(skew + np.pi / 2) * dist,
                            CoordBot[1] + np.sin(skew + np.pi / 2) * dist, zBot]
                CoordTop = [CoordBot[0], CoordBot[1], zTop]

            self.EleIDsBent.append(eleIDs)
            self.BentStartNodes.append(botNodes)
            self.BentEndNodes.append(topNodes)
            self.EleLoadsBent.append(wTOT)

    def _create_bearingmat(self, idx):
        """
        --------------------------------
        BEARING MATERIAL FOR IDXth JOINT
        --------------------------------
        # TODO:-1 BEARING MATERIALS - ADD BILINEAR SPRINGS

        """
        if self.model['Bearing']['Type'][idx] == 'Elastic':
            matTags = []
            if self.model['Bearing']['h'][idx] < 1e-10:
                eleType = 'zeroLength'
            else:
                eleType = 'twoNodeLink'
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bearing']['kz'][idx])
            matTags.append(self.EndMatTag)  # vertical direction
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bearing']['ky'][idx])
            matTags.append(self.EndMatTag)  # transverse direction
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bearing']['kx'][idx])
            matTags.append(self.EndMatTag)  # longitudinal direction
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bearing']['krz'][idx])
            matTags.append(self.EndMatTag)
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bearing']['kry'][idx])
            matTags.append(self.EndMatTag)
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bearing']['krx'][idx])
            matTags.append(self.EndMatTag)
            matArgs = ['-mat', *matTags, '-dir', 1, 2, 3, 4, 5, 6]

        elif self.model['Bearing']['Type'][idx] == 'elastomericBearingBoucWen':
            eleType = 'elastomericBearingBoucWen'
            Kvert = self.model['Bearing']['Kvert'][idx]  # vertical stiffness G.A/L
            Kinit = self.model['Bearing']['Kinit'][idx]  # initial elastic stiffness in local shear direction
            Fb = self.model['Bearing']['Fb'][idx]  # characteristic strength
            alpha1 = self.model['Bearing']['alpha1'][idx]  # post yield stiffness ratio of linear hardening component
            alpha2 = self.model['Bearing']['alpha2'][idx]  # post yield stiffness ratio of non-linear hardening component
            mu = self.model['Bearing']['mu'][idx]  # exponent of non-linear hardening component
            eta = 1.0  # yielding exponent (sharpness of hysteresis loop corners) (default = 1.0)
            beta = 0.5  # first hysteretic shape parameter (default = 0.5)
            gamma = 0.5  # second hysteretic shape parameter (default = 0.5)
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, Kvert)  # define elastic vertical stiffness material
            matArgs = [Kinit, Fb, alpha1, alpha2, mu, eta, beta, gamma,
                        '-P', self.EndMatTag, '-T', self.SmallMat, '-My', self.SmallMat, '-Mz', self.SmallMat]

        elif self.model['Bearing']['Type'][idx] == 'elastomericBearingPlasticity':
            eleType = 'elastomericBearingPlasticity'
            Kvert = self.model['Bearing']['Kvert'][idx]  # vertical stiffness G.A/L
            Kinit = self.model['Bearing']['Kinit'][idx]  # initial elastic stiffness in local shear direction
            Fb = self.model['Bearing']['Fb'][idx]  # characteristic strength
            alpha1 = self.model['Bearing']['alpha1'][idx]  # post yield stiffness ratio of linear hardening component
            alpha2 = self.model['Bearing']['alpha2'][
                idx]  # post yield stiffness ratio of non-linear hardening component
            mu = self.model['Bearing']['mu'][idx]  # exponent of non-linear hardening component
            self.EndMatTag += 1
            ops.uniaxialMaterial('Elastic', self.EndMatTag, Kvert)  # define elastic vertical stiffness material
            matArgs = [Kinit, Fb, alpha1, alpha2, mu,
                        '-P', self.EndMatTag, '-T', self.SmallMat, '-My', self.SmallMat, '-Mz', self.SmallMat]

        elif self.model['Bearing']['Type'][idx] == 'ElastomericX':
            eleType = 'ElastomericX'
            Fy = self.model['Bearing']['Fy'][idx]  # yield strength
            alpha = self.model['Bearing']['alpha'][idx]  # post-yield stiffness ratio
            G = self.model['Bearing']['G'][idx]  # shear modulus of elastomeric bearing
            K = self.model['Bearing']['K'][idx]  # bulk modulus of rubber
            D1 = self.model['Bearing']['D1'][idx]  # internal diameter
            D2 = self.model['Bearing']['D2'][idx]  # outer diameter (excluding cover thickness)
            ts = self.model['Bearing']['ts'][idx]  # single steel shim layer thickness
            tr = self.model['Bearing']['tr'][idx]  # single rubber layer thickness
            nr = self.model['Bearing']['nr'][idx]  # number of rubber layers
            matArgs = [Fy, alpha, G, K, D1, D2, ts, tr, nr]

        # singleFPBearing with coulomb friction
        elif self.model['Bearing']['Type'][idx] == 'LeadRubberX':
            eleType = 'LeadRubberX'
            Fy = self.model['Bearing']['Fy'][idx]  # yield strength
            alpha = self.model['Bearing']['alpha'][idx]  # post-yield stiffness ratio
            G = self.model['Bearing']['G'][idx]  # shear modulus of elastomeric bearing
            K = self.model['Bearing']['K'][idx]  # bulk modulus of rubber
            D1 = self.model['Bearing']['D1'][idx]  # internal diameter
            D2 = self.model['Bearing']['D2'][idx]  # outer diameter (excluding cover thickness)
            ts = self.model['Bearing']['ts'][idx]  # single steel shim layer thickness
            tr = self.model['Bearing']['tr'][idx]  # single rubber layer thickness
            nr = self.model['Bearing']['nr'][idx]  # number of rubber layers
            matArgs = [Fy, alpha, G, K, D1, D2, ts, tr, nr]

        # singleFPBearing with coulomb friction
        elif self.model['Bearing']['Type'][idx] == 'singleFPBearing_coulomb':
            eleType = 'singleFPBearing'
            R = self.model['Bearing']['R'][idx]
            mu = self.model['Bearing']['mu'][idx]
            K = self.model['Bearing']['K'][idx]
            self.EndMatTag += 1
            ops.frictionModel('Coulomb', self.EndMatTag, mu)
            matArgs = [self.EndMatTag, R, K, '-P', self.BigMat, '-T', self.SmallMat, '-My', self.SmallMat, '-Mz',
                        self.SmallMat]

        # singleFPBearing with velocity dependent friction
        elif self.model['Bearing']['Type'][idx] == 'singleFPBearing_velocity':
            eleType = 'singleFPBearing'
            R = self.model['Bearing']['R'][idx]
            K = self.model['Bearing']['K'][idx]
            muSlow = self.model['Bearing']['muSlow'][idx]
            muFast = self.model['Bearing']['muFast'][idx]
            transRate = self.model['Bearing']['transRate'][idx]
            self.EndMatTag += 1
            ops.frictionModel('VelDependent', self.EndMatTag, muSlow, muFast, transRate)
            matArgs = [self.EndMatTag, R, K, '-P', self.BigMat, '-T', self.SmallMat, '-My', self.SmallMat, '-Mz',
                        self.SmallMat]

        # flatSliderBearing with coulomb friction
        elif self.model['Bearing']['Type'][idx] == 'flatSliderBearing_coulomb':
            eleType = 'flatSliderBearing'
            mu = self.model['Bearing']['mu'][idx]
            K = self.model['Bearing']['K'][idx]
            self.EndMatTag += 1
            ops.frictionModel('Coulomb', self.EndMatTag, mu)
            matArgs = [self.EndMatTag, K, '-P', self.BigMat, '-T', self.SmallMat, '-My', self.SmallMat, '-Mz',
                        self.SmallMat]

        # flatSliderBearing with velocity dependent friction
        elif self.model['Bearing']['Type'][idx] == 'flatSliderBearing_velocity':
            eleType = 'flatSliderBearing'
            K = self.model['Bearing']['K'][idx]
            muSlow = self.model['Bearing']['muSlow'][idx]
            muFast = self.model['Bearing']['muFast'][idx]
            transRate = self.model['Bearing']['transRate'][idx]
            self.EndMatTag += 1
            ops.frictionModel('VelDependent', self.EndMatTag, muSlow, muFast, transRate)
            matArgs = [self.EndMatTag, K, '-P', self.BigMat, '-T', self.SmallMat, '-My', self.SmallMat, '-Mz',
                        self.SmallMat]

        return matArgs, eleType
    
    def _bearing(self):
        """
        --------------------------
        BEARING MODELLING
        --------------------------
        """

        # INFORMATION TO SAVE
        self.AB1Nodes = []  # Bearing bottom nodes at abutment 1
        self.AB2Nodes = []  # Bearing bottom nodes at abutment 2
        self.BcapNodes = []  # Bearing bottom nodes at bentcaps
        self.EleIDsBearing = []  # Bearing element IDs
        
        if self.model['Deck']['Type'] == 'Discontinuous':
            vecx = [0, 0, 1]
            for i in range(self.num_spans + 1):  # AT EACH SPAN END
                # INPUTS
                num_bearings = self.model['Bearing']['N'][i]
                matArgs, eleType = self._create_bearingmat(idx=i)

                if i == 0:
                    skew = self.skew[0]
                    vecyp1 = np.round(np.array([np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0]), 5)
                    vecyp1 = vecyp1.tolist()
                    if eleType != 'ElastomericX' and eleType != 'LeadRubberX':
                        if self.model['Bearing']['h'][i] < 1e-10:
                            ele1Args = [*matArgs, '-orient', *vecx, *vecyp1]  # zerolength element
                        else:
                            ele1Args = [*matArgs, '-orient', *vecyp1]
                    else:
                        ele1Args = [*matArgs, *vecx, *vecyp1]

                elif i == self.num_spans:
                    skew = self.skew[-1]
                    vecyp2 = np.round(np.array([np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0]), 5)
                    vecyp2 = vecyp2.tolist()

                    if eleType != 'ElastomericX' and eleType != 'LeadRubberX':
                        if self.model['Bearing']['h'][i] < 1e-10:
                            ele2Args = [*matArgs, '-orient', *vecx, *vecyp2]  # zerolength element
                        else:
                            ele2Args = [*matArgs, '-orient', *vecyp2]
                    else:
                        ele2Args = [*matArgs, *vecx, *vecyp2]

                else:
                    skew1 = self.skew[i]
                    skew2 = self.skew[i - 1]
                    skew = (skew1 + skew2) / 2
                    vecyp1 = np.round(np.array([np.cos(skew1 + np.pi / 2), np.sin(skew1 + np.pi / 2), 0]), 5)
                    vecyp2 = np.round(np.array([np.cos(skew2 + np.pi / 2), np.sin(skew2 + np.pi / 2), 0]), 5)
                    vecyp1 = vecyp1.tolist()
                    vecyp2 = vecyp2.tolist()

                    if eleType != 'ElastomericX' and eleType != 'LeadRubberX':
                        if self.model['Bearing']['h'][i] < 1e-10:
                            ele1Args = [*matArgs, '-orient', *vecx, *vecyp1]  # zerolength element
                            ele2Args = [*matArgs, '-orient', *vecx, *vecyp2]  # zerolength element
                        else:
                            ele1Args = [*matArgs, '-orient', *vecyp1]
                            ele2Args = [*matArgs, '-orient', *vecyp2]
                    else:
                        ele1Args = [*matArgs, *vecx, *vecyp2]
                        ele2Args = [*matArgs, *vecx, *vecyp2]

                # Nodal Coordinates
                dist_mid = self.model['Bearing']['s'][i] * (num_bearings - 1) / 2
                z2 = self.model['Deck']['Zs'][i] - self.model['Bearing']['dv'][i]
                z1 = z2 - self.model['Bearing']['h'][i]
                z0 = z1 - self.model['BentCap']['h'] / 2
                x0 = round(self.model['Deck']['Xs'][i] - np.cos(skew + np.pi / 2) * dist_mid, 5)
                y0 = round(self.model['Deck']['Ys'][i] - np.sin(skew + np.pi / 2) * dist_mid, 5)

                if i != 0 and i != self.num_spans:
                    coords_mid1 = ops.nodeCoord(self.D1Nodes[i])
                    x01 = round(coords_mid1[0] - np.cos(skew + np.pi / 2) * dist_mid, 5)
                    y01 = round(coords_mid1[1] - np.sin(skew + np.pi / 2) * dist_mid, 5)
                    coords_mid2 = ops.nodeCoord(self.D2Nodes[i - 1])
                    x02 = round(coords_mid2[0] - np.cos(skew + np.pi / 2) * dist_mid, 5)
                    y02 = round(coords_mid2[1] - np.sin(skew + np.pi / 2) * dist_mid, 5)

                Bearings = []
                BcapNodes = []

                for j in range(num_bearings):

                    if i == 0:  # AT START ABUTMENT
                        # pass
                        # NODES, node name: bent_idx_(1-2)_tag
                        Bnode11 = int(str(i + 1) + str(j + 1) + '1' + self.BTag)
                        Bnode21 = int(str(i + 1) + str(j + 1) + '2' + self.BTag)
                        ops.node(Bnode11, x0, y0, z1)
                        ops.node(Bnode21, x0, y0, z2)

                        # RIGID LINKS
                        self.RigidLinkNodes.append([self.D1Nodes[0], Bnode21])

                        # ELEMENTS, element name: bent_idx_1_tag
                        ele1Nodes = [Bnode11, Bnode21]
                        ele1Tag = int(str(i + 1) + str(j + 1) + '1' + self.BearingTag)
                        ops.element(eleType, ele1Tag, *ele1Nodes, *ele1Args)
                        # SAVE
                        self.AB1Nodes.append(Bnode11)
                        Bearings.append(ele1Tag)

                    elif i == self.num_spans:  # AT END ABUTMENT
                        # pass
                        # NODES, node name: bent_idx_(3-4)_tag
                        Bnode12 = int(str(i + 1) + str(j + 1) + '3' + self.BTag)
                        Bnode22 = int(str(i + 1) + str(j + 1) + '4' + self.BTag)
                        ops.node(Bnode12, x0, y0, z1)
                        ops.node(Bnode22, x0, y0, z2)

                        # RIGID LINKS
                        self.RigidLinkNodes.append([self.D2Nodes[-1], Bnode22])

                        # ELEMENTS, element name: bent_idx_2_tag
                        ele2Nodes = [Bnode12, Bnode22]
                        ele2Tag = int(str(i + 1) + str(j + 1) + '2' + self.BearingTag)
                        ops.element(eleType, ele2Tag, *ele2Nodes, *ele2Args)

                        # SAVE
                        self.AB2Nodes.append(Bnode12)
                        Bearings.append(ele2Tag)

                    else:  # AT BENT CAP
                        # NODES# NODES, node name: bent_idx_(1-4)_tag
                        BCnode = int(str(i) + str(j) + self.BC1Tag)
                        Bnode11 = int(str(i + 1) + str(j + 1) + '1' + self.BTag)
                        Bnode21 = int(str(i + 1) + str(j + 1) + '2' + self.BTag)
                        Bnode12 = int(str(i + 1) + str(j + 1) + '3' + self.BTag)
                        Bnode22 = int(str(i + 1) + str(j + 1) + '4' + self.BTag)
                        ops.node(BCnode, x0, y0, z0)
                        ops.node(Bnode11, x01, y01, z1)
                        ops.node(Bnode21, x01, y01, z2)
                        ops.node(Bnode12, x02, y02, z1)
                        ops.node(Bnode22, x02, y02, z2)

                        # RIGID LINKS
                        self.RigidLinkNodes.append([self.D1Nodes[i], Bnode21])
                        self.RigidLinkNodes.append([self.D2Nodes[i - 1], Bnode22])
                        self.RigidLinkNodes.append([BCnode, Bnode11])
                        self.RigidLinkNodes.append([BCnode, Bnode12])

                        # ELEMENTS, element name: bent_idx_(1-2)_tag
                        ele1Nodes = [Bnode11, Bnode21]
                        ele1Tag = int(str(i + 1) + str(j + 1) + '1' + self.BearingTag)
                        ops.element(eleType, ele1Tag, *ele1Nodes, *ele1Args)

                        ele2Nodes = [Bnode12, Bnode22]
                        ele2Tag = int(str(i + 1) + str(j + 1) + '2' + self.BearingTag)
                        ops.element(eleType, ele2Tag, *ele2Nodes, *ele2Args)

                        # UPDATE
                        x01 += np.cos(skew + np.pi / 2) * self.model['Bearing']['s'][i]
                        y01 += np.sin(skew + np.pi / 2) * self.model['Bearing']['s'][i]
                        x02 += np.cos(skew + np.pi / 2) * self.model['Bearing']['s'][i]
                        y02 += np.sin(skew + np.pi / 2) * self.model['Bearing']['s'][i]

                        # SAVE
                        Bearings.append(ele1Tag)
                        Bearings.append(ele2Tag)
                        BcapNodes.append(BCnode)

                    # UPDATE
                    x0 += np.cos(skew + np.pi / 2) * self.model['Bearing']['s'][i]
                    y0 += np.sin(skew + np.pi / 2) * self.model['Bearing']['s'][i]

                    # SAVE
                self.EleIDsBearing.append(Bearings)
                if any(BcapNodes): self.BcapNodes.append(BcapNodes)

        elif self.model['Deck']['Type'] == 'Continuous':
            vecx = [0, 0, 1]
            for i in range(self.num_spans + 1):  # AT EACH SPAN END
                num_bearings = self.model['Bearing']['N'][i]
                matArgs, eleType = self._create_bearingmat(idx=i)

                if i == 0:
                    skew = self.skew[0]
                elif i == self.num_spans:
                    skew = self.skew[-1]
                else:
                    skew = (self.skew[i] + self.skew[i - 1]) / 2

                vecyp = np.round(np.array([np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0]), 5)
                vecyp = vecyp.tolist()
                if eleType != 'ElastomericX' and eleType != 'LeadRubberX':
                    if self.model['Bearing']['h'][i] < 1e-10:
                        eleArgs = [*matArgs, '-orient', *vecx, *vecyp]  # zerolength element
                    else:
                        eleArgs = [*matArgs, '-orient', *vecyp]
                else:
                    eleArgs = [*matArgs, *vecx, *vecyp]

                # Nodal Coordinates
                dist_mid = self.model['Bearing']['s'][i] * (num_bearings - 1) / 2
                z2 = self.model['Deck']['Zs'][i] - self.model['Bearing']['dv'][i]
                z1 = z2 - self.model['Bearing']['h'][i]
                z0 = z1 - self.model['BentCap']['h'] / 2
                x0 = round(self.model['Deck']['Xs'][i] - np.cos(skew + np.pi / 2) * dist_mid, 5)
                y0 = round(self.model['Deck']['Ys'][i] - np.sin(skew + np.pi / 2) * dist_mid, 5)

                Bearings = []
                BcapNodes = []

                for j in range(num_bearings):

                    # NODES, node name: bent_idx_(1-2)_tag
                    Bnode1 = int(str(i + 1) + str(j + 1) + '1' + self.BTag)
                    Bnode2 = int(str(i + 1) + str(j + 1) + '2' + self.BTag)
                    ops.node(Bnode1, x0, y0, z1)
                    ops.node(Bnode2, x0, y0, z2)

                    # ELEMENTS, element name: bent_idx_1_tag
                    eleNodes = [Bnode1, Bnode2]
                    eleTag = int(str(i + 1) + str(j + 1) + '0' + self.BearingTag)
                    ops.element(eleType, eleTag, *eleNodes, *eleArgs)
                    Bearings.append(eleTag)

                    # RIGID LINKS
                    if i == 0:  # AT START ABUTMENT
                        self.RigidLinkNodes.append([self.D1Nodes[0], Bnode2])
                        self.AB1Nodes.append(Bnode1)
                    elif i == self.num_spans:  # AT END ABUTMENT
                        self.RigidLinkNodes.append([self.D2Nodes[-1], Bnode2])
                        self.AB2Nodes.append(Bnode1)
                    else:  # AT BENTS
                        BCnode = int(str(i) + str(j) + self.BC1Tag)  # Node on bentcap
                        ops.node(BCnode, x0, y0, z0)
                        BcapNodes.append(BCnode)
                        self.RigidLinkNodes.append([BCnode, Bnode1])
                        self.RigidLinkNodes.append([self.D1Nodes[i], Bnode2])

                    # UPDATE
                    x0 += round(np.cos(skew + np.pi / 2) * self.model['Bearing']['s'][i], 5)
                    y0 += round(np.sin(skew + np.pi / 2) * self.model['Bearing']['s'][i], 5)

                # SAVE
                self.EleIDsBearing.append(Bearings)
                if any(BcapNodes): self.BcapNodes.append(BcapNodes)
        
    def _bentcap(self):
        """
        --------------------------
        BENTCAP MODELLING
        --------------------------
        """
        # INPUTS
        DXs = self.model['Deck']['Xs']
        DYs = self.model['Deck']['Ys']
        DZs = self.model['Deck']['Zs']
        E = self.model['BentCap']['E']
        G = self.model['BentCap']['G']
        h = self.model['BentCap']['h']
        w = self.model['BentCap']['w']
        L = self.model['BentCap']['L']

        if all(num == 1 for num in self.model['Bearing']['N']) and self.num_piers == 1:

            self.PointLoadsBcap = []  # BentCap point Loads
            for i in range(self.num_bents):  # do not model bentcap (simply there is no need)
                self.RigidLinkNodes.append([self.BcapNodes[i][0], self.BentEndNodes[i][0]])
                Pload = h * w * L * gamma_c
                mass = Pload / g
                ops.mass(self.BcapNodes[i][0], mass, mass, mass, 0, 0, 0)
                self.PointLoadsBcap.append(Pload)
            pass
        else:
            # INFORMATION TO SAVE
            self.EleIDsBcap = []  # BentCap element IDs       
            self.EleLoadsBcap = []  # BentCap element Loads

            # SECTION PROPERTIES - CONSTANT AND RECTANGULAR FOR ALL
            A, Iz, Iy, J = getRectProp(h, w)
            self.EndSecTag += 1
            ops.section('Elastic', self.EndSecTag, E, A, Iz, Iy, G, J)
            self.EndIntTag += 1
            ops.beamIntegration('Legendre', self.EndIntTag, self.EndSecTag, 2)
            self.EndTransfTag += 1  # local z is in global x
            ops.geomTransf('Linear', self.EndTransfTag, 0, 0, 1)
            wTOT = gamma_c * A

            # For each bent
            for i in range(self.num_bents):

                # These are nodes where bearings are rigidly connected to
                nodes_bearing = self.BcapNodes[i]

                # NODES, node name: bent_idx_tag
                dist_list = []
                # skew angle of the bent
                skew = (self.skew[i] + self.skew[i + 1]) / 2
                # Nodal Coordinates
                dist_mid = self.model['BentCap']['L'] / 2
                z = DZs[i + 1] - self.model['Bearing']['dv'][i + 1] - self.model['Bearing']['h'][i + 1] - h / 2
                x0 = DXs[i + 1] - np.cos(skew + np.pi / 2) * dist_mid
                y0 = DYs[i + 1] - np.sin(skew + np.pi / 2) * dist_mid
                x1 = DXs[i + 1] + np.cos(skew + np.pi / 2) * dist_mid
                y1 = DYs[i + 1] + np.sin(skew + np.pi / 2) * dist_mid

                # BentCap first and last node
                idx = 1
                n1_id = int(str(i + 1) + str(idx) + self.BC3Tag)
                ops.node(n1_id, x0, y0, z)
                self.BcapNodes[i].append(n1_id)
                idx += 1
                n_id = int(str(i + 1) + str(idx) + self.BC3Tag)
                ops.node(n_id, x1, y1, z)
                self.BcapNodes[i].append(n_id)

                idx = 1
                for node in self.BentEndNodes[i]:  # Bentcap nodes connected to pier
                    coords = np.array(ops.nodeCoord(node))
                    coords[2] = z

                    flag = 0  # Do this check to avoid creating unnecessary nodes
                    for node2 in nodes_bearing:
                        diff = coords - np.array(ops.nodeCoord(node2))
                        is_all_zero = np.all((diff == 0))
                        if is_all_zero:
                            flag = 1
                            n_id = node2
                            break

                    if flag == 0:
                        idx += 1
                        n_id = int(str(i + 1) + str(idx) + self.BC2Tag)
                        ops.node(n_id, coords[0], coords[1], z)
                        self.BcapNodes[i].append(n_id)

                    # RIGID LINKS
                    self.RigidLinkNodes.append([node, n_id])

                # sort nodes by their distance to the start node
                coord1 = [x0, y0, z]
                for node in self.BcapNodes[i]:
                    coord2 = ops.nodeCoord(node)
                    dist_list.append(distance(coord1, coord2))
                self.BcapNodes[i] = [x for _, x in sorted(zip(dist_list, self.BcapNodes[i]))]

                # ELEMENTS
                for j in range(len(self.BcapNodes[i]) - 1):
                    nodeI = self.BcapNodes[i][j]
                    nodeJ = self.BcapNodes[i][j + 1]
                    eleTag = int(str(i + 1) + str(j + 1) + self.BentCapTag)
                    ops.element('dispBeamColumn', eleTag, nodeI, nodeJ, self.EndTransfTag, self.EndIntTag, '-mass',
                                wTOT / g, self.mass_type)
                    self.EleIDsBcap.append(eleTag)
                    self.EleLoadsBcap.append(wTOT)

    def _abutment(self):
        """
        --------------------------------------------
        MODELLING OF ABUTMENT BACKFILL SOIL
        --------------------------------------------
        TODO: Option of Hyperbolic Gap material 
        TODO: distributed soil springs (more than 2)
        """
        
        if self.model['Abutment_Foundation']['Type'] == 'Fixed':
            KvMat = self.BigMat
            KRxMat = self.BigMat
            KRyMat = self.BigMat
            KRzMat = self.BigMat
        else:
            KvMat = self.SmallMat
            KRxMat = self.SmallMat
            KRyMat = self.SmallMat
            KRzMat = self.SmallMat
            
        if self.model['Abutment_BackFill']['Type'] == 'None':
            self.fixed_AB1Nodes = []
            self.fixed_AB2Nodes = []
            pass

        elif self.model['Abutment_BackFill']['Type'] == 'UserDefined':
            # INPUTS
            Fyx = self.model['Abutment_BackFill']['Fyx']
            Fyy = self.model['Abutment_BackFill']['Fyy']
            bx = self.model['Abutment_BackFill']['bx']
            by = self.model['Abutment_BackFill']['by']
            Kx = self.model['Abutment_BackFill']['Kx']
            Ky = self.model['Abutment_BackFill']['Ky']
            skew1 = self.skew[0]
            skew2 = self.skew[-1]

            MatTags1 = []
            self.EndMatTag += 1
            MatTags1.append(self.EndMatTag)
            if Fyx[0] == 0:
                ops.uniaxialMaterial('Elastic', self.EndMatTag, Kx[0])
            else:
                ops.uniaxialMaterial('Steel01', self.EndMatTag, Fyx[0], Kx[0], bx[0])
            self.EndMatTag += 1
            MatTags1.append(self.EndMatTag)
            if Fyy[0] == 0:
                ops.uniaxialMaterial('Elastic', self.EndMatTag, Ky[0])
            else:
                ops.uniaxialMaterial('Steel01', self.EndMatTag, Fyy[0], Ky[0], bx[0])
            MatTags1.append(KvMat)
            MatTags1.append(KRxMat)
            MatTags1.append(KRyMat)
            MatTags1.append(KRzMat)

            MatTags2 = []
            self.EndMatTag += 1
            MatTags2.append(self.EndMatTag)
            if Fyx[0] == 0:
                ops.uniaxialMaterial('Elastic', self.EndMatTag, Kx[1])
            else:
                ops.uniaxialMaterial('Steel01', self.EndMatTag, Fyx[1], Kx[1], bx[1])
            self.EndMatTag += 1
            MatTags2.append(self.EndMatTag)
            if Fyy[0] == 0:
                ops.uniaxialMaterial('Elastic', self.EndMatTag, Ky[1])
            else:
                ops.uniaxialMaterial('Steel01', self.EndMatTag, Fyy[1], Ky[1], by[1])
            MatTags2.append(KvMat)
            MatTags2.append(KRxMat)
            MatTags2.append(KRyMat)
            MatTags2.append(KRzMat)

            # Start Abutment Coordinates
            Abut1_C = int('11' + self.ATag)  # Centroid for rigid links to bearings
            Abut1_CF = int('10' + self.ATag)  # fixed end
            Coords1_C = np.array(
                [self.model['Deck']['Xs'][0], self.model['Deck']['Ys'][0], self.model['Deck']['Zs'][0]],
                dtype=float)
            Coords1_C[2] = Coords1_C[2] - (self.model['Bearing']['dv'][0] + self.model['Bearing']['h'][0])

            # End Abutment Coordinates
            Abut2_C = int('21' + self.ATag)  # Centroid for rigid links to bearings
            Abut2_CF = int('20' + self.ATag)  # fixed end
            Coords2_C = np.array(
                [self.model['Deck']['Xs'][-1], self.model['Deck']['Ys'][-1], self.model['Deck']['Zs'][-1]],
                dtype=float)
            Coords2_C[2] = Coords2_C[2] - (self.model['Bearing']['dv'][-1] + self.model['Bearing']['h'][-1])

            Coords1_C = np.round(Coords1_C, 5)
            Coords2_C = np.round(Coords2_C, 5)

            # Create Nodes for Abutment Springs
            ops.node(Abut1_C, *Coords1_C.tolist(), '-mass', 0, 0, 0, 0, 0, 0)
            ops.node(Abut1_CF, *Coords1_C.tolist())
            ops.node(Abut2_C, *Coords2_C.tolist(), '-mass', 0, 0, 0, 0, 0, 0)
            ops.node(Abut2_CF, *Coords2_C.tolist())

            dirs = [1, 2, 3, 4, 5, 6]
            ops.element('zeroLength', int('10' + self.AbutTag), Abut1_CF, Abut1_C, '-mat', *MatTags1, '-dir', *dirs,
                        '-orient', np.cos(skew1), np.sin(skew1), 0, 0, 1, 0)
            ops.element('zeroLength', int('20' + self.AbutTag), Abut2_CF, Abut2_C, '-mat', *MatTags2, '-dir', *dirs,
                        '-orient', np.cos(skew2), np.sin(skew2), 0, 0, 1, 0)

            # Fix the springs
            ops.fix(Abut1_CF, 1, 1, 1, 1, 1, 1)
            ops.fix(Abut2_CF, 1, 1, 1, 1, 1, 1)

            # Rigid links
            for node in self.AB1Nodes:
                self.RigidLinkNodes.append([Abut1_C, node])
            for node in self.AB2Nodes:
                self.RigidLinkNodes.append([Abut2_C, node])

            self.fixed_AB1Nodes = [Abut1_CF]
            self.fixed_AB2Nodes = [Abut2_CF]
            self.fixed_AB1Nodes_backfill = [Abut1_CF]
            self.fixed_AB2Nodes_backfill = [Abut2_CF]
            self.AB1_Cnode = Abut1_C
            self.AB2_Cnode = Abut2_C

            self.EleIDsAB1 = [int('10' + self.AbutTag)]
            self.EleIDsAB2 = [int('20' + self.AbutTag)]

        elif self.model['Abutment_BackFill']['Type'] == 'SDC 2019':
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
            h = self.model['Abutment_BackFill']['height']
            w = self.model['Abutment_BackFill']['width']
            b = self.model['Abutment_BackFill']['breadth']
            gap = self.model['Abutment_BackFill']['gap']
            skew1 = self.skew[0]
            skew2 = self.skew[-1]
            mass = h * w * b / 3  # approximate abutment mass
            # mass = 0

            # # Create gap elements in longitudinal direction, if gap is a nonzero value
            # if gap != 0 and self.model['Abutment_BackFill']['gapFlag'] == 1:

            #     self.EndMatTag += 1
            #     ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, 1e10, -1e10, -gap, 0, 'damage')
            #     vecyp = [np.cos(skew1 + np.pi / 2), np.sin(skew1 + np.pi / 2), 0]
            #     count = 1
            #     temp1 = []
            #     # Gap elements at start abutment (just place at far most left and right elements)

            #     eles1 = [self.EleIDsBearing[0][0]]
            #     eles2 = [self.EleIDsBearing[-1][0]]
            #     if self.model['Bearing']['N'][0] != 1:
            #         eles1.append(self.EleIDsBearing[0][-1])
            #     if self.model['Bearing']['N'][-1] != 1:
            #         eles2.append(self.EleIDsBearing[-1][-1])

            #     for ele in eles1:
            #         eleNodes = ops.eleNodes(ele)
            #         eleTag = int(str(count) + self.GapTag)
            #         ops.element('twoNodeLink', eleTag, *eleNodes, '-mat', self.EndMatTag, '-dir', 3, '-orient', *vecyp)
            #         count += 1
            #         temp1.append(eleTag)

            #     vecyp = [np.cos(skew2 + np.pi / 2), np.sin(skew2 + np.pi / 2), 0]
            #     temp2 = []
            #     # Gap elements at end abutment
            #     for ele in eles2:
            #         eleNodes = ops.eleNodes(ele)
            #         eleTag = int(str(count) + self.GapTag)
            #         ops.element('twoNodeLink', int(str(count) + self.GapTag), *eleNodes, '-mat', self.EndMatTag, '-dir',
            #                     3, '-orient', *vecyp)
            #         count += 1
            #         temp2.append(eleTag)

            #     self.EleIDsGap = [temp1, temp2]

            # # Vertical springs - Jian Zhang and Nicos Makris (2002) But I am ignoring this
            # S = np.sin(60 * degrees)  # Embankment slope
            # Lc = 0.7 * (h * w * S)  # Crtical Length Equation 46
            # z0 = 0.5 * S * w
            # v = 0.3  # typical soil poissons ratio
            # Vs = 500  # assume some relatively stiff soil
            # rho = 18 * kN / m ** 3 / g  # some standard soil
            # Gsoil = rho * Vs ** 2
            # Esoil = 2 * (1 + v) * Gsoil
            # Kv = (Esoil * w) / (z0 * np.log((z0 + h) / h)) * Lc  # Equation 20

            # Longitudinal Springs / CALTRANS - SDC 2019: Section 6.3.1.2
            if h < 2 * ft:
                h = 2 * ft
            if h > 10 * ft:
                h = 10 * ft
            h_ft = h / ft
            w_ft = w / ft  # convert to ft
            Rsk_1 = np.exp(-skew1 / 45)  # theta < 66
            PabutL_1 = w_ft * (5.5 * h_ft ** 2.5) / (1 + 2.37 * h_ft) * Rsk_1 * kip  # convert to kN
            KabutL_1 = w_ft * (5.5 * h_ft + 20) * Rsk_1 * kip / inch  # convert to kN/m
            dy_L1 = PabutL_1 / KabutL_1

            Rsk_2 = np.exp(-skew2 / 45)  # theta < 66
            PabutL_2 = w_ft * (5.5 * h_ft ** 2.5) / (1 + 2.37 * h_ft) * Rsk_2 * kip  # convert to kN
            KabutL_2 = w_ft * (5.5 * h_ft + 20) * Rsk_2 * kip / inch  # convert to kN/m
            dy_L2 = PabutL_2 / KabutL_2

            # Instead of gap elements effective stiffness can be used, to avoid analysis problems
            if self.model['Abutment_BackFill']['gapFlag'] == 0:
                KabutL_1 = PabutL_1 / (dy_L1 + gap)
                KabutL_2 = PabutL_2 / (dy_L2 + gap)

            # # Transverse Springs / CALTRANS - SDC 2019: Section 4.3.1, Figure 4.3.1-2
            # # The assumption is that sacrificial shear key is used where gap is 2 inches
            # dy_T1 = 2 * inch
            # PabutT_1 = 0.3 * self.AB1AxialForces
            # KabutT_1 = PabutT_1 / dy_T1

            # dy_T2 = 2 * inch
            # PabutT_2 = 0.3 * self.AB2AxialForces
            # KabutT_2 = PabutT_2 / dy_T2

            # Alternatively, approach by Maroney and Chai (1994) can be followed
            # Transverse backfill pressure facator is CL*CW = 2/3*4/3 according to Maroney and Chai (1994).
            CL = 2 / 3  # Wingwall effectiveness coefficient
            CW = 4 / 3  # Wingwall participation coefficient
            ratio = 0.5  # The wing wall length can be assumed 1/2-1/3 of the back-wall length
            PabutT_1 = PabutL_1 * ratio * CL * CW
            KabutT_1 = KabutL_1 * ratio * CL * CW
            # dy_T1 = PabutT_1/KabutT_1
            PabutT_2 = PabutL_2 * ratio * CL * CW
            KabutT_2 = KabutL_2 * ratio * CL * CW
            # dy_T2 = PabutT_2/KabutT_2

            if self.model['Abutment_BackFill']['spring'] == 1:
                r = 1e-4  # strain hardening to prevent 0 stiffness
                MatTags1 = []
                self.EndMatTag += 1
                MatTags1.append(self.EndMatTag)

                if gap != 0 and self.model['Abutment_BackFill']['gapFlag'] == 1:
                    ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, KabutL_1 / 2, -PabutL_1, -gap, 0, 'damage')
                else:
                    ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutL_1, KabutL_1, r)

                self.EndMatTag += 1
                MatTags1.append(self.EndMatTag)
                ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutT_1, KabutT_1, r)
                MatTags1.append(KvMat)
                MatTags1.append(KRxMat)
                MatTags1.append(KRyMat)
                MatTags1.append(KRzMat)

                MatTags2 = []
                self.EndMatTag += 1
                MatTags2.append(self.EndMatTag)
                if gap != 0 and self.model['Abutment_BackFill']['gapFlag'] == 1:
                    ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, KabutL_2 / 2, PabutL_2, gap, 0, 'damage')
                else:
                    ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutL_2, KabutL_2, r)

                self.EndMatTag += 1
                MatTags2.append(self.EndMatTag)
                ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutT_2, KabutT_2, r)
                MatTags2.append(KvMat)
                MatTags2.append(KRxMat)
                MatTags2.append(KRyMat)
                MatTags2.append(KRzMat)

                # Start Abutment Coordinates
                Abut1_C = int('11' + self.ATag)  # Centroid for rigid links to bearings
                Abut1_CF = int('10' + self.ATag)  # fixed end
                Coords1_C = np.array(
                    [self.model['Deck']['Xs'][0], self.model['Deck']['Ys'][0], self.model['Deck']['Zs'][0]],
                    dtype=float)
                Coords1_C[2] = Coords1_C[2] - (self.model['Bearing']['dv'][0] + self.model['Bearing']['h'][0])

                # End Abutment Coordinates
                Abut2_C = int('21' + self.ATag)  # Centroid for rigid links to bearings
                Abut2_CF = int('20' + self.ATag)  # fixed end
                Coords2_C = np.array(
                    [self.model['Deck']['Xs'][-1], self.model['Deck']['Ys'][-1], self.model['Deck']['Zs'][-1]],
                    dtype=float)
                Coords2_C[2] = Coords2_C[2] - (self.model['Bearing']['dv'][-1] + self.model['Bearing']['h'][-1])

                Coords1_C = np.round(Coords1_C, 5)
                Coords2_C = np.round(Coords2_C, 5)

                # Create Nodes for Abutment Springs
                ops.node(Abut1_C, *Coords1_C.tolist(), '-mass', mass, mass, mass, 0, 0, 0)
                ops.node(Abut1_CF, *Coords1_C.tolist())
                ops.node(Abut2_C, *Coords2_C.tolist(), '-mass', mass, mass, mass, 0, 0, 0)
                ops.node(Abut2_CF, *Coords2_C.tolist())

                dirs = [1, 2, 3, 4, 5, 6]
                ops.element('zeroLength', int('10' + self.AbutTag), Abut1_CF, Abut1_C, '-mat', *MatTags1, '-dir', *dirs,
                            '-orient', np.cos(skew1), np.sin(skew1), 0, 0, 1, 0)
                ops.element('zeroLength', int('20' + self.AbutTag), Abut2_CF, Abut2_C, '-mat', *MatTags2, '-dir', *dirs,
                            '-orient', np.cos(skew2), np.sin(skew2), 0, 0, 1, 0)

                # Fix the springs
                ops.fix(Abut1_CF, 1, 1, 1, 1, 1, 1)
                ops.fix(Abut2_CF, 1, 1, 1, 1, 1, 1)

                # Rigid links
                for node in self.AB1Nodes:
                    self.RigidLinkNodes.append([Abut1_C, node])
                for node in self.AB2Nodes:
                    self.RigidLinkNodes.append([Abut2_C, node])

                self.fixed_AB1Nodes = [Abut1_CF]
                self.fixed_AB2Nodes = [Abut2_CF]
                self.fixed_AB1Nodes_backfill = [Abut1_CF]
                self.fixed_AB2Nodes_backfill = [Abut2_CF]
                self.AB1_Cnode = Abut1_C
                self.AB2_Cnode = Abut2_C

                self.EleIDsAB1 = [int('10' + self.AbutTag)]
                self.EleIDsAB2 = [int('20' + self.AbutTag)]

            elif self.model['Abutment_BackFill']['spring'] == 2:
                r = 1e-4  # strain hardening to prevent 0 stiffness
                MatTags1 = []
                self.EndMatTag += 1
                MatTags1.append(self.EndMatTag)
                if gap != 0 and self.model['Abutment_BackFill']['gapFlag'] == 1:
                    ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, KabutL_1 / 2, -PabutL_1, -gap, 0, 'damage')
                else:
                    ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutL_1 / 2, KabutL_1 / 2, r)

                self.EndMatTag += 1
                MatTags1.append(self.EndMatTag)
                ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutT_1 / 2, KabutT_1 / 2, r)
                MatTags1.append(KvMat)
                MatTags1.append(KRxMat)
                MatTags1.append(KRyMat)
                MatTags1.append(KRzMat)

                MatTags2 = []
                self.EndMatTag += 1
                MatTags2.append(self.EndMatTag)
                if gap != 0 and self.model['Abutment_BackFill']['gapFlag'] == 1:
                    ops.uniaxialMaterial('ElasticPPGap', self.EndMatTag, KabutL_2 / 2, PabutL_2, gap, 0, 'damage')
                else:
                    ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutL_2 / 2, KabutL_2 / 2, r)

                self.EndMatTag += 1
                MatTags2.append(self.EndMatTag)
                ops.uniaxialMaterial('Steel01', self.EndMatTag, PabutT_2 / 2, KabutT_2 / 2, r)
                MatTags2.append(KvMat)
                MatTags2.append(KRxMat)
                MatTags2.append(KRyMat)
                MatTags2.append(KRzMat)

                # Start Abutment Coordinates
                Abut1_C = int('10' + self.ATag)  # Centroid for rigid links
                Abut1_R = int('11' + self.ATag)  # Abutment springs here
                Abut1_RF = int('12' + self.ATag)  # Fixed
                Abut1_L = int('13' + self.ATag)  # Abutment springs here
                Abut1_LF = int('14' + self.ATag)  # Fixed
                Coords1_C = np.array(
                    [self.model['Deck']['Xs'][0], self.model['Deck']['Ys'][0], self.model['Deck']['Zs'][0]],
                    dtype=float)
                Coords1_R = np.array(
                    [self.model['Deck']['Xs'][0], self.model['Deck']['Ys'][0], self.model['Deck']['Zs'][0]],
                    dtype=float)
                Coords1_L = np.array(
                    [self.model['Deck']['Xs'][0], self.model['Deck']['Ys'][0], self.model['Deck']['Zs'][0]],
                    dtype=float)
                Coords1_C[2] = Coords1_C[2] - (self.model['Bearing']['dv'][0] + self.model['Bearing']['h'][0])
                Coords1_R[2] = Coords1_C[2]
                Coords1_L[2] = Coords1_C[2]
                Coords1_R[0] -= np.cos(skew1 + np.pi / 2) * w / 2
                Coords1_R[1] -= np.sin(skew1 + np.pi / 2) * w / 2
                Coords1_L[0] += np.cos(skew1 + np.pi / 2) * w / 2
                Coords1_L[1] += np.sin(skew1 + np.pi / 2) * w / 2

                # End Abutment Coordinates
                Abut2_C = int('20' + self.ATag)  # Centroid for rigid links
                Abut2_R = int('21' + self.ATag)  # Abutment springs here
                Abut2_RF = int('22' + self.ATag)  # Fixed
                Abut2_L = int('23' + self.ATag)  # Abutment springs here
                Abut2_LF = int('24' + self.ATag)  # Fixed
                Coords2_C = np.array(
                    [self.model['Deck']['Xs'][-1], self.model['Deck']['Ys'][-1], self.model['Deck']['Zs'][-1]],
                    dtype=float)
                Coords2_R = np.array(
                    [self.model['Deck']['Xs'][-1], self.model['Deck']['Ys'][-1], self.model['Deck']['Zs'][-1]],
                    dtype=float)
                Coords2_L = np.array(
                    [self.model['Deck']['Xs'][-1], self.model['Deck']['Ys'][-1], self.model['Deck']['Zs'][-1]],
                    dtype=float)
                Coords2_C[2] = Coords2_C[2] - (self.model['Bearing']['dv'][-1] + self.model['Bearing']['h'][-1])
                Coords2_R[2] = Coords2_C[2]
                Coords2_L[2] = Coords2_C[2]
                Coords2_R[0] -= np.cos(skew1 + np.pi / 2) * w / 2
                Coords2_R[1] -= np.sin(skew1 + np.pi / 2) * w / 2
                Coords2_L[0] += np.cos(skew1 + np.pi / 2) * w / 2
                Coords2_L[1] += np.sin(skew1 + np.pi / 2) * w / 2

                Coords1_C = np.round(Coords1_C, 5)
                Coords1_R = np.round(Coords1_R, 5)
                Coords1_L = np.round(Coords1_L, 5)
                Coords2_C = np.round(Coords2_C, 5)
                Coords2_R = np.round(Coords2_R, 5)
                Coords2_L = np.round(Coords2_L, 5)

                # Create Nodes for Abutment Springs
                ops.node(Abut1_C, *Coords1_C.tolist())
                ops.node(Abut1_R, *Coords1_R.tolist())
                ops.node(Abut1_RF, *Coords1_R.tolist())
                ops.node(Abut1_L, *Coords1_L.tolist())
                ops.node(Abut1_LF, *Coords1_L.tolist())
                ops.node(Abut2_C, *Coords2_C.tolist())
                ops.node(Abut2_R, *Coords2_R.tolist())
                ops.node(Abut2_RF, *Coords2_R.tolist())
                ops.node(Abut2_L, *Coords2_L.tolist())
                ops.node(Abut2_LF, *Coords2_L.tolist())

                dirs = [1, 2, 3, 4, 5, 6]
                ops.element('zeroLength', int('11' + self.AbutTag), Abut1_RF, Abut1_R, '-mat', *MatTags1, '-dir', *dirs,
                            '-orient', np.cos(skew1), np.sin(skew1), 0, 0, 1, 0)
                ops.element('zeroLength', int('12' + self.AbutTag), Abut1_LF, Abut1_L, '-mat', *MatTags1, '-dir', *dirs,
                            '-orient', np.cos(skew1), np.sin(skew1), 0, 0, 1, 0)
                ops.element('zeroLength', int('21' + self.AbutTag), Abut2_RF, Abut2_R, '-mat', *MatTags2, '-dir', *dirs,
                            '-orient', np.cos(skew2), np.sin(skew2), 0, 0, 1, 0)
                ops.element('zeroLength', int('22' + self.AbutTag), Abut2_LF, Abut2_L, '-mat', *MatTags2, '-dir', *dirs,
                            '-orient', np.cos(skew2), np.sin(skew2), 0, 0, 1, 0)

                # Fix the springs
                ops.fix(Abut1_RF, 1, 1, 1, 1, 1, 1)
                ops.fix(Abut1_LF, 1, 1, 1, 1, 1, 1)
                ops.fix(Abut2_RF, 1, 1, 1, 1, 1, 1)
                ops.fix(Abut2_LF, 1, 1, 1, 1, 1, 1)

                self.AB1Nodes.append(Abut1_C)
                self.AB1Nodes.append(Abut1_R)
                self.AB1Nodes.append(Abut1_L)
                # sort nodes by their distance to the start node
                dist_list = []
                for node in self.AB1Nodes:
                    coord2 = ops.nodeCoord(node)
                    dist_list.append(distance(Coords1_L.tolist(), coord2))
                self.AB1Nodes = [x for _, x in sorted(zip(dist_list, self.AB1Nodes))]

                self.AB2Nodes.append(Abut2_C)
                self.AB2Nodes.append(Abut2_R)
                self.AB2Nodes.append(Abut2_L)
                # sort nodes by their distance to the start node
                dist_list = []
                for node in self.AB2Nodes:
                    coord2 = ops.nodeCoord(node)
                    dist_list.append(distance(Coords2_L.tolist(), coord2))
                self.AB2Nodes = [x for _, x in sorted(zip(dist_list, self.AB2Nodes))]

                for i in range(len(self.AB1Nodes) - 1):
                    self.RigidLinkNodes.append([self.AB1Nodes[i], self.AB1Nodes[i + 1]])
                    self.RigidLinkNodes.append([self.AB2Nodes[i], self.AB2Nodes[i + 1]])

                self.fixed_AB1Nodes = [Abut1_RF, Abut1_LF]
                self.fixed_AB2Nodes = [Abut2_RF, Abut2_LF]
                self.fixed_AB1Nodes_backfill = [Abut1_RF, Abut1_LF]
                self.fixed_AB2Nodes_backfill = [Abut2_RF, Abut2_LF]
                self.AB1_Cnode = Abut1_C
                self.AB2_Cnode = Abut2_C

                self.EleIDsAB1 = [int('11' + self.AbutTag), int('12' + self.AbutTag)]
                self.EleIDsAB2 = [int('21' + self.AbutTag), int('22' + self.AbutTag)]

            self.AbutCNodes = [Abut1_C, Abut2_C]

    def _foundation(self):
        """
        --------------------------------------------
        FOUNDATION MODELLING
        --------------------------------------------
        TODO: p-y springs are not defined for liquefaction susceptible soils, t-z and q-z springs are defined
        based on sand, the equations to derive qult and tult might be different for clays.
        """
        count1 = 0  # counter for nodes with Found1Tag
        count2 = 0  # counter for nodes with Found2Tag
        count3 = 0  # counter for nodes with Found3Tag
        count4 = 0  # counter for nodes with Found4Tag
        count5 = 0  # counter for elements with SpringEleTag
        count6 = 0  # counter for elements with PileEleTag

        # Fix the nodes at abutment
        if self.model['Abutment_BackFill']['Type'] == 'None' and self.model['Abutment_Foundation']['Type'] == 'Fixed':
            for node in self.AB1Nodes:
                ops.fix(node, 1, 1, 1, 1, 1, 1)
            for node in self.AB2Nodes:
                ops.fix(node, 1, 1, 1, 1, 1, 1)

            self.fixed_AB1Nodes = self.AB1Nodes.copy()
            self.fixed_AB2Nodes = self.AB2Nodes.copy()

        elif self.model['Abutment_Foundation']['Type'] == 'Springs':

            dirs = [1, 2, 3, 4, 5, 6]
            self.fixed_AB1Nodes_found = []
            self.fixed_AB2Nodes_found = []
            for i in range(2):
                # NODES
                count1 += 1
                count2 += 1
                nodeI = int(str(count1) + self.Found1Tag)
                nodeJ = int(str(count2) + self.Found2Tag)
                matTags = []
                # Define skew angle of the bent
                if i == 0:
                    skew = self.skew[0]
                    self.fixed_AB1Nodes_found.append(nodeI)
                    self.fixed_AB1Nodes.append(nodeI)
                elif i == 1:
                    skew = self.skew[0]
                    self.fixed_AB2Nodes_found.append(nodeI)
                    self.fixed_AB2Nodes.append(nodeI)
                vecx = [np.cos(skew), np.sin(skew), 0]
                vecyp = [np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0]

                if self.model['Abutment_BackFill']['Type'] != 'None':
                    if i == 0:
                        Coords = np.array(ops.nodeCoord(self.AB1_Cnode))
                        self.RigidLinkNodes.append([self.AB1_Cnode, nodeJ])  # RIGID LINKS


                    elif i == 1:
                        Coords = np.array(ops.nodeCoord(self.AB2_Cnode))
                        self.RigidLinkNodes.append([self.AB2_Cnode, nodeJ])  # RIGID LINKS

                elif i == 0:
                    Coords = np.zeros(3)
                    for node in self.AB1Nodes:
                        Coords += np.array(ops.nodeCoord(node))
                        self.RigidLinkNodes.append([nodeJ, node])  # RIGID LINKS
                    Coords = Coords / len(self.AB1Nodes)

                elif i == 1:
                    Coords = np.zeros(3)
                    for node in self.AB2Nodes:
                        Coords += np.array(ops.nodeCoord(node))
                        self.RigidLinkNodes.append([nodeJ, node])  # RIGID LINKS
                    Coords = Coords / len(self.AB2Nodes)

                Coords[2] = Coords[2] - self.model['Abutment_Foundation']['Height'][i]
                ops.node(nodeI, *Coords.tolist())
                ops.node(nodeJ, *Coords.tolist())
                ops.fix(nodeI, 1, 1, 1, 1, 1, 1)

                # MATERIALS
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Abutment_Foundation']['Kx'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Abutment_Foundation']['Ky'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Abutment_Foundation']['Kz'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Abutment_Foundation']['Krx'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Abutment_Foundation']['Kry'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Abutment_Foundation']['Krz'][i])

                count5 += 1
                eleType = 'zeroLength'
                eleNodes = [nodeI, nodeJ]
                eleTag = int(str(count5) + self.SpringEleTag)
                eleArgs = ['-mat', *matTags, '-dir', *dirs, '-orient', *vecx, *vecyp]
                ops.element(eleType, eleTag, *eleNodes, *eleArgs)

        elif self.model['Abutment_Foundation']['Type'] == 'Group Pile':

            # Define integration schemes for pile elements
            IntTags, PileWs = self._def_pile_Int('Abutment')

            self.fixed_AB1Nodes_found = []
            self.fixed_AB2Nodes_found = []
            self.EleIDsPY_Abut = []  # PY spring element IDs
            self.EleIDsPile_Abut = []  # Pile element IDs
            self.EleLoadsPile_Abut = []  # Pile element Loads

            for i in range(2):
                count4 += 1

                # NODES
                top_node = int(str(count4) + self.Found4Tag)
                if self.model['Abutment_BackFill']['Type'] != 'None':
                    if i == 0:
                        TopCoords = np.array(ops.nodeCoord(self.AB1_Cnode))
                        top_node = self.AB1_Cnode

                    elif i == 1:
                        TopCoords = np.array(ops.nodeCoord(self.AB2_Cnode))
                        top_node = self.AB2_Cnode

                    TopCoords[2] = TopCoords[2] - self.model['Abutment_Foundation']['Height'][i]

                elif i == 0:
                    TopCoords = np.zeros(3)
                    for node in self.AB1Nodes:
                        TopCoords += np.array(ops.nodeCoord(node))
                        self.RigidLinkNodes.append([top_node, node])  # RIGID LINKS
                    TopCoords = TopCoords / len(self.AB1Nodes)
                    TopCoords[2] = TopCoords[2] - self.model['Abutment_Foundation']['Height'][i]
                    ops.node(top_node, *TopCoords.tolist())

                elif i == 1:
                    TopCoords = np.zeros(3)
                    for node in self.AB2Nodes:
                        TopCoords += np.array(ops.nodeCoord(node))
                        self.RigidLinkNodes.append([top_node, node])  # RIGID LINKS
                    TopCoords = TopCoords / len(self.AB2Nodes)
                    TopCoords[2] = TopCoords[2] - self.model['Abutment_Foundation']['Height'][i]
                    ops.node(top_node, *TopCoords.tolist())

                sheet = 'Abutment' + str(i + 1)
                data = pd.read_excel(open(self.model['Abutment_Foundation']['file'], 'rb'),
                                     sheet_name=sheet)
                idx = self.model['Abutment_Foundation']['Sections'][i] - 1
                Diameter = self.model['Abutment_Foundation']['D'][idx]
                nx = self.model['Abutment_Foundation']['nx'][i]
                ny = self.model['Abutment_Foundation']['ny'][i]
                sx = self.model['Abutment_Foundation']['sx'][i]
                sy = self.model['Abutment_Foundation']['sy'][i]
                if self.model['Abutment_Foundation']['Group Effect'] == 1:
                    fm1, fm2 = group_efficiency(nx, ny, sx, sy, Diameter)
                else:
                    fm1 = 1
                    fm2 = 1
                coords0 = np.array([TopCoords[0] - (nx - 1) * sx / 2,
                                    TopCoords[1] - (ny - 1) * sy / 2,
                                    TopCoords[2]])
                top_coords_list = []
                for k1 in range(nx):
                    for k2 in range(ny):
                        coords = np.array([coords0[0] + k1 * sx, coords0[1] + k2 * sy, coords0[2]])
                        top_coords_list.append(coords)

                for k in range(len(top_coords_list)):
                    pyDepth = 0
                    sigV = 0
                    coords = top_coords_list[k]
                    pile_nodes = []
                    fixed_nodes = []
                    springIDs = []
                    eleIDs = []

                    for j in range(len(data['Layer ID'])):
                        hlayer = data['Thickness'][j]
                        gamma = data['Gamma'][j]
                        phiDegree = data['Angle of Friction'][j]
                        soil = data['Soil'][j]
                        cu = data['Undrained Shear Strength'][j]
                        eps50 = data['Eps50'][j]
                        Gsoil = data['Gsoil'][j]
                        Cd = 0.3  # Variable that sets the drag resistance within a fully-mobilized gap as Cd*pult.
                        # (0.3 is a recommended value) Reference: Boulanger et al. 1999
                        pEleLength = hlayer
                        rho = gamma / g
                        Area = pEleLength * Diameter
                        Vs = (Gsoil / rho) ** 0.5
                        Vp = 1.87 * Vs
                        v = (Vp + Vs) / 2
                        # Radiation Damping
                        c = rho * Area * (Vs + v)  # The viscous damping term (dashpot) on the far-field
                        # (elastic) component of the displacement rate (velocity).
                        # (optional Default = 0.0). Nonzero c values are used
                        # to represent radiation damping effects
                        # Reference: Lysmer and Kuhlemeyer (1969)
                        # p.A.Vs is the simplest definition
                        # where A=Diameter*DeltaZ(or element length)
                        # Yet, ideally, this term is frequency dependent, Gazetas and Dobry 1984
                        # Let's make use of the approach by Berger, Mahin & Pyke 1977 for piles

                        if 'clay' in soil:
                            soilType = 1
                        else:
                            soilType = 2

                        if j == 0:  # pile node at free surface level
                            if all(coords != TopCoords):
                                count3 += 1
                                node_pile = int(str(count3) + self.Found3Tag)
                                ops.node(node_pile, *coords)
                            else:
                                node_pile = top_node
                            pile_nodes.append(node_pile)
                            self.RigidLinkNodes.append([top_node, node_pile])  # RIGID LINKS

                        # calculate spring properties based on mid-depth of the layer
                        coords[2] = coords[2] - hlayer / 2
                        pyDepth += hlayer / 2
                        sigV += gamma * hlayer / 2

                        count1 += 1
                        count3 += 1
                        node_fixed = int(str(count1) + self.Found1Tag)
                        node_pile = int(str(count3) + self.Found3Tag)

                        ops.node(node_fixed, *coords)
                        ops.node(node_pile, *coords)

                        pile_nodes.append(node_pile)
                        fixed_nodes.append(node_fixed)
                        if i == 0:
                            self.fixed_AB1Nodes.append(node_fixed)  # Fixed nodes at foundation
                        elif i == 1:
                            self.fixed_AB2Nodes.append(node_fixed)

                        # P-y springs
                        # TODO: gwtSwitch = 2 if the soil is below ground water
                        #  properties change sigV = sigV - 10*|PyDepth-gwtDepth|
                        if soilType == 1:
                            y50, pult, strain_stress = get_pyParam_clay(pyDepth, sigV, cu, eps50, Diameter, pEleLength,
                                                                        soil)
                        elif soilType == 2:
                            y50, pult, strain_stress = get_pyParam_sand(pyDepth, sigV, phiDegree, Diameter, pEleLength,
                                                                        LSwitch=1)
                        # Define P-Y Mats in dir 1 and 2
                        dirs = [1, 2]

                        if self.model['Abutment_Foundation']['py_Mat'] == 'PySimple1':
                            # Using PySimple1 Material
                            # Direction 1
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('PySimple1', self.EndMatTag, soilType, pult * fm1, y50, Cd, c)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags = [self.EndMatTag]

                            # Direction 2
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('PySimple1', self.EndMatTag, soilType, pult * fm2, y50, Cd, c)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags.append(self.EndMatTag)
                         
                        elif self.model['Abutment_Foundation']['py_Mat'] == 'NonGapping_MultiLinear':
                            # Using MultiLinear Material  
                            # Direction 1
                            self.EndMatTag += 1
                            strain_stress[np.arange(1, len(strain_stress), 2).tolist()] = \
                                strain_stress[np.arange(1, len(strain_stress), 2).tolist()] * fm1
                            ops.uniaxialMaterial('MultiLinear', self.EndMatTag, *strain_stress)
                            # Add small stiffness value to avoid zero stiffness
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags = [self.EndMatTag]

                            # Direction 2
                            self.EndMatTag += 1
                            strain_stress[np.arange(1, len(strain_stress), 2).tolist()] = \
                                strain_stress[np.arange(1, len(strain_stress), 2).tolist()] * fm2 / fm1
                            ops.uniaxialMaterial('MultiLinear', self.EndMatTag, *strain_stress)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags.append(self.EndMatTag)

                        if self.model['Abutment_Foundation']['tz_qz'] == 1:
                            # Consider vertical springs
                            # vertical effective stress at current depth
                            self.EndMatTag += 1
                            # I am not sure if this makes sense for clays, but ok
                            if j == len(data['Layer ID']) - 1:  # q-z spring
                                z50, qult = get_qzParam(phiDegree, Diameter, sigV, Gsoil)
                                ops.uniaxialMaterial('QzSimple1', self.EndMatTag, soilType, qult, z50)
                            else:  # t-z spring
                                z50, tult = get_tzParam(phiDegree, Diameter, sigV, pEleLength)
                                ops.uniaxialMaterial('TzSimple1', self.EndMatTag, soilType, tult, z50, 0.0)
                            matTags.append(self.EndMatTag)
                            dirs.append(3)

                        else:
                            # Restraint is assigned instead of vertical springs at the bottom
                            if j == len(data['Layer ID']) - 1:
                                ops.fix(node_pile, 0, 0, 1, 0, 0, 0)
                                fixed_nodes.append(node_pile)
                                if i == 0:
                                    self.fixed_AB1Nodes.append(node_pile) # Fixed nodes at foundation
                                elif i == 1:
                                    self.fixed_AB2Nodes.append(node_pile)

                        count5 += 1
                        eletag = int(str(count5) + self.SpringEleTag)
                        springIDs.append(eletag)
                        ops.element('zeroLength', eletag, node_fixed, node_pile, '-mat', *matTags, '-dir', *dirs)
                        ops.fix(node_fixed, 1, 1, 1, 1, 1, 1)

                        # update the nodal coordinate
                        coords[2] = coords[2] - hlayer / 2
                        pyDepth += hlayer / 2
                        sigV += gamma * hlayer / 2

                    IntTag = IntTags[i]
                    wTOT = PileWs[i]
                    if self.model['Abutment_Foundation']['EleType'] in [0, 1]:
                        eleType = 'dispBeamColumn'
                    else:
                        eleType = 'forceBeamColumn'

                    # create elements
                    if i == 0:
                        skew = self.skew[0]
                    elif i == 1:
                        skew = self.skew[-1]
                    # Define geometric transformation of the bent
                    self.EndTransfTag += 1  # local z is in global x
                    ops.geomTransf('Linear', self.EndTransfTag, np.cos(skew), np.sin(skew), 0)
                    for j in range(len(pile_nodes) - 1):
                        count6 += 1
                        eleTag = int(str(count6) + self.PileEleTag)
                        nodeJ = pile_nodes[j]
                        nodeI = pile_nodes[j + 1]
                        ops.element(eleType, eleTag, nodeI, nodeJ, self.EndTransfTag, IntTag, '-mass',
                                    wTOT / g, self.mass_type)
                        eleIDs.append(eleTag)

                    self.EleIDsPY_Abut.append(springIDs)  # PY spring element IDs
                    self.EleIDsPile_Abut.append(eleIDs)  # Pile element IDs
                    self.EleLoadsPile_Abut.append(wTOT)  # Pile element Loads
                    if i == 0:
                        self.fixed_AB1Nodes_found.append(fixed_nodes)  # Fixed nodes at foundation
                    elif i == 1:
                        self.fixed_AB2Nodes_found.append(fixed_nodes)

        if self.model['Bent_Foundation']['Type'] == 'Fixed':
            for node_list in self.BentStartNodes:
                for node in node_list:
                    ops.fix(node, 1, 1, 1, 1, 1, 1)

            self.fixed_BentNodes = self.BentStartNodes.copy()

        elif self.model['Bent_Foundation']['Type'] == 'Springs':

            dirs = [1, 2, 3, 4, 5, 6]
            self.fixed_BentNodes = []

            for i in range(self.num_bents):
                matTags = []
                # Define skew angle of the bent
                skew = (self.skew[i] + self.skew[i + 1]) / 2
                vecx = [np.cos(skew), np.sin(skew), 0]
                vecyp = [np.cos(skew + np.pi / 2), np.sin(skew + np.pi / 2), 0]

                # NODES
                count1 += 1
                count2 += 1
                nodeI = int(str(count1) + self.Found1Tag)
                nodeJ = int(str(count2) + self.Found2Tag)
                Coords = np.zeros(3)
                for node in self.BentStartNodes[i]:
                    Coords += np.array(ops.nodeCoord(node))
                    self.RigidLinkNodes.append([nodeJ, node])  # RIGID LINKS
                Coords = Coords / self.model['Bent']['N']
                Coords[2] = Coords[2] - self.model['Bent_Foundation']['Thickness']
                ops.node(nodeI, *Coords.tolist())
                ops.node(nodeJ, *Coords.tolist())
                ops.fix(nodeI, 1, 1, 1, 1, 1, 1)
                self.fixed_BentNodes.append(nodeI)

                # MATERIALS
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bent_Foundation']['Kx'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bent_Foundation']['Ky'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bent_Foundation']['Kz'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bent_Foundation']['Krx'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bent_Foundation']['Kry'][i])
                self.EndMatTag += 1
                matTags.append(self.EndMatTag)
                ops.uniaxialMaterial('Elastic', self.EndMatTag, self.model['Bent_Foundation']['Krz'][i])

                count5 += 1
                eleType = 'zeroLength'
                eleNodes = [nodeI, nodeJ]
                eleTag = int(str(count5) + self.SpringEleTag)
                eleArgs = ['-mat', *matTags, '-dir', *dirs, '-orient', *vecx, *vecyp]
                ops.element(eleType, eleTag, *eleNodes, *eleArgs)

        elif self.model['Bent_Foundation']['Type'] == 'Pile-Shaft':
            self.pile_nodes = []
            self.embedded_nodes = []
            self.fixed_BentNodes = []
            self.EleIDsPY = []  # PY spring element IDs  
            self.EleIDsPile = []  # Pile element IDs      
            self.EleLoadsPile = []  # Pile element Loads

            IntTags, PileWs = self._def_pile_Int('Bent')

            for i in range(self.num_bents):
                fixed_nodes_ = []
                Diameter = self.model['Bent_Foundation']['D'][i]
                sheet = 'Bent' + str(i + 1)
                data = pd.read_excel(open(self.model['Bent_Foundation']['file'], 'rb'),
                                     sheet_name=sheet)
                for k in range(self.model['Bent']['N']):
                    pile_nodes = []
                    fixed_nodes = []
                    springIDs = []
                    eleIDs = []
                    pyDepth = 0
                    sigV = 0
                    TopNode = self.BentStartNodes[i][k]
                    coords = np.array(ops.nodeCoord(TopNode))

                    for j in range(len(data['Layer ID'])):
                        hlayer = data['Thickness'][j]
                        gamma = data['Gamma'][j]
                        phiDegree = data['Angle of Friction'][j]
                        soil = data['Soil'][j]
                        cu = data['Undrained Shear Strength'][j]
                        eps50 = data['Eps50'][j]
                        Gsoil = data['Gsoil'][j]
                        Cd = 0.3  # Variable that sets the drag resistance within a fully-mobilized gap as Cd*pult.
                        # (0.3 is a recommended value) Reference: Boulanger et al. 1999
                        pEleLength = hlayer
                        rho = gamma / g
                        Area = pEleLength * Diameter
                        Vs = (Gsoil / rho) ** 0.5
                        Vp = 1.87 * Vs
                        v = (Vp + Vs) / 2
                        c = rho * Area * (Vs + v)  # The viscous damping term (dashpot) on the far-field
                        # (elastic) component of the displacement rate (velocity).
                        # (optional Default = 0.0). Nonzero c values are used
                        # to represent radiation damping effects
                        # Reference: Lysmer and Kuhlemeyer (1969)
                        # p.A.Vs is the simplest definition
                        # where A=Diameter*DeltaZ(or element length)
                        # Yet, ideally, this term is frequency dependent, Gazetas and Dobry 1984
                        # Let's make use of the approach by Berger, Mahin & Pyke 1977 for piles

                        if 'clay' in soil:
                            soilType = 1
                        else:
                            soilType = 2

                        if j == 0:  # pile node at free surface level
                            count3 += 1
                            node_pile = int(str(count3) + self.Found3Tag)
                            ops.node(node_pile, *coords.tolist())
                            pile_nodes.append(node_pile)
                            self.RigidLinkNodes.append([node_pile, self.BentStartNodes[i][k]])  # RIGID LINKS

                        # calculate spring properties based on mid-depth of the layer
                        coords[2] = coords[2] - hlayer / 2
                        pyDepth += hlayer / 2
                        sigV += gamma * hlayer / 2

                        count1 += 1
                        count3 += 1
                        node_fixed = int(str(count1) + self.Found1Tag)
                        node_pile = int(str(count3) + self.Found3Tag)

                        ops.node(node_fixed, *coords.tolist())
                        ops.node(node_pile, *coords.tolist())

                        pile_nodes.append(node_pile)
                        fixed_nodes.append(node_fixed)

                        # P-y springs
                        # TODO: gwtSwitch = 2 if the soil is below ground water
                        #  properties change sigV = sigV - 10*|PyDepth-gwtDepth|
                        if soilType == 1:
                            y50, pult, strain_stress = get_pyParam_clay(pyDepth, sigV, cu, eps50, Diameter, pEleLength,
                                                                        soil)
                        elif soilType == 2:
                            y50, pult, strain_stress = get_pyParam_sand(pyDepth, sigV, phiDegree, Diameter, pEleLength,
                                                                        LSwitch=1)
                        # Define P-Y Mats in dir 1 and 2
                        dirs = [1, 2]

                        if self.model['Bent_Foundation']['py_Mat'] == 'PySimple1':
                            # Using PySimple1 Material
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('PySimple1', self.EndMatTag, soilType, pult, y50, Cd, c)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags = [self.EndMatTag, self.EndMatTag]
                         
                        elif self.model['Bent_Foundation']['py_Mat'] == 'NonGapping_MultiLinear':
                            # Using MultiLinear Material  
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('MultiLinear', self.EndMatTag, *strain_stress)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags = [self.EndMatTag, self.EndMatTag]

                        if self.model['Bent_Foundation']['tz_qz'] == 1:
                            # Consider vertical springs
                            # vertical effective stress at current depth
                            self.EndMatTag += 1
                            # I am not sure if this makes sense for clays, but ok
                            if j == len(data['Layer ID']) - 1:  # q-z spring
                                z50, qult = get_qzParam(phiDegree, Diameter, sigV, Gsoil)
                                ops.uniaxialMaterial('QzSimple1', self.EndMatTag, soilType, qult, z50)
                            else:  # t-z spring
                                z50, tult = get_tzParam(phiDegree, Diameter, sigV, pEleLength)
                                ops.uniaxialMaterial('TzSimple1', self.EndMatTag, soilType, tult, z50, 0.0)
                            matTags.append(self.EndMatTag)
                            dirs.append(3)

                        else:
                            # Restraint is assigned instead of vertical springs at the bottom
                            if j == len(data['Layer ID']) - 1:
                                ops.fix(node_pile, 0, 0, 1, 0, 0, 0)
                                fixed_nodes.append(node_pile)

                        count5 += 1
                        eletag = int(str(count5) + self.SpringEleTag)
                        springIDs.append(eletag)
                        ops.element('zeroLength', eletag, node_fixed, node_pile, '-mat', *matTags, '-dir', *dirs)
                        ops.fix(node_fixed, 1, 1, 1, 1, 1, 1)

                        # update the nodal coordinate
                        coords[2] = coords[2] - hlayer / 2
                        pyDepth += hlayer / 2
                        sigV += gamma * hlayer / 2

                    IntTag = IntTags[i]
                    wTOT = PileWs[i]
                    if self.model['Bent_Foundation']['EleType'] in [0, 1]:
                        eleType = 'dispBeamColumn'
                    else:
                        eleType = 'forceBeamColumn'

                    # create elements                
                    skew = (self.skew[i] + self.skew[i + 1]) / 2
                    # Define geometric transformation of the bent
                    self.EndTransfTag += 1  # local z is in global x
                    ops.geomTransf('Linear', self.EndTransfTag, np.cos(skew), np.sin(skew), 0)
                    for j in range(len(pile_nodes) - 1):
                        count6 += 1
                        eleTag = int(str(count6) + self.PileEleTag)
                        nodeJ = pile_nodes[j]
                        nodeI = pile_nodes[j + 1]
                        ops.element(eleType, eleTag, nodeI, nodeJ, self.EndTransfTag, IntTag, '-mass',
                                    wTOT / g, self.mass_type)
                        eleIDs.append(eleTag)

                    self.EleIDsPY.append(springIDs)  # PY spring element IDs
                    self.EleIDsPile.append(eleIDs)  # Pile element IDs      
                    self.EleLoadsPile.append(wTOT)  # Pile element Loads
                    fixed_nodes_.append(fixed_nodes)
                self.fixed_BentNodes.append(fixed_nodes_)  # Fixed nodes at foundation

        elif self.model['Bent_Foundation']['Type'] == 'Group Pile':

            # Note: There must be a single pier for each bent in this case!
            cap_mass = self.model['Bent_Foundation']['cap_t'] * self.model['Bent_Foundation']['cap_A'] * gamma_c / g
            self.PcapWeight = self.model['Bent_Foundation']['cap_t'] * self.model['Bent_Foundation']['cap_A'] * gamma_c

            self.pile_nodes = []
            self.fixed_BentNodes = []
            self.EleIDsPY = []  # PY spring element IDs  
            self.EleIDsPile = []  # Pile element IDs      
            self.EleLoadsPile = []  # Pile element Loads
            self.PcapNodes = []  # Pile cap nodes

            # Define integration schemes for pile elements
            IntTags, PileWs = self._def_pile_Int('Bent')

            for i in range(self.num_bents):
                cap_coords = np.zeros(3)
                for k in range(self.model['Bent']['N']):
                    bentNode = self.BentStartNodes[i][k]
                    cap_coords += np.array(ops.nodeCoord(bentNode))

                cap_coords = cap_coords / self.model['Bent']['N']
                cap_coords[2] = cap_coords[2] - self.model['Bent_Foundation']['cap_t'] / 2
                count4 += 1
                CapNode = int(str(count4) + self.Found4Tag)
                ops.node(CapNode, *cap_coords.tolist(), cap_mass, cap_mass, cap_mass, 0, 0, 0)
                self.PcapNodes.append(CapNode)

                for k in range(self.model['Bent']['N']):
                    bentNode = self.BentStartNodes[i][k]
                    self.RigidLinkNodes.append([CapNode, bentNode])

                fixed_nodes_ = []
                sheet = 'Bent' + str(i + 1)
                data = pd.read_excel(open(self.model['Bent_Foundation']['file'], 'rb'),
                                     sheet_name=sheet)
                idx = self.model['Bent_Foundation']['Sections'][i] - 1
                Diameter = self.model['Bent_Foundation']['D'][idx]
                nx = self.model['Bent_Foundation']['nx'][i]
                ny = self.model['Bent_Foundation']['ny'][i]
                sx = self.model['Bent_Foundation']['sx'][i]
                sy = self.model['Bent_Foundation']['sy'][i]
                if self.model['Bent_Foundation']['Group Effect'] == 1:
                    fm1, fm2 = group_efficiency(nx, ny, sx, sy, Diameter)
                else:
                    fm1 = 1
                    fm2 = 1

                coords0 = np.array([cap_coords[0] - (nx - 1) * sx / 2, cap_coords[1] - (ny - 1) * sy / 2,
                                    cap_coords[2] - self.model['Bent_Foundation']['cap_t'] / 2])
                top_coords_list = []
                for k1 in range(nx):
                    for k2 in range(ny):
                        coords = np.array([coords0[0] + k1 * sx, coords0[1] + k2 * sy, coords0[2]])
                        top_coords_list.append(coords.tolist())

                for k in range(len(top_coords_list)):
                    pyDepth = 0
                    sigV = 0
                    coords = top_coords_list[k]
                    pile_nodes = []
                    fixed_nodes = []
                    springIDs = []
                    eleIDs = []

                    for j in range(len(data['Layer ID'])):
                        hlayer = data['Thickness'][j]
                        gamma = data['Gamma'][j]
                        phiDegree = data['Angle of Friction'][j]
                        soil = data['Soil'][j]
                        cu = data['Undrained Shear Strength'][j]
                        eps50 = data['Eps50'][j]
                        Gsoil = data['Gsoil'][j]
                        Cd = 0.3  # Variable that sets the drag resistance within a fully-mobilized gap as Cd*pult.
                        # (0.3 is a recommended value) Reference: Boulanger et al. 1999
                        pEleLength = hlayer
                        rho = gamma / g
                        Area = pEleLength * Diameter
                        Vs = (Gsoil / rho) ** 0.5
                        Vp = 1.87 * Vs
                        v = (Vp + Vs) / 2
                        c = rho * Area * (Vs + v)  # The viscous damping term (dashpot) on the far-field
                        # (elastic) component of the displacement rate (velocity).
                        # (optional Default = 0.0). Nonzero c values are used
                        # to represent radiation damping effects
                        # Reference: Lysmer and Kuhlemeyer (1969)
                        # p.A.Vs is the simplest definition
                        # where A=Diameter*DeltaZ(or element length)
                        # Yet, ideally, this term is frequency dependent, Gazetas and Dobry 1984
                        # Let's make use of the approach by Berger, Mahin & Pyke 1977 for piles

                        if 'clay' in soil:
                            soilType = 1
                        else:
                            soilType = 2

                        if j == 0:  # pile node at free surface level
                            count3 += 1
                            node_pile = int(str(count3) + self.Found3Tag)
                            ops.node(node_pile, *coords)
                            pile_nodes.append(node_pile)
                            self.RigidLinkNodes.append([CapNode, node_pile])  # RIGID LINKS

                        # calculate spring properties based on mid-depth of the layer
                        count1 += 1
                        count3 += 1
                        coords[2] = coords[2] - hlayer / 2
                        pyDepth += hlayer / 2
                        sigV += gamma * hlayer / 2

                        node_fixed = int(str(count1) + self.Found1Tag)
                        node_pile = int(str(count3) + self.Found3Tag)

                        ops.node(node_fixed, *coords)
                        ops.node(node_pile, *coords)

                        pile_nodes.append(node_pile)
                        fixed_nodes.append(node_fixed)

                        # P-y springs
                        # TODO: gwtSwitch = 2 if the soil is below ground water
                        #  properties change sigV = sigV - 10*|PyDepth-gwtDepth|
                        if soilType == 1:
                            y50, pult, strain_stress = get_pyParam_clay(pyDepth, sigV, cu, eps50, Diameter, pEleLength,
                                                                        soil)
                        elif soilType == 2:
                            y50, pult, strain_stress = get_pyParam_sand(pyDepth, sigV, phiDegree, Diameter, pEleLength,
                                                                        LSwitch=1)

                        # Define P-Y Mats in dir 1 and 2
                        dirs = [1, 2]

                        if self.model['Bent_Foundation']['py_Mat'] == 'PySimple1':
                            # Using PySimple1 Material
                            # Direction 1
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('PySimple1', self.EndMatTag, soilType, pult * fm1, y50, Cd, c)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags = [self.EndMatTag]

                            # Direction 2
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('PySimple1', self.EndMatTag, soilType, pult * fm2, y50, Cd, c)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags.append(self.EndMatTag)
                         
                        elif self.model['Bent_Foundation']['py_Mat'] == 'NonGapping_MultiLinear':
                            # Using MultiLinear Material  
                            # Direction 1
                            self.EndMatTag += 1
                            strain_stress[np.arange(1, len(strain_stress), 2).tolist()] = \
                                strain_stress[np.arange(1, len(strain_stress), 2).tolist()] * fm1
                            ops.uniaxialMaterial('MultiLinear', self.EndMatTag, *strain_stress)
                            # Add small stiffness value to avoid zero stiffness
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags = [self.EndMatTag]

                            # Direction 2
                            self.EndMatTag += 1
                            strain_stress[np.arange(1, len(strain_stress), 2).tolist()] = \
                                strain_stress[np.arange(1, len(strain_stress), 2).tolist()] * fm2 / fm1
                            ops.uniaxialMaterial('MultiLinear', self.EndMatTag, *strain_stress)
                            # To avoid zero stiffness values
                            self.EndMatTag += 1
                            ops.uniaxialMaterial('Parallel', self.EndMatTag, self.EndMatTag - 1, self.SmallMat)
                            matTags.append(self.EndMatTag)

                        if self.model['Bent_Foundation']['tz_qz'] == 1:
                            # Consider vertical springs
                            # vertical effective stress at current depth
                            self.EndMatTag += 1
                            # I am not sure if this makes sense for clays, but ok
                            if j == len(data['Layer ID']) - 1:  # q-z spring
                                z50, qult = get_qzParam(phiDegree, Diameter, sigV, Gsoil)
                                ops.uniaxialMaterial('QzSimple1', self.EndMatTag, soilType, qult, z50)
                            else:  # t-z spring
                                z50, tult = get_tzParam(phiDegree, Diameter, sigV, pEleLength)
                                ops.uniaxialMaterial('TzSimple1', self.EndMatTag, soilType, tult, z50, 0.0)
                            matTags.append(self.EndMatTag)
                            dirs.append(3)

                        else:
                            # Restraint is assigned instead of vertical springs at the bottom
                            if j == len(data['Layer ID']) - 1:
                                ops.fix(node_pile, 0, 0, 1, 0, 0, 0)
                                fixed_nodes.append(node_pile)

                        count5 += 1
                        eletag = int(str(count5) + self.SpringEleTag)
                        springIDs.append(eletag)
                        ops.element('zeroLength', eletag, node_fixed, node_pile, '-mat', *matTags, '-dir', *dirs)
                        ops.fix(node_fixed, 1, 1, 1, 1, 1, 1)

                        # update the nodal coordinate
                        coords[2] = coords[2] - hlayer / 2
                        pyDepth += hlayer / 2
                        sigV += gamma * hlayer / 2

                    IntTag = IntTags[i]
                    wTOT = PileWs[i]
                    if self.model['Bent_Foundation']['EleType'] in [0, 1]:
                        eleType = 'dispBeamColumn'
                    else:
                        eleType = 'forceBeamColumn'

                    # create elements
                    skew = (self.skew[i] + self.skew[i + 1]) / 2
                    # Define geometric transformation of the bent
                    self.EndTransfTag += 1  # local z is in global x
                    ops.geomTransf('Linear', self.EndTransfTag, np.cos(skew), np.sin(skew), 0)
                    for j in range(len(pile_nodes) - 1):
                        count6 += 1
                        eleTag = int(str(count6) + self.PileEleTag)
                        nodeJ = pile_nodes[j]
                        nodeI = pile_nodes[j + 1]
                        ops.element(eleType, eleTag, nodeI, nodeJ, self.EndTransfTag, IntTag, '-mass',
                                    wTOT / g, self.mass_type)
                        eleIDs.append(eleTag)

                    self.EleIDsPY.append(springIDs)  # PY spring element IDs
                    self.EleIDsPile.append(eleIDs)  # Pile element IDs      
                    self.EleLoadsPile.append(wTOT)  # Pile element Loads
                    fixed_nodes_.append(fixed_nodes)
                self.fixed_BentNodes.append(fixed_nodes_)  # Fixed nodes at foundation


    def _constraints(self):
        matTags = [self.BigMat, self.BigMat, self.BigMat, self.BigMat, self.BigMat, self.BigMat]
        dirs = [1, 2, 3, 4, 5, 6]
        RigidCount = 1

        # Use beam column elements with very high stiffness
        if self.const_opt == 1:
            self.EndTransfTag += 1
            ops.geomTransf('Linear', self.EndTransfTag, 1, 0, 0)
            for i in range(len(self.RigidLinkNodes)):
                eleTag = int(str(RigidCount) + self.RigidTag)
                RigidCount += 1
                eleNodes = self.RigidLinkNodes[i]
                coords1 = np.array(ops.nodeCoord(eleNodes[0]))
                coords2 = np.array(ops.nodeCoord(eleNodes[1]))
                if (coords1 == coords2).all():
                    ops.element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs)
                else:
                    ops.element('dispBeamColumn', eleTag, *eleNodes, self.EndTransfTag, self.BigInt)

        # Use rigid link constraints                    
        else:
            for i in range(len(self.RigidLinkNodes)):
                eleNodes = self.RigidLinkNodes[i]
                coords1 = np.array(ops.nodeCoord(eleNodes[0]))
                coords2 = np.array(ops.nodeCoord(eleNodes[1]))
                if (coords1 == coords2).all():
                    eleTag = int(str(RigidCount) + self.RigidTag)
                    RigidCount += 1
                    ops.element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs)
                else:
                    ops.rigidLink('beam', *self.RigidLinkNodes[i])
    
    def _def_bent_Int(self):
        # Define sections
        if self.model['Bent']['SectionTypes'] == 'Circ':
            SecTags, SecWs = self._def_BentCirc_Sec()

        # Define integrations for bents
        IntTags = []
        BentWs = []
        for i in range(self.num_bents):
            idx = self.model['Bent']['Sections'][i] - 1
            self.EndIntTag += 1

            # Legendre
            if self.model['Bent']['EleType'] == 0:
                ops.beamIntegration('Legendre', self.EndIntTag, SecTags[i], 2)

                # Lobatto with 5 integration points
            elif self.model['Bent']['EleType'] == 4:
                ops.beamIntegration('Lobatto', self.EndIntTag, SecTags[i], 5)

            else:  # HingeRadau based on predetermined plastic hinge length
                Hcol = self.model['Bent']['H'][i]
                Fyle = self.model['Bent']['Fyle'][idx]
                dl = self.model['Bent']['dl'][idx]
                Lpl = (0.08 * Hcol / mm + 0.022 * dl * Fyle) * mm
                # Lpl = (0.08 * Hcol/2 / mm + 0.044 * dl * Fyle) * mm # for double bending case, more suitable.

                # Lumped plasticity with elastic interior
                if self.model['Bent']['EleType'] in [1, 2]:  # Elastic interior
                    sec1 = SecTags[i][0]  # inelastic section
                    sec2 = SecTags[i][1]  # elastic section
                    ops.beamIntegration('HingeRadau', self.EndIntTag, sec1, Lpl, sec1, Lpl, sec2)
                    # Distributed plasticity with adjusted integration weights
                elif self.model['Bent']['EleType'] == 3:  # Inelastic interior
                    ops.beamIntegration('HingeRadau', self.EndIntTag, SecTags[idx], Lpl, SecTags[i], Lpl, SecTags[i])

            IntTags.append(self.EndIntTag)
            BentWs.append(SecWs[i])

        return IntTags, BentWs

    def _def_pile_Int(self, case):
        # Define Integration Schemes
        SecTags, SecWs = self._def_PileCirc_Sec(case)
        # Define integrations for bents
        IntTags = []
        PileWs = []

        if case == 'Bent':
            for i in range(self.num_bents):
                self.EndIntTag += 1

                # Gauss Legendre with 2 integration points
                if self.model['Bent_Foundation']['EleType'] in [1, 0]:
                    ops.beamIntegration('Legendre', self.EndIntTag, SecTags[i], 2)

                # Gauss Lobatto with 3 integration points
                elif self.model['Bent_Foundation']['EleType'] == 2:
                    ops.beamIntegration('Lobatto', self.EndIntTag, SecTags[i], 3)

                IntTags.append(self.EndIntTag)
                PileWs.append(SecWs[i])

        elif case == 'Abutment':
            for i in range(2):
                self.EndIntTag += 1

                # Gauss Legendre with 2 integration points
                if self.model['Abutment_Foundation']['EleType'] in [1, 0]:
                    ops.beamIntegration('Legendre', self.EndIntTag, SecTags[i], 2)

                # Gauss Lobatto with 3 integration points
                elif self.model['Abutment_Foundation']['EleType'] == 2:
                    ops.beamIntegration('Lobatto', self.EndIntTag, SecTags[i], 3)

                IntTags.append(self.EndIntTag)
                PileWs.append(SecWs[i])

        return IntTags, PileWs

def DiscretizeMember(ndI, ndJ, numEle, eleType, integrTag, transfTag, nodeTag, eleTag, Mass, MType):
    """
    ------------------------------------
    DISCRETIZATION OF DECK MEMBERS
    ------------------------------------
    """
    nodeList = []
    eleList = []
    if numEle <= 1:
        ops.element(eleType, eleTag, ndI, ndJ, transfTag, integrTag, '-mass', Mass, MType)
        eleList.append(eleTag)
        nodeList.append(ndI)      
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
        nodeTag = nodeTag + 100
    ops.element(eleType, eleTag, ndI, nodes[1], transfTag, integrTag, '-mass', Mass, MType)
    eleList.append(eleTag)
    eleTag = eleTag + 100

    for i in range(1, numEle - 1):
        ops.element(eleType, eleTag, nodes[i], nodes[i + 1], transfTag, integrTag, '-mass', Mass, MType)
        eleList.append(eleTag)
        eleTag = eleTag + 100

    ops.element(eleType, eleTag, nodes[numEle - 1], ndJ, transfTag, integrTag, '-mass', Mass, MType)
    eleList.append(eleTag)

    return eleList, nodeList


def group_efficiency(nx, ny, sx, sy, D):
    """
    Source, FEMA P-1051 Design Examples:
        Section: 7.2.1.4.4 Group Effect Factors

    ref:
        Rollins et al., “Pile Spacing Effects on Lateral Pile Group Behavior: Analysis,”
        Journal of Geotechnical and Geoenvironmental Engineering, October 2006.
    Parameters
    ----------
    nx : int
        number of piles in x drection.
    ny : int
        number of piles in y direction.
    sx : float
        pile spacing in x direction.
    sy : float
        pile spacing in y direction.
    D : float
        single pile diameter.

    Returns
    -------
    fmx : float
        group effect factor for loading in x-direction.
    fmy : float
        group effect factor for loading in y-direction.

    """
    # account for shadowing effects due to pile-soil-pile interaction
    fmx = []
    fmy = []
    for i in range(nx):
        if i == 0:
            fm = min(0.26 * np.log(sx / D) + 0.5, 1)
        elif i == 1:
            fm = min(0.52 * np.log(sx / D), 1)
        else:
            fm = min(0.52 * np.log(sx / D) - 0.25, 1)
        fmx.append(fm)
    for i in range(ny):
        if i == 0:
            fm = min(0.26 * np.log(sy / D) + 0.5, 1)
        elif i == 1:
            fm = min(0.52 * np.log(sy / D), 1)
        else:
            fm = min(0.52 * np.log(sy / D) - 0.25, 1)
        fmy.append(fm)

    fmx = np.mean(fmx)
    fmy = np.mean(fmy)

    return fmx, fmy



            
