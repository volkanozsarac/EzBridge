import openseespy.opensees as ops
from .Utility import def_units, create_outdir
import numpy as np
import os
import pandas as pd

class PierInfo():

    def __init__(self):
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = def_units(0)
    
    def _get_CircPier_info(self):
        # These are extra stuff that can be saved to calculate ductility related EDPs for piers.
        # Note that in case of rectangular sections, curvature must be  calculated separately for two local axis
        # Therefore calculation of displacement ductility becomes rather difficult if you have
        # sections which are not circular, and rotating.
        # For now I am assuming the piers act as cantilever in any direction, which is true 
        # in case there is a single pier per bent
        # However, this must be changed if more than one piers are used per bent
        # In such case pier can be assumed to have double bending in transverse direction
        
        # at section level
        # 1 refers to dof: Mz  y
        # 2 refers to dof: My  |_z <-- Top view
        # at element level
        # 1 refers to dof: x   y
        # 2 refers to dof: y   |_x  <-- Top view
        
        Dir1 = 'single bending'
        
        if len(self.EleIDsBent[0]) > 1:
            Dir2 = 'double bending'
        else:
            Dir2 = 'single bending'
        
        if Dir1 == 'single bending':
            C1_1 = 1/3     # coeff for yield displacement calculations
            C2_1 = 0.022   # coeff for strain penetration length 
        elif Dir1 == 'double bending':
            C1_1 = 1/6     # coeff for yield displacement calculations
            C2_1 = 0.044   # coeff for strain penetration length 
        
        if Dir2 == 'single bending':
            C1_2 = 1/3     # coeff for yield displacement calculations
            C2_2 = 0.022   # coeff for strain penetration length 
        elif Dir2 == 'double bending':
            C1_2 = 1/6     # coeff for yield displacement calculations
            C2_2 = 0.044   # coeff for strain penetration length 
        
        Esteel = 200*GPa
        EleKy1 = []       # Yield curvature for different pier element
        EleKy2 = []       # Yield curvature for different pier element
        Eledispy1 = []    # Yield displacement for different pier elements
        Eledispy2 = []    # Yield displacement for different pier elements
        EleH = []
        
        self.BentKy1 = []       # Yield curvature in dir 1
        self.BentKy2 = []       # Yield curvature in dir 2
        self.BentDispy1 = []    # Yield displacement in dir 1
        self.BentDispy2 = []    # Yield displacement in dir 2
        self.BentHeight = []    # Height of bent elements       
        
        for i in range(self.num_bents):
            idx = self.model['Bent']['Sections'][i] - 1
            D = self.model['Bent']['D'][idx]  # Section diameter
            Fyle = self.model['Bent']['Fyle'][idx]
            dl = self.model['Bent']['dl'][idx]
            H = self.model['Bent']['H'][i]
            for j in range(len(self.EleIDsBent[i])):
                EleH.append(H)
                
                Ky = 2.25*Fyle/(Esteel)/D # yield curvature
                EleKy1.append(Ky) 
                EleKy2.append(Ky) 
                
                # Strain penetration lengths in both directions
                Lsp1 = C2_1*Fyle/MPa*dl
                Lsp2 = C2_2*Fyle/MPa*dl
                
                # Yield displacements in both directions
                Eledispy1.append(C1_1*Ky*(H+Lsp1)**2)
                Eledispy2.append(C1_2*Ky*(H+Lsp2)**2)

            self.BentKy1.append(EleKy1)
            self.BentKy2.append(EleKy2)
            self.BentDispy1.append(Eledispy1)
            self.BentDispy2.append(Eledispy2)
            self.BentHeight.append(EleH)
        
class BridgeSummary:
    
    def __init__(self):
        
        # ESTIMATION OF CONSTRUCTION, REPLACEMENT AND REPAIR COSTS FOR THE BRIDGE
        # Note these calculations are not exact but they are approximations. 
        # There might be missing info thus we make some assumptions based on what is available in Main.py.
        # In case of a specific case study bridge more accurate info could be available.
        # TODO -1 expenses for the bearings and expansion joints are missing for now
        
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi
        global SATDA, ASLB, AST, SRslab, SRbeams, SRPbeams, ARR, AAT, AAH, ABWT, AWWT
        global AWWL, ASWT, AFW, AFT, SRabut, gammas, SRcapbeam, SRfooting, Vs30
        global OPShm, OPAhm, OPPhm, OPCPhm, OPFhm, OPPIhm, OPADLS1hm
        global OPADLS2hm, OPADLS3hm, OPPDLS1hm, OPPDLS2hm, OPPDLS3hm
        # Define units
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = def_units(0)
        
        # MATERIAL QUANTITIES
        # Superstructure Related
        SATDA=0.5 #Average ratio for the slab area to total deck area [--]
        ASLB=3.0 #Average Spacing of longitudinal beams [m]
        AST=0.2 #Average slab thickness in [m]
        SRslab=150 #Average steel to concrete ratio for the slab [kg/m3]
        SRbeams=100 #Average steel to concrete ratio for the longitudinal beams [kg/m3]
        SRPbeams=17 #Average steel ratio of pre-stressing steel to deck area [kg/m2]
        ARR=100 #Average railing Weight per unit of length [kg/m]
        AAT=0.05 #Average asphalt thickness [m]
        
        #Abutment Related
        #Assumption: Seat-Type with shallow foundation
        AAH=5.0 #Assumed average abutment height [m]
        ABWT=0.30 #Assumed abutment back-wall thickness [m]
        AWWT=0.30 #Assumed abutment wing-wall thickness [m]
        AWWL=4.00 #Assumed abutment wing-wall length [m]
        ASWT=1.50 #Assumed abutment seat-wall thickness [m]
        AFW=2.50 #Assumed abutment foundation footing width [m]
        AFT=1.50 #Assumed abutment foundation fotting thickness [m]
        SRabut=90 #Average steel to concrete ratio for abutments [kg/m3]
        
        #Pier Related
        # dbt=12 #Assumed diameter of the transverse reinforcement [mm]
        gammas=7800 #Average unitary steel Weight [kg/m3]
        
        #Cap-Beam related
        # CBH=2.00 #Assumed average height of the Cap-Beam [m], trapezoidal case, general
        SRcapbeam=140 #Average steel to concrete ratio in cap beams [kg/m3]
        
        #Foundation Related
        #Estimation of founation quantities will be performed based on several
        #assumptions based on the expected type of soil as described by the Vs30.
        #Namely:
        #If Vs30>=760m/s, the foundation is considered composed with 4 D=1.50m
        #L=10m piles with 6.50x6.50x2.50 footing
        #If 200m/s<=Vs30<760m/s, the foundation is considered composed with 9
        #D=1.00m L=15m piles with 7.50x7.50x2.00 footing
        #If Vs30<200m/s the foundation is considered composed with 16 D=0.80m L=20m
        #piles with 8.50x8.50x1.50 footing.
        #otherwise if exact info available calculate it (Manually add that part)
        SRfooting = 120 #Average steel to concrete ratio in footings [kg/m3]
        Vs30 = 650 #Shear Wave velocity for the first 30m [m/s]
        
        # UNITARY CONSTRUCTION COSTS
        # Related in the corresponding functions
        OPShm=1.8 #Overprice for handwork and machinery in the superstructure
        OPAhm=1.5 #Overprice for handwork and machinery in the abutment
        OPPhm=1.6 #Overprice for handwork and machinery in the piers
        OPCPhm=1.6 #Overprice for handwork and machinery in the cap-beams
        OPFhm=1.5 #Overprice for handwork and machinery in the footings
        OPPIhm=1.8 #Overprice for handwork and machinery in the piles
        
        # REPAIR ACTIONS UNITARY COSTS
        #Related in their respective functions and/or computation lines
        OPADLS1hm=1.1 #Overprice for handworks and machinery for repair actions of DLS-1 in abutments
        OPADLS2hm=1.2 #Overprice for handworks and machinery for repair actions of DLS-2 in abutments
        OPADLS3hm=1.4 #Overprice for handworks and machinery for repair actions of DLS-3 in abutments
        
        OPPDLS1hm=1.15 #Overprice for handworks and machinery for repair actions of DLS-1 in columns
        OPPDLS2hm=1.5 #Overprice for handworks and machinery for repair actions of DLS-2 in columns
        OPPDLS3hm=1.5 #Overprice for handworks and machinery for repair actions of DLS-3 in columns
        

    # In this script unitary costs for the materials are given
    # Reference:
    # A.N.A.S. S.p.A. [2020] LISTINO PREZZI 2020, NC-MS.2018 – REV.1, Nuove Costruzioni –
    # Manutenzione Straordinaria, Direzione Ingegneria e Verifiche, Roma, Italy. 
    # (in Italian) available at http://www.stradeanas.it/it/elenco-prezzi
    @staticmethod
    def CONCCOST(fc): # concrete costs (EUR/m3)
        # ANAS unitary prices list B.03.035 and B.03.040
        if fc>=55:
            costconc=169.76
        elif fc>=50 and fc<55: 
            costconc=162.68
        elif fc>=45 and fc<50:
            costconc=132.58
        elif fc>=40 and fc<45:
            costconc=126.05
        elif fc>=35 and fc<40:
            costconc=118.22
        elif fc<35:
            costconc=106.46
    
        return costconc
    
    @staticmethod
    def STEELCOST(dbl): # reinforcing steel costs (EUR/kg)
        # ANAS unitary price list B.05.030 and B.05.031
        if dbl<=(10/1000):
            op=0.69
        elif dbl>(10/1000) and dbl<=(15/1000):
            op=0.54
        elif dbl>(15/1000) and dbl<=(20/1000):
            op=0.41
        elif dbl>(20/1000) and dbl<=(30/1000):
            op=0.35
        elif dbl>(30/1000):
            op=0.30
        coststeel=1.04+op
        return coststeel
    
    @staticmethod
    def PSTEELCOST(): # post-tensioning steel costs (EUR/kg)
        costpsteel=2.59 # ANAS unitary price list B.05.057 and B.05.060
        return costpsteel
    
    @staticmethod
    def RAILING(): #railing barrier costs (EUR/kg)
        costrail=2.15 # ANAS unitary price list B.05.017c
        return costrail
    
    @staticmethod
    def ASPHALT(): # asphalt costs (EUR/m3)
        costasphalt=95 # Assumed
        return costasphalt
    
    @staticmethod
    def FORMWORK(t): # formwork costs (EUR/m2)
        #t=1 for superstructure formwork
        #t=2 for substructure formwork
        #t=3 for substructure repair formwork
        #t=4 for self-supporting formwork for temporal support repairing
        if t==1:
            costformwork=37.62 # ANAS unitary price list B.04.001
        elif t==2:
            costformwork=22.19 # ANAS unitary price list B.04.003
        elif t==3:
            costformwork=30.76 # ANAS unitary price list B.04.004.f
        elif t==4:
            costformwork=364.64 # ANAS unitary price list B.04.012.a
        return costformwork
    
    @staticmethod
    def PILECOST(DPile): # pile construction cost (EUR/m)
        if DPile>=1.5:
            costpilecons=305.15+176.61 # ANAS Unitary price list B.02.035.d and B.02.046.d 
        elif 0.8<DPile<1.5:
            costpilecons=151.73 # ANAS Unitary price list B.02.035.b 
        elif DPile<=0.8:
            costpilecons=108.03 # ANAS Unitary price list B.02.035.a
        return costpilecons
    
    @staticmethod
    def FILL(): # landfill cost (EUR/m3)
        # Assumed
        fillcost=70
        return fillcost
    
    @staticmethod
    def EXCA(): # land excavation costs (EUR/m3)
        exccost=3.25 # ANAS Unitary price list A.01.001
        return exccost
    
    @staticmethod
    def DEMOL(t): # demolition costs (EUR/m3)
        if t==1:
            demolcost=99.68 # ANAS A.03.008, demolition of decks
        elif t==2:
            demolcost=25.3 # ANAS A.03.019, Demolition of substrucure elements
        elif t==3:
            demolcost=180.2 # ANAS A.03.007, demolition of foundation elements
        return demolcost
    
    @staticmethod
    def CLNCOST(): # Unitary cost for cleaning and superficial treatment (EUR/m2) 
        cleaningcost=21.18 # ANAS Unitary price list B.09.212
        return cleaningcost
    
    @staticmethod
    def CONPATCHCOST(): # Unitary cost for concrete patch (EUR/m3)
        concretepatchcost=178.76  # ANAS Unitary price list B.04.003
        return concretepatchcost
    
    @staticmethod
    def CRACKSEALCOST(): # Unitary cost for crack sealing (EUR/m)
        concretesealingcost = 191 # Assumed
        return concretesealingcost
    
    @staticmethod
    def DEMOLCOVERCOST(): # Unitary cost of demolition of cover concrete (EUR/m3)
        conccoverdemolcost=289.84 # ANAS Unitary price list ANAS A.03.007
        return conccoverdemolcost
    
    @staticmethod
    # For elastomer bearings with a volume between 10 and 50 dm3
    def ELASTOBRCOST(Lsup):
        # Unitary bearing cost is given EUR per concrete topping volume under the bearing
        ubearingcost=43.24 # Unitary cost for elastomeric bearings (EUR/dm3) (ANAS B.07.010)
        if Lsup<=133:
            uvb=3*3*0.5 # approximate support volume per bearing
        elif Lsup>133 and Lsup<=200:
            uvb=4*4*0.5 # approximate support volume per bearing
        elif Lsup>200:
            uvb=5*5*0.5 # approximate support volume per bearing
        bearingcost = uvb*ubearingcost
        return bearingcost

    def _get_summary(self):
        
        ######### SUPERSTRUCTURE MATERIALS QUANTIFICATION #########
        SlabConc_Vol = 0; SlabSteel_Weight = 0; BeamsConc_Vol = 0; BeamsSteel_Weight = 0
        BeamsPreSteel_Weight = 0; Asphalt_Vol = 0; Rail_Weight = 0; Ltot = 0
        fc_super = []
        Lspans = []
        
        DeckArea = self.model['Deck']['A']
        fc_super = (self.model['Deck']['E']/(5000*MPa))**2
        
        for i in range(self.num_spans):
            Coord1 = ops.nodeCoord(self.D1Nodes[i])
            Coord2 = ops.nodeCoord(self.D2Nodes[i])
            Lele = ((Coord2[0]-Coord1[0])**2 + (Coord2[1]-Coord1[1])**2 + (Coord2[2]-Coord1[2])**2)**0.5
            
            Ltot += Lele
            Lspans.append(Lele)
                
            # Assumption #1: The Slab area corresponds to certain percentage of the total deck area
            SlabArea=SATDA*DeckArea
            BeamAreas=DeckArea-SlabArea #Cross section area corresponding to the beams
            
            # It is possible to compute the approximated slab concrete Volume and
            #reinforcement steel Weight
            SlabConc_Vol+=SlabArea*Lele #Concrete Volume of the slab
            SlabSteel_Weight+=SRslab*SlabArea*Lele #Reinforcement steel Weight of the slab
            
            # It is possible to compute the approximated longitudinal beams concrete
            # _Volume and reinformcement steel Weight
            BeamsConc_Vol+=BeamAreas*Lele #Concrete Volume of the longitudinal beams
            BeamsSteel_Weight+=SRbeams*BeamAreas*Lele #Reinforcement steel Weight of the slab
            
            # Assumption #2: The slab has a typical average thickness
            # With Assumptions 1 and 2, it is possible to compute the superstructure width
            W_deck=SlabArea/AST #Deck Width
            NBEAMS=int(W_deck/ASLB)+1 #Number of superstructure longitudinal beams
        
            # It is also possible to approximate the Weight of pre-stressing steel
            BeamsPreSteel_Weight+=SRPbeams*W_deck*Lele #Reinforcement steel Weight of the slab
            
            # Weight of the railing elements
            Rail_Weight+=2*ARR*Lele
            
            # Volume of deck asphalt
            Asphalt_Vol+=W_deck*Lele*AAT

        ######### ABUTMENT MATERIAL QUANTIFICATION #########
        aw=(NBEAMS-1)*ASLB+2 #Abutment width
        if aw<W_deck:
            aw=W_deck
    
        lspan1=Lspans[0] #Length first span
        lspan2=Lspans[-1] #Length last span
        #Height of back-wall abutment 1
        if lspan1<=20:
            bwh1=1.3
        elif lspan1>20 and lspan1<=30:
            bwh1=1.9
        elif lspan1>30:
            bwh1=2.4
    
        #Height od back-wall abutment 2
        if lspan2<=20:
            bwh2=1.3
        elif lspan2>20 and lspan2<=30:
            bwh2=1.9
        elif lspan2>30:
            bwh2=2.4
        
        #Concrete Volume abutment 1
        Vol1=bwh1*aw
        Vol2=(AAH-bwh1)*aw
        Vol3=AFW*AFT*aw
        Vol4=2*AWWT*AWWL*AAH
        Abut1Conc_Vol=Vol1+Vol2+Vol3+Vol4 #Concrete Volume abutment 1
        
        #Concrete Volume abutment 2
        Vol1=bwh2*aw
        Vol2=(AAH-bwh2)*aw
        Vol3=AFW*AFT*aw
        Vol4=2*AWWT*AWWL*AAH
        Abut2Conc_Vol=Vol1+Vol2+Vol3+Vol4 #Concrete Volume abutment 2
        
        #Steel weigth for abutment 1 and 2
        Abut1Steel_Weight=SRabut*Abut1Conc_Vol #Steel Weight abutment 1
        Abut2Steel_Weight=SRabut*Abut2Conc_Vol #Steel Weight abutment 2
        
        #Excavation and fill Volume
        AbutExc_Vol=2*aw*AAH*(AWWL+2) #Excavation Volume for both abutments
        AbutFill_Vol=2*aw*AAH*(AWWL+2) #Earth fill Volume for both abutments
    
        ######### PIERS MATERIAL QUANTITIES #########
        #########   PIER CONSTRUCTION COST  #########
        HPiers = []
        DPiers = []
        fyPiers = []
        fcPiers = []
        rholongPiers = []
        rhotransPiers = []
        alrsPiers = []
        
        PiersConc_Vol = []
        PiersSteel_Weight = []
        dblPiers = []
        DCovers = []

        PierCons_Costs = [] # for each pier
        fw_type=2 #Type of formwork
        for i in range(self.num_bents):
            idx = self.model['Bent']['Sections'][i] - 1
            D = self.model['Bent']['D'][idx]  # Section diameter
            Fce = self.model['Bent']['Fce'][idx]  # Concrete compressive strength
            E = 5000 * MPa * (Fce / MPa) ** 0.5  # Concrete Elastic Modulus
            G = E / (2 * (1 + 0.2))  # Concrete Elastic Shear Modulus
            Fyle = self.model['Bent']['Fyle'][idx]
            Fyhe = self.model['Bent']['Fyhe'][idx]
            cc = self.model['Bent']['cover'][idx]
            numBars = self.model['Bent']['numBars'][idx]
            dl = self.model['Bent']['dl'][idx]
            s = self.model['Bent']['s'][idx]
            dh = self.model['Bent']['dh'][idx]
            TransReinfType = self.model['Bent']['TransReinfType'][idx]
            Hele = self.model['Bent']['H'][i]
            for j in range(len(self.EleIDsBent[i])):
                Pele = self.BentAxialForces[i][j]    
                ds = D-2*cc  # Core diameter
                cs = s - dh  # Clear vertical spacing between spiral or hoop bars
                Acc = np.pi*ds**2/4  # Area of core of section enclosed by the center lines of the perimeter spiral or hoop
                Asl = numBars*(np.pi*dl**2)/4  # Total area of longitudinal steel reinforcements
                Ag = np.pi*D**2/4   # Total area of concrete section
                pcc = Asl/Acc  # Ratio of area of longitudinal reinforcement to area of core of section
                pl = Asl/Ag  # Ratio of area of longitudinal reinforcement to area of gross section
                
                # Confinement effectiveness coefficient
                if TransReinfType == 'Hoops':
                    ke  = (1-cs/2/ds)**2/(1-pcc)
                elif TransReinfType == 'Spirals':
                    ke  = (1-cs/2/ds)/(1-pcc)
                ps = 4*(np.pi*dh**2/4)/(ds*s)  # Ratio of the volume of transverse
                psconf=max(0.45*(Ag/Acc-1)*Fce/Fyhe,0.12*Fce/Fyhe) # code requirement by AASHTO
                psratio = ps/psconf 
                alr = Pele/(Ag*Fce)
                
                # These are necessary for cost estimation
                Conc_Vol = Ag*Hele
                lb_Vol = Asl*(Hele+2) # Long Reinf. Assuming 2m for anchorage (could be inconsistent) 
                Atb = np.pi*dh**2/4 # Trans. Reinf. Area
                tb_Vol=(Atb*(np.pi*ds+1.5))*round((Hele+2)/s) # Assuming 1.5m for hooks and splices
                Steel_Weight = (tb_Vol+lb_Vol)*gammas
                
                PiersConc_Vol.append(Conc_Vol)
                PiersSteel_Weight.append(Steel_Weight)
                dblPiers.append(dl)
                DCovers.append(cc)
                
                # These are necessary to pier generate damage models
                HPiers.append(Hele)
                DPiers.append(D)
                fyPiers.append(Fyle/MPa)
                fcPiers.append(Fce/MPa)   
                rhotransPiers.append(psratio)
                rholongPiers.append(pl)
                alrsPiers.append(alr)

                # Pier Concrete Cost
                costpierconc=self.CONCCOST(Fce)
                PierConc_Cost= Conc_Vol*costpierconc
                
                # Pier Steel Cost
                costpiersteel=self.STEELCOST(dl)
                PierSteel_Cost=Steel_Weight*costpiersteel

                # Pier Formwork Cost
                costpierformwork=self.FORMWORK(fw_type)
                APierFW=Hele*np.pi*D
                PierFormwork_Cost=costpierformwork*APierFW
                
                # Pier Construction Cost
                PierCons_Costs.append(OPPhm*(PierConc_Cost+PierSteel_Cost+PierFormwork_Cost))
        PierCons_Cost = sum(PierCons_Costs)
        ######### CAP-BEAM MATERIAL QUANTITIES #########
        L_bc = self.model['BentCap']['L']
        b_bc = self.model['BentCap']['w']
        h_bc = self.model['BentCap']['h']
        # B1=(NBEAMS-1)*ASLB+2
        # B2=DPier+1; # Assuming 0.5 'capitel' at each side
        # CapBeamConcVol=self.num_bents*(B1*0.5*CBH*(DPier+0.5)+0.5*(B1+B2)*0.5*CBH*(DPier+0.5)) # trapezoidal shape
        # ACapBeamFW=self.num_bents*(2*(B1*0.5*CBH+(0.5*0.5*CBH*(B1+B2)))+2*0.5*CBH*(DPier+0.5)+2*(DPier+0.5)*sqrt((0.5*CBH)^2+(0.5*(B2-B1))^2))
        CapBeamConc_Vol=self.num_bents*(b_bc*h_bc*L_bc) # rectangular shape
        CapBeamSteel_Weight=SRcapbeam*CapBeamConc_Vol
        ACapBeamFW = self.num_bents*(2*(b_bc*h_bc)+2*(L_bc*h_bc)+(b_bc*L_bc))
        
        ######### FOUNDATION FOOTING MATERIAL QUANTITIES #########
        if self.model['Bent_Foundation']['Type'] == 'Fixed' or self.model['Bent_Foundation']['Type'] == 'Springs':
            if Vs30>=760:
                FootingConc_Vol=self.num_bents*(6.5*6.5*2.5)
                AFootingFW=self.num_bents*((2*6.5*6.5)+4*(6.5*2.5))
            elif Vs30>=200 and Vs30<760:
                FootingConc_Vol=self.num_bents*(7.5*7.5*2.0)
                AFootingFW=self.num_bents*((2*7.5*7.5)+4*(6.5*2.0))
            elif Vs30<200:
                FootingConc_Vol=self.num_bents*(8.5*8.5*1.5)
                AFootingFW=self.num_bents*((2*8.5*8.5)+4*(8.5*1.5))
            # PILE FOUNDATION MATERIAL QUANTITIES
            if Vs30>=760:
                PileLength=self.num_bents*(4*10)
                DPile=1.50
            elif Vs30>=200 and Vs30<760:
                PileLength=self.num_bents*(9*15)
                DPile=1.0
            elif Vs30<200:
                PileLength=self.num_bents*(16*20)
                DPile=0.80
            PileVolume=0.25*np.pi*(DPile**2)*PileLength
            FootingSteel_Weight=SRfooting*FootingConc_Vol
        elif self.model['Bent_Foundation']['Type'] == 'Pile-Shaft':
            PileVolume=0
            DPiles = []
            PileLengths = []
            for i in range(self.num_bents):
                data = pd.read_excel(open('SoilProfiles.xlsx', 'rb'),
                                     sheet_name='Bent' + str(i + 1))
                PileLengths.append(np.sum(data['Thickness'])*self.model['Bent']['N'])
                idx = self.model['Bent_Foundation']['Sections'][i]-1
                DPiles.append(self.model['Bent_Foundation']['D'][idx])
                PileVolume += 0.25*np.pi*(DPiles[i]**2)*PileLengths[i]
            DPile = np.mean(DPiles)
            PileLength = np.sum(PileLengths)
            AFootingFW=0
            FootingConc_Vol = 0
            FootingSteel_Weight = 0
        elif self.model['Bent_Foundation']['Type'] == 'Group Pile':
            PileVolume=0
            DPiles = []
            PileLengths = []
            for i in range(self.num_bents):
                data = pd.read_excel(open('SoilProfiles.xlsx', 'rb'),
                                     sheet_name='Bent' + str(i + 1))
                PileLengths.append(np.sum(data['Thickness'])*self.model['Bent_Foundation']['nx'][idx]*self.model['Bent_Foundation']['ny'][idx])
                idx = self.model['Bent_Foundation']['Sections'][i]-1
                DPiles.append(self.model['Bent_Foundation']['D'][idx])
                PileVolume += 0.25*np.pi*(DPiles[i]**2)*PileLengths[i]
            cap_A = self.model['Bent_Foundation']['cap_A']
            cap_t = self.model['Bent_Foundation']['cap_t']
            AFootingFW=self.num_bents*((2*cap_A)+4*(cap_A/10*cap_t))
            FootingConc_Vol = self.num_bents*cap_A*cap_t
            FootingSteel_Weight=SRfooting*FootingConc_Vol
            DPile = np.mean(DPiles)
            PileLength = np.sum(PileLengths)
        ######### SUPERSTRUCTURE CONSTRUCTION COST #########
        # Superstructure Concrete Cost
        costsupconc=self.CONCCOST(fc_super)
        SupConc_Vol=SlabConc_Vol+BeamsConc_Vol
        SupConc_Cost=SupConc_Vol*costsupconc
        # Superstructure Steel Cost
        dbl_super = np.mean(dblPiers) # yet another assumption
        costsupsteel=self.STEELCOST(dbl_super)
        SupSteel_Weight=SlabSteel_Weight+BeamsSteel_Weight
        SupSteel_Cost=SupSteel_Weight*costsupsteel
        # Superstructure Pre-Stressing Steel Cost
        costsuppsteel = self.PSTEELCOST()
        SupPSteel_Cost=BeamsPreSteel_Weight*costsuppsteel
        # Superstructure Railing Cost
        costsuprail=self.RAILING()
        SupRail_Cost=Rail_Weight*costsuprail
        # Superstructur Asphalt Layer Cost
        costsupasphalt=self.ASPHALT()
        SupAsphalt_Cost=Asphalt_Vol*costsupasphalt
        # Superstructure Formwork Cost
        fw_type=1 # Type of formwork
        costsupformwork=self.FORMWORK(fw_type)
        SupFormwork_Cost=Ltot*W_deck*costsupformwork
        # Superstructure Construction Cost
        SupCons_Cost=OPShm*(SupConc_Cost+SupSteel_Cost+SupPSteel_Cost+SupRail_Cost+SupAsphalt_Cost+SupFormwork_Cost)
    
        ######### ABUTMENT CONSTRUCTION COST #########
        # Abutment 1 Concrete Cost
        costabt1conc=self.CONCCOST(fc_super)
        Abt1Conc_Cost=Abut1Conc_Vol*costabt1conc
        #Abutment 2 Concrete Cost
        costabt2conc=self.CONCCOST(fc_super)
        Abt2Conc_Cost=Abut2Conc_Vol*costabt2conc
        #Abutment 1 Steel Cost
        costabt1steel=self.STEELCOST(dbl_super)
        Abt1Steel_Cost=Abut1Steel_Weight*costabt1steel
        #Abutment 2 Steel Cost
        costabt2steel=self.STEELCOST(dbl_super)
        Abt2Steel_Cost=Abut2Steel_Weight*costabt2steel
        #Abutment Excavation Cost
        abtexccost=self.EXCA()
        AbtExc_Cost=AbutExc_Vol*abtexccost
        #Abutment Fill Grading Cost
        abtfillcost=self.FILL()
        AbtFill_Cost=AbutFill_Vol*abtfillcost
        #Abutment Formwork Cost
        fw_type=2 #Type of formwork
        costabtformwork=self.FORMWORK(fw_type)
        AbtFormwork_Cost=2*costabtformwork*(2*aw*AAH+2*AAH*ASWT+4*AAH*AWWL)
        #Abutment Construction Cost
        AbtCons_Cost=OPAhm*(Abt1Conc_Cost+Abt2Conc_Cost+Abt1Steel_Cost+Abt2Steel_Cost+AbtExc_Cost+AbtFill_Cost+AbtFormwork_Cost)
    
        ######### CAP-BEAM CONSTRUCTION COST #########
        # Cap-Beam Concrete Cost
        fc_bc = fc_super
        costcapbeamconc=self.CONCCOST(fc_bc)
        CapBeamConc_Cost=CapBeamConc_Vol*costcapbeamconc
        # Cap-Beam Steel Cost
        dbl_bc = dbl_super
        costcapbeamsteel=self.STEELCOST(dbl_bc)
        CapBeamSteel_Cost=CapBeamSteel_Weight*costcapbeamsteel
        # Cap-Beam Formwork Cost
        fw_type=2 #Type of formwork
        costcapbeamformwork=self.FORMWORK(fw_type)
        CapBeamFormwork_Cost=costcapbeamformwork*ACapBeamFW
        # Cap-Beam Construction Cost
        CapBeamCons_Cost=OPCPhm*(CapBeamConc_Cost+CapBeamSteel_Cost+CapBeamFormwork_Cost)
    
        ######### FOOTING CONSTRUCTION COST #########
        # Footing Concrete Cost
        fc_found = fc_super
        costcfootingconc=self.CONCCOST(fc_found)
        FootingConc_Cost=FootingConc_Vol*costcfootingconc
        # Footing Steel Cost
        dbl_found = dbl_super
        costfootingsteel=self.STEELCOST(dbl_found)
        FootingSteel_Cost=FootingSteel_Weight*costfootingsteel
        # Footing Formwork Cost
        fw_type=2 #Type of formwork
        costfootingformwork=self.FORMWORK(fw_type)
        FootingFormwork_Cost=costfootingformwork*AFootingFW
        # Footing Construction Cost
        FootingCons_Cost=OPFhm*(FootingConc_Cost+FootingSteel_Cost+FootingFormwork_Cost)
    
        ######### PILES CONSTRUCTION COST #########
        UCostPileCons = self.PILECOST(DPile)
        PileCons_Cost=OPPIhm*(UCostPileCons*PileLength)
    
        ######### BRIDGE CONSTRUCTION COST #########
        # Total Brige Construction Cost
        BridgeCons_Cost=SupCons_Cost+AbtCons_Cost+PierCons_Cost+CapBeamCons_Cost+FootingCons_Cost+PileCons_Cost # (EUR)
        # Unitary Bridge Construction Cost
        UBridgeCons_Cost=BridgeCons_Cost/(Ltot*W_deck) # this is for comparison with typical values in the market, literature (EUR/m2)
    
        ######### DEMOLITION COST #########
        #Superstructure demolition
        demol_type=1
        supdemolcost=self.DEMOL(demol_type)
        SupDemol_Cost=supdemolcost*(SlabConc_Vol+BeamsConc_Vol)
        # Piers, Abutment and Footing Demolition
        demol_type=2
        subdemolcost=self.DEMOL(demol_type)
        SubDemol_Cost=subdemolcost*(Abut1Conc_Vol+Abut2Conc_Vol+AbutFill_Vol+sum(PiersConc_Vol)+CapBeamConc_Vol+FootingConc_Vol)
        # Piles Demolition
        demol_type=3
        piledemolcost=self.DEMOL(demol_type)
        PileDemol_Cost=piledemolcost*PileVolume
        # Bridge Demolition Cost
        BridgeDemol_Cost=SupDemol_Cost+SubDemol_Cost+PileDemol_Cost
    
        ######### BRIDGE REPLACEMENT COST #########
        # Total Bridge Replacement Cost
        BridgeRep_Cost=BridgeCons_Cost+BridgeDemol_Cost # (EUR)
        # Unitary Bridge Replacement Cost
        UBridgeRep_Cost=BridgeRep_Cost/(Ltot*W_deck) # (EUR/m2)
    
        ######### REPAIR ACTIONS MEAN COSTS #########
        # Mean Repair Costs for Abutments
        AbutMeanRepair_Cost=np.zeros([2,3])
        for i in range(2):
            for j in range(3):
            
                if j==0: # Abutment Repair Cost for DLS-1
                    # Repair Action: Cleaning and Superficial Treatment
                    UCC=self.CLNCOST() # Unitary cost for cleaning and superficial treatment (EUR/m2)
                    ACC=2*AWWL*AAH+AAH*aw # Area of cleaning cost and superficial treatment
                    TCC=OPADLS1hm*ACC*UCC # Total cost for cleaning cost and superficial treatment
                    AbutMeanRepair_Cost[i,j]=TCC
                    
                elif j==1: # Abutment Repair Cost for DLS-2
                    # Repair Action: Cleaning and Superficial Treatment
                    UCC=self.CLNCOST() # Unitary cost for cleaning and superficial treatment (EUR/m2)
                    ACC=2*AWWL*AAH+AAH*aw #Area of cleaning cost and superficial treatment
                    TCC=ACC*UCC #Total cost for cleaning cost and superficial treatment
                    
                    # Repair Action: Concrete patch in affected wing-walls
                    UCPC=self.CONPATCHCOST() # Unitary cost for concrete patch (EUR/m3)
                    co=0.05 # Assumed concrete cover thickness (m)
                    VPC=2*co*AWWL*AAH # Concete volumne for patching
                    TCPC=UCPC*VPC # Total cost for concrete patching
                    
                    # Rapair Action: Formwork for concrete patch
                    fw_type = 3
                    UCFCP=self.FORMWORK(fw_type) # Unitary cost for formwork
                    AFCP=2*AWWL*AAH # Area of formwork for concrete patch
                    TCFCP=UCFCP*AFCP # Total cost of formwork for concrete patching
                    
                    # Repair Action: Backfill excavation in the height of the backwall
                    UCE=self.EXCA() # Unitary cost for excavation
                    if i==0:
                        bwh=0.5*bwh1
                    elif i==1:
                        bwh=0.5*bwh2
    
                    VE=bwh*aw*(AWWL+2) # Excavation volume
                    TCE=UCE*VE # Total Cost of Excavation
                    
                    # Repair Action: Crack sealing on the backfill side
                    UCS=self.CRACKSEALCOST() # Unitary cost for crack sealing (EUR/m)
                    ASC=0.4 # Assumed average spacing of cracks to seal
                    NCS=2*(round(bwh/ASC)+1) # Number of crack to seal
                    TCS=UCS*0.5*AWWL*NCS # Total cost of sealing cracks
                    
                    # Repair Action: Backfill replacement
                    UCF=self.FILL() # Unitary cost of fill replacement
                    TCF=0.25*UCF*VE # Total Cost of fill replacement
                    
                    # Repair Action: Asphalt Replacement
                    UCA=self.ASPHALT()
                    VA=W_deck*AAT*(AWWL+2)
                    TCA=UCA*VA
                    
                    # Total Repair Cost
                    TRC=OPADLS2hm*(TCC+TCPC+TCFCP+TCE+TCS+TCF+TCA)
                    AbutMeanRepair_Cost[i,j]=TRC
                                        
                elif j==2: # Abutment Repair Cost for DLS-3
                    
                    # Repair Action: Temporal Support of Superstructure
                    fw_type = 4
                    UCTS=self.FORMWORK(fw_type) # Unitary cost of temporal support
                    if i==0: # Length to be supported ≈ 15%
                        LTS=0.15*Lspans[0] 
                    elif i==1:
                        LTS=0.15*Lspans[-1]
                    ATS=W_deck*LTS # Area of temporal support
                    TCTS=UCTS*ATS # Total Cost of temporal support
                    
                    # Repair Action: Backfill Excavation
                    UCBE=self.EXCA() # Unitary cost of excavation
                    VE=W_deck*AAH*(AWWL+2) # Excavation volume
                    TCBE=UCBE*VE
                    
                    # Repair Action: Abutment Demolition
                    demol_type = 2
                    UCAD=self.DEMOL(demol_type)
                    if i==0: # Volume to be demolished
                        VAD=Abut1Conc_Vol
                    elif i==1:
                        VAD=Abut2Conc_Vol
                    TCAD=UCAD*VAD # Total cost of abutment demolition
                    
                    # Repair Action: Abutment structure replacement
                    UCC=self.CONCCOST(fc_super) # Unitary Cost of Concrete
                    UCS=self.STEELCOST(dbl_super) # Unitary Cost of Steel
                    if i==0: # Concrete Volumne to be constructed
                        VA=Abut1Conc_Vol
                    elif i==1:
                        VA=Abut2Conc_Vol
    
                    if i==0: #Steel Weight to be constructed
                        SA=Abut1Steel_Weight
                    elif i==1:
                        SA=Abut2Steel_Weight
    
                    TCAC=UCC*VA+UCS*SA # Total cost of abutment construction
                    
                    # Repair Action: Abutment Backfill regrading
                    UCF=self.FILL() # Unitary cost of fill
                    VF=W_deck*AAH*(AWWL+2) # Fill Volume
                    TCF=UCF*VF # Total cost of abutment regrading
                    
                    # Repair Action: Asphalt replacing
                    UCA=self.ASPHALT()
                    VA=W_deck*AAT*(AWWL+2) # Asphalt Volume
                    TCA=UCA*VA
                    
                    # Total Repair Cost
                    TRC=OPADLS3hm*(TCTS+TCBE+TCAD+TCAC+TCF+TCA)
                    AbutMeanRepair_Cost[i,j]=TRC
        
        
        # Mean Repair Costs for Piers
        PierMeanRepair_Cost=np.zeros([len(HPiers),3])
        for i in range(self.num_bents):
            
            NPIERS = len(self.EleIDsBent[i])
            
            for k in range(len(self.EleIDsBent[i])):
    
                p_idx = NPIERS*i + k
                
                for j in range(3):  # DLS besides the collapse
                
                    if j==0: # Pier Repair Cost for DLS-1
                        
                        # Repair Action: Sealing of Cracks
                        UCSC=self.CRACKSEALCOST() # Unitary cost of crack sealing (EUR/m)
                        UCL=0.5*np.pi*DPiers[p_idx] # Unitary crack length to be sealed
                        LDC=0.8*DPiers[p_idx] # Length over which cracks to be sealed develop
                        ACS=0.25*DPiers[p_idx] # Average spacing of cracks to be sealed
                        NCS=round(LDC/ACS)+1
                        TLCS=UCL*NCS # Total length of cracks to be sealed
                        TCCS=OPPDLS1hm*UCSC*TLCS # Total cost of crack sealing
                        PierMeanRepair_Cost[p_idx,j]=TCCS
                        
                    elif j==1: # Column Repair Cost for DLS-2
                        
                        # Repair Action: Demolition of cover concrete
                        UCCD=self.DEMOLCOVERCOST() # Unitary cost of demolition of cover concrete (EUR/m3)
                        AD=(0.25*np.pi*DPiers[p_idx]**2)-(0.25*np.pi*(DPiers[p_idx]-2*DCovers[p_idx])**2) # Demolition area - unconfined area
                        if HPiers[i]>3*DPiers[p_idx]:
                            LPier=3*DPiers[p_idx]
                        else:
                            LPier=HPiers[p_idx]
                        VD=AD*LPier # Demolition volumne
                        TCCCD=UCCD*VD
                        
                        # Repair Action: Cleaning and Surface Preparing
                        UCC=self.CLNCOST() # Unitary cost for cleaning and superficial treatment (EUR/m2)
                        AC=np.pi*DPiers[p_idx]*HPiers[p_idx] # Surface area of cleaning and preparing
                        TCC=UCC*AC # Total cost of cleaning and surface preparing
                        
                        # Repair Action: Concrete Patching
                        UCCP=self.CONPATCHCOST() # Unitary cost for concrete patch (EUR/m3)
                        TCCP=UCCP*VD # Total Cost of Concrete Patching
                        
                        # Repair Action: Formwork for patching
                        fw_type = 3
                        UCFCP=self.FORMWORK(fw_type) # Unitary cost for formwork
                        AFCP=np.pi*DPiers[p_idx]*LPier # Area of formwork for concrete patch
                        TCFCP=UCFCP*AFCP # Total cost of formwork for concrete patching
                        
                        # Total cost of concrete patching
                        TOCCP=OPPDLS2hm*(TCCCD+TCC+TCCP+TCFCP)
                        PierMeanRepair_Cost[p_idx,j]=TOCCP
                        
                    elif j==2: # Column Repair Cost for DLS-3
                        # in practice, I believe all the bent will be replaced, not the single pier.
                        # But anyways, lets assume that only a pier will be replaced together with
                        # Beam-cap, otherwise, I am not so sure how to calculate the cost.
                        # Most probably, the thicker section will never be in this damage state,
                        # While the thinner section will be in DLS-3 or DLS-4
                        
                        # Repair Action: Temporal Superstructure Support
                        fw_type = 4
                        UCTS=self.FORMWORK(fw_type) # Unitary cost of temporal support
                        LTS=0.15*0.5*(Lspans[i]+Lspans[i+1]) # assume 15%
                        ATS=W_deck*LTS # Area of temporal support
                        TCTS=UCTS*ATS # Total Cost of temporal support
                        
                        # Repair Action: Demolition of Existing Damaged Column
                        demol_type = 3
                        UCD=self.DEMOL(demol_type) # Unitary cost of demolition
                        VD=0.25*np.pi*(DPiers[p_idx]**2)*HPiers[p_idx]+(CapBeamConc_Vol/self.num_bents)
                        TCD=UCD*VD
                        
                        # Repair Action: Intervention in the Foundation
                        UCFI=self.DEMOLCOVERCOST() # Unitary cost of demolition of cover concrete (EUR/m3)
                        if DPile>=1.5:
                            hf=2.5
                        elif DPile>0.8 and DPile<1.5:
                            hf=2
                        elif DPile<=0.8:
                            hf=1.5
                        VFI=0.25*np.pi*(DPiers[p_idx]**2)*0.8*hf # Embedded volume of pier
                        TCFI=UCFI*VFI
                         
                        # Repair Action: Columns and Cap-Beam Construction
                        UCC=self.CONCCOST(fcPiers[p_idx]) # Unitary Cost of Concrete
                        UCS=self.STEELCOST(dblPiers[p_idx]) # Unitary Cost of Steel
                        PV=PiersConc_Vol[p_idx]
                        PS=PiersSteel_Weight[p_idx]
                        CBV=CapBeamConc_Vol/NPIERS
                        CBS=CapBeamSteel_Weight/NPIERS
                        TCCRC=UCC*(PV+CBV)+UCS*(PS+CBS)
                        
                        # Repair Action: Elastomeric bearings replacement
                        # Let's add this for now although it is not true
                        # later separate calculations for bearings can be added, and this part can be removed
                        Lsup = 0.5*(Lspans[i]+Lspans[i+1]) # support deck length
                        UCEB=self.ELASTOBRCOST(Lsup) # Unitary cost for elastomeric bearings (EUR)
                        TOEB=UCEB*2*NBEAMS # Assume pair of bearings for each girder
                         
                        # Repair action: Formwork for reconstruction of pier
                        fw_type = 3
                        UCFCP=self.FORMWORK(fw_type) # Unitary cost for formwork
                        APF=HPiers[p_idx]*np.pi*DPiers[p_idx] # Area of column formwork
                        ACBF=ACapBeamFW/self.num_bents # Area of cap-beam formwork
                        TCPFW=UCFCP*(APF+ACBF)
                       
                        # Total cost of pier replacement
                        TCPR=OPPDLS3hm*(TCTS+TCD+TCFI+TCCRC+TOEB+TCPFW)
                        # TCPR=OPPDLS3hm*(TCD+TCFI+TCCRC+TOEB+TCPFW)
                        PierMeanRepair_Cost[p_idx,j]=TCPR
                    
        # Output directory to
        # save necessary info
        info_dir = os.path.join(self.out_dir,'MeanRepairCosts')
        create_outdir(info_dir)
        
        # Save necessary info to generate pier damage models
        np.savetxt(os.path.join(info_dir,'ALRS.out'),np.asarray(alrsPiers))
        np.savetxt(os.path.join(info_dir,'DPier.out'),np.asarray(DPiers))
        np.savetxt(os.path.join(info_dir,'rholong.out'),np.asarray(rholongPiers))
        np.savetxt(os.path.join(info_dir,'rhotrans.out'),np.asarray(rhotransPiers))
        np.savetxt(os.path.join(info_dir,'HPiers.out'),np.asarray(HPiers))
        np.savetxt(os.path.join(info_dir,'fy.out'),np.asarray(fyPiers))
        np.savetxt(os.path.join(info_dir,'fc.out'),np.asarray(fcPiers)) 
        
        # Save necessary info to estimate direct seismic losses
        np.savetxt(os.path.join(info_dir,'BRepCost.out'),np.array(BridgeRep_Cost, ndmin=1))
        np.savetxt(os.path.join(info_dir,'UBRepCost.out'),np.array(UBridgeRep_Cost, ndmin=1))
        np.savetxt(os.path.join(info_dir,'PierMeanRepairCost.out'),PierMeanRepair_Cost)
        np.savetxt(os.path.join(info_dir,'AbutMeanRepairCost.out'),AbutMeanRepair_Cost)