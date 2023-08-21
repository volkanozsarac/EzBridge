"""
This module is used to perform seismic risk assessment of the bridge models
analyzed using "structural_analysis" module.

Author: Volkan Ozsarac, Earthquake Engineering PhD Candidate
Affiliation: University School for Advanced Studies IUSS Pavia
e-mail: volkanozsarac@iusspavia.it
"""

#  ----------------------------------------------------------------------------
#  Import Python Libraries
#  ----------------------------------------------------------------------------
import pickle
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
from .utils import get_distance, get_units, create_dir, mle_fit, find_nearest, ecdf, normal_cdf, do_sampling

class _cost:
    """
    In this script cost of repair actions and replacement cost for a bridge are computed

    References
    ----------
    A.N.A.S. S.p.A. [2022] LISTINO PREZZI 2022, NC-MS.2022 – REV.2, Nuove Costruzioni –
    Manutenzione Straordinaria, Direzione Ingegneria e Verifiche, Roma, Italy.
    (in Italian) available at http://www.stradeanas.it/it/elenco-prezzi
    Perdomo, C. 2020. Direct economic loss assessment of multi-span continuous RC bridges under seismic hazard.
    Ph.D. thesis, Istituto Universitario di Studi Superiori di Pavia, Italy.
    """

    def __init__(self, Vs30):
        # UNITARY COSTS FROM A.N.A.S.
        # Cost of prestressing steel (EUR/kg) based on B.05.057 and B.05.060
        self.prestressing_steel = 3.97
        # Cost of railing barrier (EUR/kg) based on B.05.017.c
        self.railing = 3.38
        # Cost of land excavation (EUR/m3) based on A.01.001
        self.excavation = 3.65
        # Cost of cleaning and superficial treatment (EUR/m2) based on B.09.212
        self.cleaning = 28.64
        # Cost of demolition of concrete cover (EUR/m3) based on A.03.007
        self.cover_demolition = 316.88
        # Cost of concrete (EUR/m3) --> varies based on concrete strength
        self.concrete = 132.86
        # Cost of reinforcing steel (EUR/kg) --> varies based on steel diameter
        self.reinforcing_steel = 2.38
        # Cost of formwork (EUR/m2) --> varies based on formwork type
        self.formwork = 40.23
        # Cost of a pile (EUR/m) --> varies with pile diameter
        self.pile = 180.22
        # Cost of a demolition (EUR/m3) --> varies with demolition case
        self.demolition = 107.43
        # # Cost of a neoprene bearing (EUR/dm3) with a volume between 10 and 50 dm3 based on B.07.010
        self.bearing = 66.33
        # Cost of jacking up the bridge deck (EUR/deck) --> assumed based on Kameshwar and Padgett, 2017
        self.jacking = 6000
        # Cost of concrete patch (EUR/m3) -- assumed based on concrete unit prices
        self.concrete_patch = 185
        # Cost of crack sealing (EUR/m) -- assumed
        self.crack_seal = 191
        # Cost of asphalt (EUR/m2) -- assumed
        self.asphalt = 100
        # Cost of land fill (EUR/m3) -- assumed
        self.fill = 80

        # AVERAGE MATERIAL QUANTITIES
        # Superstructure Related
        self.slab_deck_area_ratio = 0.5  # Average ratio for the slab area to total deck area [--]
        self.long_beam_spacing = 3.0  # Average Spacing of longitudinal beams [m]
        self.slab_thickness = 0.2  # Average slab thickness in [m]
        self.steel_ratio_slab = 150  # Average steel to concrete ratio for the slab [kg/m3]
        self.steel_ratio_beam = 100  # Average steel to concrete ratio for the longitudinal beams [kg/m3]
        self.pre_steel_ratio_deck = 17  # Average steel ratio of pre-stressing steel to deck area [kg/m2]
        self.railing_weight = 100  # Average railing weight per unit of length [kg/m]
        self.asphalt_thickness = 0.05  # Average asphalt thickness [m]
        self.d_super = 24  # Average diameter of the steel reinforcement [mm]
        # Abutment Related
        # Assumption: Seat-Type with shallow foundation
        self.abutment_height = 5.0  # Assumed average abutment height [m]
        self.abutment_backwall_thickness = 0.30  # Assumed abutment back-wall thickness [m]
        self.abutment_wingwall_thickness = 0.30  # Assumed abutment wing-wall thickness [m]
        self.abutment_wingwall_length = 4.00  # Assumed abutment wing-wall length [m]
        self.abutment_seatwall_thickness = 1.50  # Assumed abutment seat-wall thickness [m]
        self.abutment_foundation_width = 2.50  # Assumed abutment foundation footing width [m]
        self.abutment_foundation_thickness = 1.50  # Assumed abutment foundation footing thickness [m]
        self.abutment_steel_ratio = 90  # Average steel to concrete ratio for abutments [kg/m3]
        # Bearing related
        self.bearing_volume = 6 * 6 * 0.3  # Assuming pier dimensions are 6dm x 6dm x 0.2dm
        self.deck_alignment = 8000  # Assumed cost of deck alignment (EUR/deck)
        # Pier related
        # dbt=12 #Assumed diameter of the transverse reinforcement [mm]
        self.gamma_steel = 7850  # Average unitary steel Weight [kg/m3]
        # Bent-Cap related
        # CBH=2.00 #Assumed average height of the Cap-Beam [m], trapezoidal case, general
        self.bentcap_steel_ratio = 140  # Average steel to concrete ratio in bent cap [kg/m3]
        # Foundation related
        # Estimation of foundation quantities will be performed based on several
        # assumptions based on the expected type of soil as described by the Vs30:
        # If Vs30>=760m/s, the foundation is considered composed with 4
        # D=1.50m L=10m piles with 6.50x6.50x2.50 footing
        # If 200m/s<=Vs30<760m/s, the foundation is considered composed with 9
        # D=1.00m L=15m piles with 7.50x7.50x2.00 footing
        # If Vs30<200m/s the foundation is considered composed with 16 D=0.80m L=20m piles with 8.50x8.50x1.50 footing.
        self.footing_steel_ratio = 120  # Average steel to concrete ratio in footings [kg/m3]
        self.Vs30 = Vs30  # Shear Wave velocity for the first 30m [m/s]
        self.d_footing = 32  # Average reinforcement diameter [mm]
        self.fc_footing = 30  # Average concrete strength [MPa]

        # OVER PRICE FACTORS FOR CONSTRUCTION COSTS
        # Related in the corresponding functions
        self.ophm_super = 1.8  # Overprice for handwork and machinery in the superstructure
        self.ophm_abutment = 1.5  # Overprice for handwork and machinery in the abutment
        self.ophm_pier = 1.6  # Overprice for handwork and machinery in the piers
        self.ophm_bentcap = 1.6  # Overprice for handwork and machinery in the cap-beams
        self.ophm_footing = 1.5  # Overprice for handwork and machinery in the footings
        self.ophm_bearing = 1.8  # Overprice for handwork and machinery in the bearings
        self.ophm_pile = 1.8  # Overprice for handwork and machinery in the piles

        # OVER PRICE FACTORS FOR COST OF REPAIR ACTIONS
        # Related in their respective functions and/or computation lines
        self.ophm_dls1_abutment = 1.1  # Overprice for handworks and machinery for repair actions of DLS-1 in abutments
        self.ophm_dls2_abutment = 1.2  # Overprice for handworks and machinery for repair actions of DLS-2 in abutments
        self.ophm_dls3_abutment = 1.4  # Overprice for handworks and machinery for repair actions of DLS-3 in abutments

        self.ophm_dls1_pier = 1.15  # Overprice for handworks and machinery for repair actions of DLS-1 in columns
        self.ophm_dls2_pier = 1.5  # Overprice for handworks and machinery for repair actions of DLS-2 in columns
        self.ophm_dls3_pier = 1.5  # Overprice for handworks and machinery for repair actions of DLS-3 in columns

        self.ophm_dls1_bearing = 1.4  # Overprice for handworks and machinery for repair actions of DLS-1 in bearings
        self.ophm_dls2_bearing = 2.0  # Overprice for handworks and machinery for repair actions of DLS-2 in bearings
        self.ophm_dls3_bearing = 2.4  # Overprice for handworks and machinery for repair actions of DLS-3 in bearings

    def update_concrete(self, fc):
        """
        Updates concrete unit cost (EUR/m3) based on concrete strength

        Parameters
        ----------
        fc: float
            Concrete compressive strength (MPa)

        Returns
        -------

        """

        if 75 <= fc:  # B.03.040.e
            self.concrete = 237.07
        elif 67 <= fc < 75:  # B.03.040.d
            self.concrete = 230.07
        elif 60 <= fc < 67:  # B.03.040.c
            self.concrete = 220.89
        elif 55 <= fc < 60:  # B.03.040.b
            self.concrete = 212.84
        elif 50 <= fc < 55:  # B.03.040.a
            self.concrete = 203.64
        elif 45 <= fc < 50:  # B.03.035.e
            self.concrete = 166.81
        elif 40 <= fc < 45:  # B.03.035.d
            self.concrete = 158.31
        elif 37 <= fc < 40:  # B.03.035.c
            self.concrete = 152.38
        elif 35 <= fc < 37:  # B.03.035.b
            self.concrete = 148.13
        elif fc < 35:  # B.03.035.a
            self.concrete = 132.86

    def update_reinforcing_steel(self, diam):
        """
        Updates reinforcing steel unit cost (EUR/kg) based on diameter

        Parameters
        ----------
        diam: float
              diameter of reinforcing steel

        Returns
        -------

        """

        # get units to prepare the input with m, kN, sec
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units(0)

        # ANAS unitary price list for steel material B.05.030 (EUR/kg)
        material = 1.86

        # ANAS unitary price list for coating B.05.031 (EUR/kg)
        if diam <= 10 * mm:  # B.05.031.a
            coating = 0.91
        elif 10 * mm < diam <= 15 * mm:  # B.05.031.b
            coating = 0.69
        elif 15 * mm < diam <= 20 * mm:  # B.05.031.c
            coating = 0.52
        elif 20 * mm < diam <= 30 * mm:  # B.05.031.d
            coating = 0.43
        elif diam > 30 * mm:  # B.05.031.e
            coating = 0.36

        self.reinforcing_steel = material + coating

    def update_formwork(self, case):
        """
        Updates cost of formwork (EUR/m2) based on the formwork case

        Parameters
        ----------
        case: int
            1: superstructure formwork
            2: substructure formwork
            3: substructure repair formwork
            4: self-supporting formwork for temporal support repairing

        Returns
        -------

        """

        if case == 1:
            self.formwork = 40.23  # ANAS unitary price list B.04.003
        elif case == 2:
            self.formwork = 22.43  # ANAS unitary price list B.04.001
        elif case == 3:
            self.formwork = 44.27  # ANAS unitary price list B.04.004.f
        elif case == 4:
            self.formwork = 364.64  # ANAS unitary price list B.04.012.a

    def update_pile(self, diam):
        """
        Updates cost of pile (EUR/m) based on pile diameter

        Parameters
        ----------
        diam: float
            Pile diameter

        Returns
        -------

        """

        # get units to prepare the input with m, kN, sec
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units(0)

        if diam <= 800 * mm:  # B.02.035.a
            self.pile = 126.62
        elif 800 * mm < diam <= 1200 * mm:  # B.02.035.b
            self.pile = 180.22
        elif 1200 * mm < diam <= 1500 * mm:  # B.02.035.c
            self.pile = 250.72
        elif 1500 * mm < diam <= 1800 * mm:  # B.02.035.d
            self.pile = 368.61
        elif 1800 * mm < diam <= 2000 * mm:  # B.02.035.e
            self.pile = 524.01
        elif 2000 * mm < diam <= 2500 * mm:  # B.02.035.f
            self.pile = 627.93
        elif 2500 * mm < diam:  # B.02.035.g
            self.pile = 958.97

    def update_demolition(self, case):
        """
        Updates cost of demolition (EUR/m3) based on the demolition case

        Parameters
        ----------
        case: int
            1: Demolition of superstructure elements
            2: Demolition of substructure elements
            3: Demolition of foundation elements

        Returns
        -------

        """
        if case == 1:
            self.demolition = 107.43  # ANAS A.03.008, demolition of decks
        elif case == 2:
            self.demolition = 27.16  # ANAS A.03.019, Demolition of substructure elements
        elif case == 3:
            self.demolition = 191.93  # ANAS A.03.007, demolition of foundation elements

    def calculate(self, bridge_data):

        # Get units to prepare the input with m, kN, sec
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units(0)

        # Update bearing volume, can be different than default value
        if 'bearing_volume' in bridge_data.model.keys():
            self.bearing_volume = bridge_data.model['bearing_volume'] * 1000 # conversion to dm3

        # --------------------------------------------------------------
        # Superstructure Material Quantification and Construction Cost
        # --------------------------------------------------------------
        span_lengths = []  # Span lengths
        fw_type = 1  # Type of formwork
        super_cons_cost = 0  # Total construction cost
        super_tot_conc_volume = 0  # Total concrete volume for superstructure

        for i in range(bridge_data.num_spans):
            # TODO: Modify this part later, I can directly save span lengths in EzBridge instead.
            # Get span lengths directly from pickle file instead of computing using the coordinates
            Coord1 = [bridge_data.model['General']['Joint Xs'][i], bridge_data.model['General']['Joint Ys'][i],
                      bridge_data.model['General']['Joint Zs'][i]]
            Coord2 = [bridge_data.model['General']['Joint Xs'][i + 1], bridge_data.model['General']['Joint Ys'][i + 1],
                      bridge_data.model['General']['Joint Zs'][i + 1]]
            Lele = get_distance(Coord1, Coord2)
            span_lengths.append(Lele)
            deck_sec = bridge_data.model['General']['Decks'][i]
            deck_area = bridge_data.model[deck_sec]['A']
            fc = (bridge_data.model[deck_sec]['E'] / (5000 * MPa)) ** 2

            # Assumption #1: The Slab area corresponds to certain percentage of the total deck area
            slab_area = self.slab_deck_area_ratio * deck_area
            beam_areas = deck_area - slab_area  # Cross-section area corresponding to the beams
            # It is possible to compute the approximated slab concrete Volume and
            # reinforcement steel Weight
            slab_conc_vol = slab_area * Lele  # Concrete Volume of the slab
            slab_steel_weight = self.steel_ratio_slab * slab_area * Lele  # Reinforcement steel Weight of the slab
            # It is possible to compute the approximated longitudinal beams concrete
            # Concrete volume and reinforcement steel weight
            beams_conc_vol = beam_areas * Lele  # Concrete Volume of the longitudinal beams
            beams_steel_weight = self.steel_ratio_beam * beam_areas * Lele  # Reinforcement steel weight of the slab
            # Assumption #2: The slab has a typical average thickness
            # With Assumptions 1 and 2, it is possible to compute the superstructure width
            W_deck = slab_area / self.slab_thickness  # Deck Width
            num_beams = int(W_deck / self.long_beam_spacing) + 1  # Number of superstructure longitudinal beams
            # It is also possible to approximate the Weight of pre-stressing steel
            beams_pre_steel_weight = self.pre_steel_ratio_deck * W_deck * Lele  # Reinforcement steel weight of slab
            # Weight of the railing elements
            rail_weight = 2 * self.railing_weight * Lele
            # Volume of deck asphalt
            asphalt_vol = W_deck * Lele * self.asphalt_thickness

            # Superstructure Concrete Cost
            self.update_concrete(fc)
            super_conc_vol = slab_conc_vol + beams_conc_vol
            super_conc_cost = super_conc_vol * self.concrete
            super_tot_conc_volume += super_conc_vol
            # Superstructure Steel Cost
            self.update_reinforcing_steel(self.d_super)
            super_steel_weight = slab_steel_weight + beams_steel_weight
            super_steel_cost = super_steel_weight * self.reinforcing_steel
            # Superstructure Pre-Stressing Steel Cost
            super_psteel_cost = beams_pre_steel_weight * self.prestressing_steel
            # Superstructure Railing Cost
            super_rail_cost = rail_weight * self.railing
            # Superstructure Asphalt Layer Cost
            super_asphalt_cost = asphalt_vol * self.asphalt
            # Superstructure Formwork Cost
            self.update_formwork(fw_type)
            super_formwork_cost = Lele * W_deck * self.formwork
            # Superstructure Construction Cost
            super_cons_cost += self.ophm_super * (super_conc_cost + super_steel_cost + super_psteel_cost +
                                                  super_rail_cost + super_asphalt_cost + super_formwork_cost)

        total_length = sum(span_lengths)

        # --------------------------------------------------------------
        # Abutment Material Quantification and Construction Cost
        # --------------------------------------------------------------
        aw = (num_beams - 1) * self.long_beam_spacing + 2  # Abutment width
        if aw < W_deck:
            aw = W_deck
        span1_length = span_lengths[0]  # Length first span
        span2_length = span_lengths[-1]  # Length last span
        # Height of back-wall abutment 1
        if span1_length <= 20:
            bwh1 = 1.3
        elif 20 < span1_length <= 30:
            bwh1 = 1.9
        elif span1_length > 30:
            bwh1 = 2.4
        # Height od back-wall abutment 2
        if span2_length <= 20:
            bwh2 = 1.3
        elif 20 < span2_length <= 30:
            bwh2 = 1.9
        elif span2_length > 30:
            bwh2 = 2.4
        # Concrete Volume abutment 1
        Vol1 = bwh1 * aw
        Vol2 = (self.abutment_height - bwh1) * aw
        Vol3 = self.abutment_foundation_width * self.abutment_foundation_thickness * aw
        Vol4 = 2 * self.abutment_wingwall_thickness * self.abutment_wingwall_length * self.abutment_height
        abut1_conc_vol = Vol1 + Vol2 + Vol3 + Vol4  # Concrete Volume abutment 1
        # Concrete Volume abutment 2
        Vol1 = bwh2 * aw
        Vol2 = (self.abutment_height - bwh2) * aw
        Vol3 = self.abutment_foundation_width * self.abutment_foundation_thickness * aw
        Vol4 = 2 * self.abutment_wingwall_thickness * self.abutment_wingwall_length * self.abutment_height
        abut2_conc_vol = Vol1 + Vol2 + Vol3 + Vol4  # Concrete Volume abutment 2
        # Steel weight for abutment 1 and 2
        abut1_steel_weight = self.abutment_steel_ratio * abut1_conc_vol  # Steel weight abutment 1
        abut2_steel_weight = self.abutment_steel_ratio * abut2_conc_vol  # Steel weight abutment 2
        # Excavation Volume for both abutments
        abut_exc_vol = 2 * aw * self.abutment_height * (self.abutment_wingwall_length + 2)
        # Earth fill Volume for both abutments
        abut_fill_vol = 2 * aw * self.abutment_height * (self.abutment_wingwall_length + 2)

        # Abutment Concrete Cost
        self.update_concrete(self.fc_footing)
        abutment_concrete_volume = abut1_conc_vol + abut2_conc_vol
        abutment_concrete_cost = abutment_concrete_volume * self.concrete
        # Abutment Steel Cost
        self.update_reinforcing_steel(self.d_footing)
        abutment_concrete_weight = abut1_steel_weight + abut2_steel_weight
        abutment_steel_cost = abutment_concrete_weight * self.reinforcing_steel
        # Abutment Excavation Cost
        abutment_exc_cost = abut_exc_vol * self.excavation
        # Abutment Fill Grading Cost
        abutment_fill_cost = abut_fill_vol * self.fill
        # Abutment Formwork Cost
        fw_type = 2  # Type of formwork
        self.update_formwork(fw_type)
        abutment_formwork_cost = 2 * self.formwork * (2 * aw * self.abutment_height +
                                                      2 * self.abutment_height * self.abutment_seatwall_thickness +
                                                      4 * self.abutment_height * self.abutment_wingwall_length)
        # Abutment Construction Cost
        abutment_cons_cost = self.ophm_abutment * (abutment_concrete_cost + abutment_steel_cost + abutment_exc_cost +
                                                   abutment_fill_cost + abutment_formwork_cost)

        # --------------------------------------------------------------
        # Pier Material Quantification and Construction Cost
        # --------------------------------------------------------------
        pier_cons_costs = {}  # for each pier
        pier_tot_concrete_volume = 0  # total concrete volume for piers
        fw_type = 2  # Type of formwork
        for i, pier_tag in enumerate(bridge_data.EleIDsPier):
            pier = bridge_data.EleIDsPier[pier_tag]
            section = bridge_data.model[pier['section']]
            if section['Type'] != 'Solid Circular':
                raise ValueError('Damage models for sections other than solid circular are not defined yet..')
            elif section['Type'] == 'Solid Circular':
                # Pier properties
                D_pier = section['D']  # Pier diameter
                H_pier = pier['H']  # Pier height
                fc = section['Fce']  # Unconfined concrete compressive strength
                num_bars = section['number of bars']  # Number of longitudinal bars
                dl = section['dl']  # Diameter of longitudinal steel reinforcement bars
                cover = section['cover']  # Concrete cover
                dh = section['dh']  # Diameter of transverse reinforcement
                sh = section['sh']  # Spacing of transverse reinforcement

                # Pier Concrete Cost
                gross_area = np.pi * D_pier ** 2 / 4  # Total area of concrete section
                concrete_volume = gross_area * H_pier  # Pier concrete volume
                pier_tot_concrete_volume += concrete_volume  # Add to the total concrete volume for piers
                self.update_concrete(fc)
                pier_concrete_cost = concrete_volume * self.concrete

                # Pier Steel Cost
                ds = D_pier - 2 * cover  # Core diameter
                # Assuming 2m for anchorage length (could be inconsistent)
                long_steel_volume = num_bars * (np.pi * dl ** 2) / 4 * (H_pier + 2)
                self.update_reinforcing_steel(dl)  # longitudinal steel
                pier_steel_cost = self.gamma_steel * long_steel_volume * self.reinforcing_steel
                # Assuming 1.5m for hooks and splices
                transv_steel_volume = (np.pi * dh ** 2 / 4 * (np.pi * ds + 1.5)) * round((H_pier + 2) / sh)
                self.update_reinforcing_steel(dh)  # transverse steel
                pier_steel_cost += self.gamma_steel * transv_steel_volume * self.reinforcing_steel

                # Pier Formwork Cost
                pier_formwork_area = H_pier * np.pi * D_pier
                self.update_formwork(fw_type)
                pier_formwork_cost = pier_formwork_area * self.formwork

                # Total Pier Construction Cost
                pier_cons_costs[pier_tag] = self.ophm_pier * (pier_concrete_cost + pier_steel_cost + pier_formwork_cost)
        pier_cons_cost = sum(pier_cons_costs.values())

        # --------------------------------------------------------------
        # Bent-Cap Material Quantification and Construction Cost
        # --------------------------------------------------------------
        bentcap_cons_costs = []  # for each pier
        bentcap_concrete_volumes = []  # total concrete volume of bentcaps
        fw_type = 2  # Type of formwork
        for i, cap_tag in enumerate(bridge_data.model['General']['Bent Caps']):
            L_bc = bridge_data.model[cap_tag]['length']
            h_bc = bridge_data.model[cap_tag]['height']
            b_bc = bridge_data.model[cap_tag]['width']
            fc = (bridge_data.model[cap_tag]['E'] / (5000 * MPa)) ** 2

            # Concrete cost
            cap_concrete_volume = bridge_data.model[cap_tag]['length'] * bridge_data.model[cap_tag]['A']
            bentcap_concrete_volumes.append(cap_concrete_volume)
            self.update_concrete(fc)
            cap_concrete_cost = cap_concrete_volume * self.concrete
            # Reinforcing steel cost
            cap_steel_weight = self.bentcap_steel_ratio * cap_concrete_volume
            self.update_reinforcing_steel(self.d_super)
            cap_steel_cost = cap_steel_weight * self.reinforcing_steel
            # Formwork cost
            cap_area_formwork = 2 * (b_bc * h_bc) + 2 * (L_bc * h_bc) + (b_bc * L_bc)
            self.update_formwork(fw_type)
            cap_formwork_cost = self.formwork * cap_area_formwork
            # Construction cost
            tot_cost = self.ophm_bentcap * (cap_concrete_cost + cap_steel_cost + cap_formwork_cost)
            bentcap_cons_costs.append(tot_cost)
        bentcap_tot_concrete_volume = sum(bentcap_concrete_volumes)
        bentcap_cons_cost = sum(bentcap_cons_costs)

        # --------------------------------------------------------------
        # Bearing Material Quantification and Construction Cost
        # --------------------------------------------------------------
        bearing_cons_cost = 0
        # Loop through each bridge joint
        for joint in bridge_data.EleIDsBearing:
            num_bearings = len(bridge_data.EleIDsBearing[joint])
            bearing_cons_cost += num_bearings * self.ophm_bearing * self.bearing * self.bearing_volume

        # --------------------------------------------------------------
        # Foundation Material Quantification and Construction Cost
        # --------------------------------------------------------------
        num_bents = bridge_data.num_spans - 1
        # Footing materials
        if self.Vs30 >= 760:
            footing_conc_vol = num_bents * (6.5 * 6.5 * 2.5)
            footing_formwork_area = num_bents * ((2 * 6.5 * 6.5) + 4 * (6.5 * 2.5))
        elif 200 <= self.Vs30 < 760:
            footing_conc_vol = num_bents * (7.5 * 7.5 * 2.0)
            footing_formwork_area = num_bents * ((2 * 7.5 * 7.5) + 4 * (6.5 * 2.0))
        elif self.Vs30 < 200:
            footing_conc_vol = num_bents * (8.5 * 8.5 * 1.5)
            footing_formwork_area = num_bents * ((2 * 8.5 * 8.5) + 4 * (8.5 * 1.5))
        # Footing Concrete Cost
        self.update_concrete(self.fc_footing)
        footing_conc_cost = footing_conc_vol * self.concrete
        # Footing Steel Cost
        self.update_reinforcing_steel(self.d_footing)
        footing_steel_weight = self.footing_steel_ratio * footing_conc_vol
        footing_steel_cost = footing_steel_weight * self.reinforcing_steel
        # Footing Formwork Cost
        fw_type = 2  # Type of formwork
        self.update_formwork(fw_type)
        footing_formwork_cost = self.formwork * footing_formwork_area
        # Footing Construction Cost
        footing_cons_cost = self.ophm_footing * (footing_conc_cost + footing_steel_cost + footing_formwork_cost)

        # Pile related
        if self.Vs30 >= 760:
            pile_length = num_bents * (4 * 10)
            D_pile = 1.50
        elif 200 <= self.Vs30 < 760:
            pile_length = num_bents * (9 * 15)
            D_pile = 1.0
        elif self.Vs30 < 200:
            pile_length = num_bents * (16 * 20)
            D_pile = 0.80
        pile_volume = 0.25 * np.pi * (D_pile ** 2) * pile_length
        # Pile construction cost
        self.update_pile(D_pile)
        pile_cons_cost = self.ophm_pile * (self.pile * pile_length)

        # --------------------------------------------------------------
        # Total Bridge Construction Cost (EUR)
        # --------------------------------------------------------------
        bridge_cons_cost = super_cons_cost + abutment_cons_cost + pier_cons_cost + bentcap_cons_cost + \
                           bearing_cons_cost + footing_cons_cost + pile_cons_cost
        # Unitary Bridge Construction Cost; this is for comparison with typical values in the literature (EUR/m2)
        bridge_cons_unit_cost = bridge_cons_cost / (total_length * W_deck)

        # --------------------------------------------------------------
        # Bridge Demolition Cost (EUR)
        # --------------------------------------------------------------
        # Superstructure
        demolition_type = 1
        self.update_demolition(demolition_type)
        super_demol_cost = self.demolition * super_tot_conc_volume
        # Piers, abutment and footing
        demolition_type = 2
        self.update_demolition(demolition_type)
        sub_demol_cost = self.demolition * (abutment_concrete_volume + abut_fill_vol + pier_tot_concrete_volume +
                                            bentcap_tot_concrete_volume + footing_conc_vol)
        # Piles
        demolition_type = 3
        self.update_demolition(demolition_type)
        pile_demol_cost = self.demolition * pile_volume
        # Bridge Demolition Cost
        bridge_demol_cost = super_demol_cost + sub_demol_cost + pile_demol_cost

        # --------------------------------------------------------------
        # Bridge Replacement Cost (EUR)
        # --------------------------------------------------------------
        bridge_rep_cost = bridge_cons_cost + bridge_demol_cost  # (EUR)
        bridge_rep_unit_cost = bridge_rep_cost / (total_length * W_deck)  # Unitary Bridge Replacement Cost (EUR/m2)

        # --------------------------------------------------------------
        # Mean Costs for Repair Actions of Abutments (EUR)
        # --------------------------------------------------------------
        abut_mean_repair_cost = np.zeros([2, 3])
        for i in range(2):
            for j in range(3):

                if j == 0:  # Abutment Repair Cost for DLS-1
                    # Repair Action: Cleaning and Superficial Treatment
                    # Area of cleaning cost and superficial treatment
                    cleaning_area = 2 * self.abutment_wingwall_length * self.abutment_height + self.abutment_height * aw
                    # Total cost for cleaning cost and superficial treatment
                    abut_mean_repair_cost[i, j] = self.ophm_dls1_abutment * cleaning_area * self.cleaning

                elif j == 1:  # Abutment Repair Cost for DLS-2
                    # Repair Action: Cleaning and Superficial Treatment
                    cleaning_area = 2 * self.abutment_wingwall_length * self.abutment_height + self.abutment_height * aw
                    cleaning_cost = cleaning_area * self.cleaning

                    # Repair Action: Concrete patch in affected wing-walls
                    co = 0.05  # Assumed concrete cover thickness (m)
                    patch_volume = 2 * co * self.abutment_wingwall_length * self.abutment_height
                    patch_cost = self.concrete_patch * patch_volume

                    # Repair Action: Formwork for concrete patch
                    fw_type = 3
                    self.update_formwork(fw_type)
                    patch_formwork_area = 2 * self.abutment_wingwall_length * self.abutment_height
                    formwork_cost = patch_formwork_area * self.concrete_patch

                    # Repair Action: Backfill excavation in the height of the backwall
                    if i == 0:
                        bwh = 0.5 * bwh1
                    elif i == 1:
                        bwh = 0.5 * bwh2

                    exc_vol = bwh * aw * (self.abutment_wingwall_length + 2)  # Excavation volume
                    exc_cost = self.excavation * exc_vol  # Total Cost of Excavation

                    # Repair Action: Crack sealing on the backfill side
                    crack_spacing = 0.4  # Assumed average spacing of cracks to seal
                    num_crack_seal = 2 * (round(bwh / crack_spacing) + 1)
                    crack_seal_cost = self.crack_seal * 0.5 * self.abutment_wingwall_length * num_crack_seal

                    # Repair Action: Backfill replacement
                    fill_cost = 0.25 * self.fill * exc_vol  # Total Cost of fill replacement

                    # Repair Action: Asphalt Replacement
                    asphalt_vol = W_deck * self.asphalt_thickness * (self.abutment_wingwall_length + 2)
                    asphalt_cost = self.asphalt * asphalt_vol

                    # Total Repair Cost
                    abut_mean_repair_cost[i, j] = self.ophm_dls2_abutment * (
                            cleaning_cost + patch_cost + formwork_cost +
                            exc_cost + crack_seal_cost + fill_cost + asphalt_cost)

                elif j == 2:  # Abutment Repair Cost for DLS-3

                    # Repair Action: Temporal Support of Superstructure
                    fw_type = 4
                    self.update_formwork(fw_type)  # Unitary cost of temporal support
                    if i == 0:  # Length to be supported ≈ 15%
                        temporal_support_length = 0.15 * span_lengths[0]
                    elif i == 1:
                        temporal_support_length = 0.15 * span_lengths[-1]
                    temporal_support_area = W_deck * temporal_support_length  # Area of temporal support
                    temporal_support_cost = self.formwork * temporal_support_area  # Total Cost of temporal support

                    # Repair Action: Backfill Excavation
                    exc_vol = W_deck * self.abutment_height * (self.abutment_wingwall_length + 2)  # Excavation volume
                    exc_cost = self.excavation * exc_vol

                    # Repair Action: Abutment Demolition
                    demol_type = 2
                    self.update_demolition(demol_type)
                    if i == 0:  # Volume to be demolished
                        demolition_vol = abut1_conc_vol
                    elif i == 1:
                        demolition_vol = abut2_conc_vol
                    demol_cost = self.demolition * demolition_vol  # Total cost of abutment demolition

                    # Repair Action: Abutment structure replacement
                    self.update_concrete(self.fc_footing)  # Unitary Cost of Concrete
                    self.update_reinforcing_steel(self.d_footing)  # Unitary Cost of Steel
                    if i == 0:  # Steel Weight to be constructed
                        steel_weight = abut1_steel_weight
                    elif i == 1:
                        steel_weight = abut1_steel_weight
                    # Total cost of abutment construction
                    cons_cost = self.concrete * demolition_vol + self.reinforcing_steel * steel_weight

                    # Repair Action: Abutment Backfill regrading
                    fill_vol = W_deck * self.abutment_height * (self.abutment_wingwall_length + 2)
                    fill_cost = self.fill * fill_vol  # Total cost of abutment regrading

                    # Repair Action: Asphalt replacing
                    asphalt_vol = W_deck * self.asphalt_thickness * (self.abutment_wingwall_length + 2)
                    asphalt_cost = self.asphalt * asphalt_vol

                    # Total Repair Cost
                    abut_mean_repair_cost[i, j] = self.ophm_dls3_abutment * (temporal_support_cost + exc_cost +
                                                                             demol_cost + cons_cost + fill_cost +
                                                                             asphalt_cost)

        # --------------------------------------------------------------
        # Mean Costs for Repair Actions of Bents/Piers (EUR)
        # --------------------------------------------------------------
        # Note: Even if a single pier is damaged in a bent, both will be repaired.
        bent_mean_repair_cost = np.zeros([len(bridge_data.model['General']['Bents']), 3])

        for i, bent_tag in enumerate(bridge_data.model['General']['Bents']):
            num_piers = len(bridge_data.model[bent_tag]['sections'])
            for k in range(num_piers):
                pier_tag = 'B' + str(i + 1) + '-P' + str(k + 1)  # pier tag
                pier = bridge_data.EleIDsPier[pier_tag]  # pier properties
                section = bridge_data.model[pier['section']]  # section properties
                if section['Type'] != 'Solid Circular':
                    raise ValueError('Damage models for sections other than solid circular are not defined yet..')
                elif section['Type'] == 'Solid Circular':
                    # Pier properties
                    D_pier = section['D']  # Pier diameter
                    H_pier = pier['H']  # Pier height
                    cover = section['cover']  # Concrete cover

                    for j in range(3):  # DLS besides the collapse

                        if j == 0:  # Pier Repair Cost for DLS-1

                            # Repair Action: Sealing of Cracks
                            unit_crack_length = 0.5 * np.pi * D_pier  # Unitary crack length to be sealed
                            crack_height = 0.8 * D_pier  # Length over which cracks to be sealed develop
                            crack_spacing = 0.25 * D_pier  # Average spacing of cracks to be sealed
                            num_crack_seal = round(crack_height / crack_spacing) + 1  # number of cracks
                            # Total length of cracks to be sealed along the pier
                            tot_crack_length = unit_crack_length * num_crack_seal
                            # Total cost of crack sealing
                            bent_mean_repair_cost[i, j] += self.ophm_dls1_pier * self.crack_seal * tot_crack_length

                        elif j == 1:  # Column Repair Cost for DLS-2
                            # Repair Action: Demolition of cover concrete
                            # Demolition area - unconfined area
                            demol_area = (0.25 * np.pi * D_pier ** 2) - (0.25 * np.pi * (D_pier - 2 * cover) ** 2)
                            if H_pier > 3 * D_pier:  # Smaller length for slender pier
                                L_pier = 3 * D_pier
                            else:  # Full height for a squat pier
                                L_pier = H_pier
                            demol_vol = demol_area * L_pier  # Demolition volume
                            cover_demol_cost = demol_vol * self.cover_demolition

                            # Repair Action: Cleaning and Surface Preparing
                            cleaning_area = np.pi * D_pier * H_pier  # Surface area of cleaning and preparing
                            cleaning_cost = self.cleaning * cleaning_area

                            # Repair Action: Concrete Patching
                            patch_cost = self.concrete_patch * demol_vol  # Total Cost of Concrete Patching

                            # Repair Action: Formwork for patching
                            fw_type = 3
                            self.update_formwork(fw_type)  # Unitary cost for formwork
                            patch_formwork_area = np.pi * D_pier * L_pier  # Area of formwork for concrete patch
                            formwork_cost = self.formwork * patch_formwork_area

                            # Total cost of concrete patching
                            bent_mean_repair_cost[i, j] += self.ophm_dls2_pier * (cover_demol_cost + cleaning_cost +
                                                                                  patch_cost + formwork_cost)

                        elif j == 2:  # Column Repair Cost for DLS-3
                            # in practice, I believe all the bent will be replaced, not the single pier.
                            # For this purpose, lets divide the total cost of bentcap and bearing replacements
                            # by number of piers

                            # Repair Action: Demolition of Existing Damaged Column
                            demol_type = 3
                            self.update_demolition(demol_type)
                            demol_pier_vol = 0.25 * np.pi * (D_pier ** 2) * H_pier + (
                                    bentcap_concrete_volumes[i] / num_piers)
                            demol_pier_cost = self.demolition * demol_pier_vol

                            # Repair Action: Intervention in the Foundation
                            if D_pile >= 1.5:
                                hf = 2.5
                            elif 0.8 < D_pile < 1.5:
                                hf = 2
                            elif D_pile <= 0.8:
                                hf = 1.5
                            demol_found_vol = 0.25 * np.pi * (D_pile ** 2) * 0.8 * hf  # Embedded volume of pier
                            demol_found_cost = self.cover_demolition * demol_found_vol

                            # Repair Actions: Columns construction and required formwork for reconstruction
                            cons_formwork_cost = pier_cons_costs[pier_tag] / self.ophm_pier

                            # Total cost of pier replacement
                            bent_mean_repair_cost[i, j] += self.ophm_dls3_pier * (demol_pier_cost + demol_found_cost +
                                                                                  cons_formwork_cost)

            # Repair Action: Temporal Superstructure Support at DLS-3
            fw_type = 4
            self.update_formwork(fw_type)  # Unitary cost of temporal support
            temporal_support_length = 0.15 * 0.5 * (span_lengths[i] + span_lengths[i + 1])  # assume 15%
            temporal_support_area = W_deck * temporal_support_length  # Area of temporal support
            temporal_support_cost = self.formwork * temporal_support_area
            # Add the bentcap and temporal_support cost to the bearing
            bent_mean_repair_cost[i, 2] += self.ophm_dls3_pier * (
                    temporal_support_cost + bentcap_cons_costs[i] / self.ophm_bentcap)

        # --------------------------------------------------------------
        # Mean Costs for Repair Actions of Bearing Performance Group (EUR)
        # --------------------------------------------------------------
        temp_support_costs = []
        # Note: Even if a single bearing is damaged in a joint, all will be repaired.
        bearing_mean_repair_cost = np.zeros([len(bridge_data.EleIDsBearing), 3])
        bearing_cost = self.bearing * self.bearing_volume
        # Loop through each bridge joint
        for joint in bridge_data.EleIDsBearing.keys():
            num_bearings = len(bridge_data.EleIDsBearing[joint])
            # Repair Action: Sealing of Cracks
            unit_crack_length = 0.5 * b_bc  # Unitary crack length to be sealed
            crack_height = 0.8 * h_bc  # Length over which cracks to be sealed develop
            crack_spacing = 1.0  # Average spacing of cracks to be sealed
            num_crack_seal = round(crack_height / crack_spacing) + 1  # number of cracks
            # Total length of cracks to be sealed along the pier
            tot_crack_length = unit_crack_length * num_crack_seal
            crack_seal_cost = num_bearings / 2 * self.crack_seal * tot_crack_length
            # Repair Action: Temporal Superstructure Support at DLS-3 and DLS-2
            if i == 0:
                temporal_support_length = 0.15 * span_lengths[0] * self.ophm_dls3_abutment
            elif i == len(bridge_data.EleIDsBearing.keys()) - 1:
                temporal_support_length = 0.15 * span_lengths[-1] * self.ophm_dls3_abutment
            else:
                temporal_support_length = 0.15 * 0.5 * (span_lengths[i] + span_lengths[i - 1]) * self.ophm_dls3_pier
            fw_type = 4
            self.update_formwork(fw_type)  # Unitary cost of temporal support
            temporal_support_area = W_deck * temporal_support_length  # Area of temporal support
            temporal_support_cost = self.formwork * temporal_support_area
            temp_support_costs.append(temporal_support_cost)
            # substract this later from bearings to avoid double counting

            # Bearing Repair Cost at DLS-1
            # Repair Action: clean concrete surface of shear keys and seal the cracks
            bearing_mean_repair_cost[joint, 0] += self.ophm_dls1_bearing * crack_seal_cost
            # Bearing Repair Cost at DLS-2
            # Repair Action: clean concrete surface of shear keys and seal the cracks (1), align the deck (2), inspect bearings (3)
            # Assume 20% of total bearing cost for the inspection of bearings
            bearing_mean_repair_cost[joint, 1] += self.ophm_dls2_bearing * (
                    crack_seal_cost + 0.2 * bearing_cost * num_bearings + self.deck_alignment)
            bearing_mean_repair_cost[joint, 1] += temporal_support_cost
            # Bearing Repair Cost at DLS-3
            # Repair Action: clean concrete surface of shear keys and seal the cracks (1), lift up the girders with hydraulic jacks (2), align the deck (3) and replace the bearings (4)
            bearing_mean_repair_cost[joint, 2] += self.ophm_dls3_bearing * (
                    crack_seal_cost + bearing_cost * num_bearings + self.deck_alignment + self.jacking)
            bearing_mean_repair_cost[joint, 2] += temporal_support_cost

        return bridge_rep_cost, bridge_rep_unit_cost, bridge_cons_cost, bridge_cons_unit_cost, \
               bent_mean_repair_cost, abut_mean_repair_cost, bearing_mean_repair_cost, temp_support_costs


class msa:
    """
    Module perform economic loss estimation (direct+indirect) for reinforced concrete bridges using multiple-stripe analysis
    results. A procedure equivalent or similar to the PEER-PBEE framework is implemented.
    Time-based assessment is performed.

    Author: Volkan Ozsarac
    First Version: Oct 2021
    """

    def __init__(self, input_folder='Inputs', output_folder='Outputs', tw=50, n_interp=1e4,
                 pier_edp_type=3, beta_c=0.25, beta_q=0.25, Vs30=400, num_crews=1):
        """

        Details
        -------
        Initializes the object.
        Reads the input data and creates output directory for each bridge being processed.

        Notes
        -----
        None.

        References
        ----------
        Volkan Ozsarac, Ricardo Monteiro & Gian Michele Calvi (2021) Probabilistic seismic
        assessment of reinforced concrete bridges using simulated records, Structure and Infrastructure Engineering,
        DOI: 10.1080/15732479.2021.1956551

        Parameters
        ----------
        input_folder: str
            Input directory where all of required inputs are stored, e.g. 'Inputs'
        output_folder: str
            Output directory where all the results are saved, e.g. 'Outputs'
        tw: int
            Time window for which the structural demands where computed, e.g. 50 yrs
        n_interp: int
            Number of interpolation points to fit hazard curve, e.g. 1e5
        pier_edp_type: int
            Engineering Design Parameter used for pier damage model:
            pier_edp_type=1 for curvature ductility
            pier_edp_type=2 for displacement ductility
            pier_edp_type=3 for drift ratio
        beta_c : float, optional (The default is 0.25)
            Value of dispersion for construction quality
        beta_q : float, optional (The default is 0.25)
            Value of dispersion for analytical model completeness

        Returns
        -------
        None.

        """
        # TODO: Explain the necessary input files to run the complete framework

        # Set plotting properties
        matplotlib.use("Agg")
        SMALL_SIZE = 12
        MEDIUM_SIZE = 13
        BIG_SIZE = 15
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # Turn interactive plotting off
        plt.ioff()

        # Initialize object parameters
        self.pier_edp_type = pier_edp_type  # Adopted pier edp type
        self.beta_c = beta_c  # Value of dispersion for construction quality
        self.beta_q = beta_q  # Value of dispersion for analytical model completeness
        self.output_folder = output_folder  # output folder
        self.tw = tw  # Time window for which the structural demands where computed, e.g. 50 yrs
        self.n_interp = n_interp  # Number of interpolation points to fit hazard curve, e.g. 1e5
        self.loss_simplified = {}  # Expected losses computed using HAZUS approach
        self.loss_comprehensive = {}  # Expected losses computed using FEMA P-58 approach
        self.frag_comp_param = None  # Component level fragility curve parameters
        self.frag_sys = None  # Adopted system level fragility curves
        self.frag_sys_param = None  # Fragility curve parameters for the adopted system level fragility curves
        self.frag_upper = None  # Upper bound for system level fragility curves
        self.frag_lower = None  # Lower bound for system level fragility curves
        self.input = None  # dictionary containing all input information necessary to compute direct losses
        self.rep_cost = None  # Bridge replacement costs
        self.urep_cost = None  # Unit bridge replacement costs
        self.cons_cost = None  # Bridge construction costs
        self.ucons_cost = None  # Unit bridge replacement costs
        self.pier_cost = None  # Mean pier repair costs
        self.abut_cost = None  # Mean abutment repair costs
        self.bearing_cost = None  # Mean bearing repair costs
        self.pier_dmdls1 = None  # Pier damage models at DLS-1
        self.pier_dmdls2 = None  # Pier damage models at DLS-2
        self.pier_dmdls3 = None  # Pier damage models at DLS-3
        self.pier_dmdls4 = None  # Pier damage models at DLS-4 (Flexure)
        self.pier_dmdls4S = None  # Pier damage models at DLS-4S (Shear)
        self.abut_dmdls1 = None  # Abutment damage models at DLS-1
        self.abut_dmdls2 = None  # Abutment damage models at DLS-2
        self.abut_dmdls3 = None  # Abutment damage models at DLS-3
        self.abut_dmdls4 = None  # Abutment damage models at DLS-4
        self.bearing_dmdls1 = None  # Bearing damage models at DLS-1
        self.bearing_dmdls2 = None  # Bearing damage models at DLS-2
        self.bearing_dmdls3 = None  # Bearing damage models at DLS-3
        self.bearing_dmdls4 = None  # Bearing damage models at DLS-4
        self.iml_fit = None  # Intensity measure values for the fitted hazard curve
        self.lambda_fit = None  # Mean annual frequency of exceedance values for the fitted hazard curve
        self.poe_fit = None  # Mean probability of exceedance values in 50 years for the fitted hazard curve
        self.repair_duration = {}  # The repair duration associated with repair of each component is here along with resulted functionality values
        self.num_crews = num_crews # The number of available crews which can work simultaneously on the repair of different performance groups

        # Hazard Curve
        hazard_data = os.path.join('Hazard', 'hazard_curve.txt')
        # Intensity measure levels
        iml_data = os.path.join('Hazard', 'imls.txt')
        # Probability of exceedance values given investigation time
        poe_data = os.path.join('Hazard', 'poes.txt')
        # Engineering demand parameters
        edp_data = 'edps.pkl'
        # Bridge data
        bridge_data = 'bridge_data.pkl'
        # Network analysis results
        network_data = os.path.join('Network', 'daily_indirect_loss.txt')
        # Get the folder where duration is stored
        duration_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repair durations')
        # Read the duration data
        self.repair_duration['abutment'] = np.loadtxt(os.path.join(duration_data, 'Abutment.txt'))
        self.repair_duration['bearing'] = np.loadtxt(os.path.join(duration_data, 'Bearing.txt'))
        self.repair_duration['bent'] = np.loadtxt(os.path.join(duration_data, 'Bent.txt'))
        self.repair_duration['collapse'] = np.loadtxt(os.path.join(duration_data, 'Collapse.txt'))
        self.repair_duration['planning'] = np.loadtxt(os.path.join(duration_data, 'Planning.txt'))

        # Initialize cost calculations
        bridge_cost = _cost(Vs30)
        # Read get folder names
        msa_path = os.path.join(input_folder, 'MSA')
        # Read the input for bridge mode
        with open(os.path.join(input_folder, bridge_data), "rb") as file:
            self.input = pickle.load(file)
        # Read the hazard input
        self.input.hazard_curve = np.loadtxt(os.path.join(input_folder, hazard_data))
        imls = np.loadtxt(os.path.join(input_folder, iml_data))
        poes = np.loadtxt(os.path.join(input_folder, poe_data))
        self.input.msa_imls = np.array([imls, poes]).T
        self.input.msa_edps = {}
        # Get the network analysis results
        self.daily_indirect_loss = np.loadtxt(os.path.join(input_folder, network_data))

        for poe in poes:
            poe_fol = 'POE-' + str(poe) + '-in-' + str(tw) + '-years'
            edp_path = os.path.join(msa_path, poe_fol, edp_data)
            with open(edp_path, "rb") as file:
                self.input.msa_edps[poe] = pickle.load(file)

        # Fit an hazard curve to perform more accurate calculations
        iml_hazard = self.input.hazard_curve[:, 0]  # IM values, e.g. SA(T1), for known Mean PoE Values
        poe_hazard = self.input.hazard_curve[:, 1]  # Mean PoE of IM=im in tw years, e.g. 10%, 5%, 2%
        mask = (poe_hazard != 1) * (poe_hazard != 0)
        poe_hazard = poe_hazard[mask]
        iml_hazard = iml_hazard[mask]  # Do not include if poe=1 since it cannot be converted
        # Intensity Measure Range for which structure level and collapse fragility curves are constructed
        self.iml_fit = np.arange(min(iml_hazard), max(iml_hazard),
                                 (max(iml_hazard) - min(iml_hazard)) / n_interp)
        # Fit the hazard curve
        self.lambda_fit, self.poe_fit = _hazard_fit(self.iml_fit, poe_hazard, iml_hazard, tw)

        # TODO: reorganize data, and assign performance groups
        self.pg_input = {'bent': {}, 'abutment': {}, 'bearing': {}}
        self.pg_cost = {'bent': {}, 'abutment': {}, 'bearing': {}}
        self.pg_edp = {'bent': {}, 'abutment': {}, 'bearing': {}}

        for i in range(1, self.input.num_spans):
            pier_ids = [key for key in self.input.EleIDsPier.keys() if key.startswith('B' + str(i))]
            self.pg_input['bent']['B' + str(i)] = {pier.split('-')[1]: self.input.EleIDsPier[pier] for pier in pier_ids}

        self.pg_input['bearing'] = self.input.EleIDsBearing
        self.pg_input['abutment'] = self.input.EleIDsAbut

        # Compute costs
        self.rep_cost, self.urep_cost, self.cons_cost, self.ucons_cost, \
        self.pg_cost['bent'], self.pg_cost['abutment'], self.pg_cost['bearing'], \
        self.temporal_support_costs = bridge_cost.calculate(self.input)

        # Create output folder
        create_dir(self.output_folder)

    def get_damage_models(self):
        """
        Details
        -------
        Retrieves component damage models

        """

        bent_tags = list(set(self.input.model['General']['Bents']))
        sec_tags = []
        for bent_tag in bent_tags:
            sec_tags.extend(self.input.model[bent_tag]['sections'])
        sec_tags = list(set(sec_tags))

        pier_data = self.pg_input['bent'].copy()
        sections_data = {sec_tag: self.input.model[sec_tag] for sec_tag in sec_tags}
        bearing_data = self.input.EleIDsBearing

        self.pier_dmdls1, self.pier_dmdls2, self.pier_dmdls3, self.pier_dmdls4, self.pier_dmdls4S = \
            _get_pier_dm_dls(sections_data, pier_data, self.pier_edp_type)
        self.abut_dmdls1, self.abut_dmdls2, self.abut_dmdls3, self.abut_dmdls4 = _get_abut_dm_dls()
        self.bearing_dmdls1, self.bearing_dmdls2, self.bearing_dmdls3, self.bearing_dmdls4 = \
            _get_bearing_dm_dls(bearing_data)

    def plot_damage_models(self):

        """
        Details
        -------
        Plots damage model per damage limit state for piers and abutments.
        Must have run get_damage_models method before using the method.

        """

        # Plot Damage Models of the Piers
        if self.pier_edp_type == 1:
            xlabel_name = r'Curvature Ductility - $\mu_{\phi}$'
        elif self.pier_edp_type == 2:
            xlabel_name = r'Displacement Ductility - $\mu_{\Delta}$'
        elif self.pier_edp_type == 3:
            xlabel_name = r'Pier Drift Ratio - $\Delta$/H'

        for bent in self.pier_dmdls1:  # loop for each bent
            for i in range(self.pier_dmdls1[bent].shape[0]):
                dmax = 2 * self.pier_dmdls4[bent][i, 0]
                dplot = np.arange(1e-5, dmax, dmax / 100)
                plt.figure()
                plt.plot(dplot, norm.cdf((np.log(dplot / self.pier_dmdls1[bent][i, 0])) / self.pier_dmdls1[bent][i, 1]),
                         color='green', lw=1.5, label='DLS-1')
                plt.plot(dplot, norm.cdf((np.log(dplot / self.pier_dmdls2[bent][i, 0])) / self.pier_dmdls2[bent][i, 1]),
                         color='orange', lw=1.5, label='DLS-2')
                plt.plot(dplot, norm.cdf((np.log(dplot / self.pier_dmdls3[bent][i, 0])) / self.pier_dmdls3[bent][i, 1]),
                         color='red', lw=1.5, label='DLS-3')
                plt.plot(dplot, norm.cdf((np.log(dplot / self.pier_dmdls4[bent][i, 0])) / self.pier_dmdls4[bent][i, 1]),
                         color='black', lw=1.5, label='DLS-4')

                if self.pier_dmdls4S is not None:
                    if self.pier_dmdls4S[bent][i, 0] == 0:
                        plt.plot(dplot, np.zeros([len(dplot), 1]), color='black', lw=1, ls='--', label='DLS-4-S')
                    else:
                        plt.plot(dplot, norm.cdf(
                            (np.log(dplot / self.pier_dmdls4S[bent][i, 0])) / self.pier_dmdls4S[bent][i, 1]),
                                 color='black', lw=1, ls='--', label='DLS-4-S')

                plt.title(bent + '-P' + str(i + 1) + ' Damage Models')
                plt.xlabel(xlabel_name)
                plt.ylabel('Probability of Exceedance')
                plt.legend(frameon=False)
                plt.grid(True)
                plt.xlim([0, dmax])
                plt.ylim([0, 1])

                # Save
                fname = os.path.join(self.output_folder, 'DM_' + bent + '-P' + str(i + 1) + '.png')
                plt.savefig(fname)
                plt.close('all')

        # Plot Damage Models of the Abutments (same for both)
        dmax = self.abut_dmdls3[0, 0] * 3
        dplot = np.arange(1e-5, dmax, dmax / 100)
        plt.figure()
        plt.plot(dplot, norm.cdf((np.log(dplot / self.abut_dmdls1[0, 0])) / self.abut_dmdls1[0, 1]),
                 color='green', lw=1.5, label='DLS-1')
        plt.plot(dplot, norm.cdf((np.log(dplot / self.abut_dmdls2[0, 0])) / self.abut_dmdls2[0, 1]),
                 color='orange', lw=1.5, label='DLS-2')
        plt.plot(dplot, norm.cdf((np.log(dplot / self.abut_dmdls3[0, 0])) / self.abut_dmdls3[0, 1]),
                 color='red', lw=1.5, label='DLS-3')
        # plt.plot(dplot, norm.cdf((np.log(dplot / abut_dm[3, 0])) / abut_dm[3, 1]),
        #          color='black', lw=1.5, label='DLS-4')
        plt.title('Abutment Damage Models')
        plt.xlabel(r'Abutment Displacement - $\Delta$ [m]')
        plt.ylabel('Probability of Exceedance')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.xlim([0, dmax])
        plt.ylim([0, 1])
        # Save directory for the results
        fname = os.path.join(self.output_folder, "DM_Abut.png")
        plt.savefig(fname)
        plt.close('all')

        # Plot Damage Models of the Bearings (same at joints)
        for joint in self.bearing_dmdls1:
            dmax = self.bearing_dmdls4[joint][0, 0] * 2
            dplot = np.arange(1e-5, dmax, dmax / 100)
            plt.figure()
            plt.plot(dplot, norm.cdf(
                (np.log(dplot / self.bearing_dmdls1[joint][0, 0])) / self.bearing_dmdls1[joint][0, 1]),
                     color='green', lw=1.5, label='DLS-1')
            plt.plot(dplot, norm.cdf(
                (np.log(dplot / self.bearing_dmdls2[joint][0, 0])) / self.bearing_dmdls2[joint][0, 1]),
                     color='orange', lw=1.5, label='DLS-2')
            plt.plot(dplot, norm.cdf(
                (np.log(dplot / self.bearing_dmdls3[joint][0, 0])) / self.bearing_dmdls4[joint][0, 1]),
                     color='red', lw=1.5, label='DLS-3')
            plt.plot(dplot, norm.cdf(
                (np.log(dplot / self.bearing_dmdls4[joint][0, 0])) / self.bearing_dmdls4[joint][0, 1]),
                     color='black', lw=1.5, label='DLS-4')
            plt.title('Bearing Damage Models')
            plt.xlabel(r'Bearing Displacement - $\Delta$ [m]')
            plt.ylabel('Probability of Exceedance')
            plt.legend(frameon=False)
            plt.grid(True)
            plt.xlim([0, dmax])
            plt.ylim([0, 1])
            # Save directory for the results
            fname = os.path.join(self.output_folder, "DM_Bearing_joint_" + str(joint) + ".png")
            plt.savefig(fname)
            plt.close('all')

    def get_fragility(self):
        """
        Details
        -------
        Generates component and system level fragility curves

        """

        # Uncertainty parameters
        beta_mod = np.sqrt(self.beta_c ** 2 + self.beta_q ** 2)  # Modelling Uncertainty

        # Number of components
        joints = list(self.input.EleIDsBearing.keys())
        bearing_per_joint = [len(self.input.EleIDsBearing[joint]) for joint in self.input.EleIDsBearing]
        num_bearings = sum(bearing_per_joint)
        num_piers = len(self.input.EleIDsPier)
        num_abutments = 2

        msa_imls = self.input.msa_imls[:, 0]
        msa_poes = self.input.msa_imls[:, 1]
        edp_pier = {}
        edp_abut = {}
        edp_bearing = {}
        for poe in msa_poes:
            if self.pier_edp_type == 1:
                edp_pier[poe] = self.input.msa_edps[poe]['mu_curv']
            elif self.pier_edp_type == 2:
                edp_pier[poe] = self.input.msa_edps[poe]['mu_disp']
            elif self.pier_edp_type == 3:
                edp_pier[poe] = self.input.msa_edps[poe]['drift_ratio']
            # TODO: Instead of retrieving SRSS of displacements, retrieve distinct edps for these components
            edp_abut[poe] = self.input.msa_edps[poe]['abut_disp']
            edp_bearing[poe] = self.input.msa_edps[poe]['bearing_disp']

        # initialize some parameters
        frag_pier = {}  # contains fragility curves of each pier
        pier_coeff = {}  # contains fragility curve parameters for piers
        frag_abut = {}  # contains fragility curves for abutments
        abut_coeff = {}  # contains fragility curve parameters for abutments
        frag_bearing = {}  # contains fragility curves for bearings
        bearing_coeff = {}  # contains fragility curve parameters for bearings
        self.frag_upper = np.zeros([len(self.iml_fit), 5])  # Upper bound fragility curve
        self.frag_upper[:, 0] = self.iml_fit  # Add imls to the first column
        self.frag_lower = self.frag_upper.copy()  # Lower bound fragility curve
        self.frag_sys = self.frag_upper.copy()  # Adopted fragility curve
        self.frag_sys_param = np.zeros([4, 2])  # Fragility curve parameters for each DLS

        for dls_i in range(0, 4):  # computation per dls

            # Component Type: Pier
            frag_pier['DLS-' + str(dls_i + 1)] = np.zeros([len(self.iml_fit), num_piers])
            pier_coeff['DLS-' + str(dls_i + 1)] = np.zeros([2, num_piers])
            tmp = eval('self.pier_dmdls' + str(dls_i + 1))
            for bent_i, bent in enumerate(tmp):
                if bent_i == 0:
                    pier_dls = tmp[bent]
                else:
                    pier_dls = np.append(pier_dls, tmp[bent], axis=0)

            for p_i in range(num_piers):  # Computation per pier
                num_collapse = np.zeros([len(msa_poes)])
                num_gms = []
                threshold, beta_cap = pier_dls[p_i]

                for poe_i in range(len(msa_poes)):  # Computation per poe
                    # Count exceedance of the DLS for the pier responses
                    poe = msa_poes[poe_i]
                    num_gms.append(len(edp_pier[poe][:, p_i]))
                    num_collapse[poe_i] = np.sum(edp_pier[poe][:, p_i] >= threshold)

                # Estimation of fragility curve parameter for the element
                theta, beta_rec = mle_fit(msa_imls, num_gms, num_collapse)
                if max(num_collapse) == 0 or sum(num_collapse) == 1:
                    theta = 1e15
                    beta = 0.1
                else:
                    # Total Standard Deviation
                    beta = np.sqrt(beta_mod ** 2 + beta_cap ** 2 + beta_rec ** 2)
                p_occur = norm.cdf((np.log(self.iml_fit / theta)) / beta)
                frag_pier['DLS-' + str(dls_i + 1)][:, p_i] = p_occur
                pier_coeff['DLS-' + str(dls_i + 1)][0, p_i] = theta
                pier_coeff['DLS-' + str(dls_i + 1)][1, p_i] = beta

            if dls_i == 3 and self.pier_dmdls4S is not None:  # include the shear failure for pier
                frag_pier['DLS-4S'] = np.zeros([len(self.iml_fit), num_piers])
                pier_coeff['DLS-4S'] = np.zeros([2, num_piers])
                for bent_i, bent in enumerate(self.pier_dmdls4S):
                    if bent_i == 0:
                        pier_dls = self.pier_dmdls4S[bent]
                    else:
                        pier_dls = np.append(pier_dls, self.pier_dmdls4S[bent], axis=0)

                for p_i in range(num_piers):  # Computation per pier
                    num_collapse = np.zeros([len(msa_poes)])
                    num_gms = []
                    threshold, beta_cap = pier_dls[p_i]

                    for poe_i in range(len(msa_poes)):  # Computation per poe
                        # Count exceedance of the DLS for the pier responses
                        poe = msa_poes[poe_i]
                        num_gms.append(len(edp_pier[poe][:, p_i]))
                        if threshold != 0:
                            num_collapse[poe_i] = np.sum(edp_pier[poe][:, p_i] >= threshold)

                    # Estimation of fragility curve parameter for the element
                    theta, beta_rec = mle_fit(msa_imls, num_gms, num_collapse)
                    if max(num_collapse) == 0 or sum(num_collapse) == 1:
                        theta = 1e15
                        beta = 0.1
                    else:
                        # Total Standard Deviation
                        beta = np.sqrt(beta_mod ** 2 + beta_cap ** 2 + beta_rec ** 2)
                    p_occur = norm.cdf((np.log(self.iml_fit / theta)) / beta)
                    frag_pier['DLS-4S'][:, p_i] = p_occur
                    pier_coeff['DLS-4S'][0, p_i] = theta
                    pier_coeff['DLS-4S'][1, p_i] = beta

            # Component Type: Abutment
            # TODO: Modify this part to consider failure mechanisms separately
            # 1-) Abutment passive displacement
            # 2-) Abutment activate displacement
            # 3-) Abutment transverse displacement
            frag_abut['DLS-' + str(dls_i + 1)] = np.zeros([len(self.iml_fit), 2])
            abut_coeff['DLS-' + str(dls_i + 1)] = np.zeros([2, num_abutments])
            abut_dls = eval('self.abut_dmdls' + str(dls_i + 1))
            for a_i in range(num_abutments):  # Computation for each abutment
                num_collapse = np.zeros([len(msa_poes)])
                num_gms = []
                threshold, beta_cap = abut_dls[a_i]

                for poe_i in range(len(msa_poes)):  # Computation per poe
                    poe = msa_poes[poe_i]
                    # Count exceedance of the DLS for the Abutment Response
                    num_gms.append(len(edp_abut[poe][:, a_i]))
                    # if dls_i!=3: # assume that dls4 is never reached
                    num_collapse[poe_i] = np.sum(edp_abut[poe][:, a_i] >= threshold)

                # Estimation of fragility curve parameter for the element
                theta, beta_rec = mle_fit(msa_imls, num_gms, num_collapse)
                if max(num_collapse) == 0 or sum(num_collapse) == 1:
                    theta = 1e15
                    beta = 0.1
                else:
                    # Total Standard Deviation
                    beta = np.sqrt(beta_mod ** 2 + beta_cap ** 2 + beta_rec ** 2)
                p_occur = norm.cdf((np.log(self.iml_fit / theta)) / beta)
                frag_abut['DLS-' + str(dls_i + 1)][:, a_i] = p_occur
                abut_coeff['DLS-' + str(dls_i + 1)][0, a_i] = theta
                abut_coeff['DLS-' + str(dls_i + 1)][1, a_i] = beta

            # Component Type: Bearing
            # TODO: Modify this part to consider failure mechanisms separately
            # 1-) Bearing longitudinal displacement
            # 2-) Bearing transverse displacement
            # 3-) Bearing unseating displacement
            frag_bearing['DLS-' + str(dls_i + 1)] = np.zeros([len(self.iml_fit), num_bearings])
            bearing_coeff['DLS-' + str(dls_i + 1)] = np.zeros([2, num_bearings])
            bearing_dls = eval('self.bearing_dmdls' + str(dls_i + 1))
            for joint_i, joint in enumerate(joints):
                for bearing_i_per_joint in range(bearing_per_joint[joint_i]):
                    bearing_i = sum(bearing_per_joint[:joint_i]) + bearing_i_per_joint
                    threshold, beta_cap = bearing_dls[joint][bearing_i_per_joint]
                    num_collapse = np.zeros([len(msa_poes)])
                    num_gms = []
                    for poe_i in range(len(msa_poes)):  # Computation per poe
                        # Count exceedance of the DLS for the pier responses
                        poe = msa_poes[poe_i]
                        num_gms.append(len(edp_bearing[poe][joint][:, bearing_i_per_joint]))
                        num_collapse[poe_i] = np.sum(edp_bearing[poe][joint][:, bearing_i_per_joint] >= threshold)

                    # Estimation of fragility curve parameter for the element
                    for val_i, val in enumerate(num_collapse):
                        if val_i != len(num_collapse) - 1 and num_collapse[val_i] > num_collapse[val_i + 1]:
                            num_collapse[val_i + 1] = num_collapse[val_i]

                    theta, beta_rec = mle_fit(msa_imls, num_gms, num_collapse)
                    if max(num_collapse) == 0 or sum(num_collapse) == 1:
                        theta = 1e15
                        beta = 0.1
                    else:
                        # Total Standard Deviation
                        beta = np.sqrt(beta_mod ** 2 + beta_cap ** 2 + beta_rec ** 2)
                    p_occur = norm.cdf((np.log(self.iml_fit / theta)) / beta)
                    frag_bearing['DLS-' + str(dls_i + 1)][:, bearing_i] = p_occur
                    bearing_coeff['DLS-' + str(dls_i + 1)][0, bearing_i] = theta
                    bearing_coeff['DLS-' + str(dls_i + 1)][1, bearing_i] = beta

            # Computation of System Fragility Curves for each DLS
            frag_all = np.concatenate((frag_pier['DLS-' + str(dls_i + 1)], frag_abut['DLS-' + str(dls_i + 1)],
                                       frag_bearing['DLS-' + str(dls_i + 1)]), axis=1)
            if dls_i == 3 and self.pier_dmdls4S is not None:
                frag_all = np.concatenate((frag_all, frag_pier['DLS-4S']), axis=1)
                temp = np.concatenate((frag_pier['DLS-4'], frag_pier['DLS-4S']), axis=1)
                m_frag_pier = np.max(temp, axis=1)
            else:
                m_frag_pier = np.max(frag_pier['DLS-' + str(dls_i + 1)], axis=1)
            m_frag_abut = np.max(frag_abut['DLS-' + str(dls_i + 1)], axis=1)
            m_frag_bearing = np.max(frag_bearing['DLS-' + str(dls_i + 1)], axis=1)
            # TODO discuss this part with Ricardo.
            # if dls_i == 0 or dls_i == 1:
            #     m_frag_bearing = np.zeros(m_frag_bearing.shape)

            # Lower bound fragility, estimated assuming perfect correlation between components (unconservative)
            self.frag_lower[:, dls_i + 1] = np.max(frag_all, axis=1)
            # Upper bound fragility, estimated assuming no correlation between components (conservative)
            no_collapse_probabilities = np.ones(frag_all.shape[0])
            for comp_i in range(frag_all.shape[1]):
                no_collapse_probabilities = no_collapse_probabilities * (1 - frag_all[:, comp_i])
            self.frag_upper[:, dls_i + 1] = 1 - no_collapse_probabilities
            # System-level fragility (something in between, Perdomo et al. 2020)
            # Assuming perfect correlation for failure mechanisms of components with same type:
            # Assuming no correlation between different type of components
            self.frag_sys[:, dls_i + 1] = 1 - (1 - m_frag_pier) * (1 - m_frag_abut) * (1 - m_frag_bearing)

            # theta_str0, beta_str0 = fit_log_normal_distribution(self.frag_sys[:, 0],
            #                                                     self.frag_sys[:, dls_i + 1])
            if np.all(self.frag_sys[:, dls_i + 1] <= 1e-5):
                beta_str = 0.1
                theta_str = 1e15
            else:
                idx84 = find_nearest(self.frag_sys[:, dls_i + 1], 0.84)
                idx16 = find_nearest(self.frag_sys[:, dls_i + 1], 0.16)
                idx50 = find_nearest(self.frag_sys[:, dls_i + 1], 0.5)
                beta_str = np.log(self.frag_sys[idx84, 0] / self.frag_sys[idx16, 0]) / 2
                theta_str = self.frag_sys[idx50, 0]

            self.frag_sys_param[dls_i, :] = theta_str, beta_str
            # self.frag_sys[:, dls_i + 1] = norm.cdf((np.log(self.iml_fit / theta_str)) / beta_str)
        self.frag_comp_param = {'pier': pier_coeff, 'abutment': abut_coeff, 'bearing': bearing_coeff}

    def plot_fragility(self):
        """
        Details
        -------
        Plots fragility curves. Must have run get_fragility before using it.

        """

        colors = ['green', 'orange', 'red', 'black']
        legends = ['DLS-1', 'DLS-2', 'DLS-3', 'DLS-4']

        # system level fragility curves
        plt.figure(figsize=(8, 6))
        temp = ['none'] * 4
        for dls_i in range(0, 4):
            theta = self.frag_sys_param[dls_i, 0]
            beta = self.frag_sys_param[dls_i, 1]
            p_occur = self.frag_sys[:, dls_i + 1]
            plt.plot(self.iml_fit, p_occur, c=colors[dls_i], label=legends[dls_i])
            temp[dls_i] = fr'$\theta_{{DLS{(dls_i + 1):d}}}={theta:.3f} | \beta_{{DLS{(dls_i + 1):d}}}={beta:.3f}$'

        textstr = '\n'.join(temp)
        props = dict(boxstyle='round', facecolor='none', edgecolor='none')
        plt.text(max(self.iml_fit), 0.0, textstr,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props, family='Times New Roman')

        plt.xlim([0, max(self.iml_fit)])
        plt.ylim([0, 1])
        plt.title('Structure Fragility Curves')
        plt.xlabel('Intensity Measure Level')
        plt.ylabel('Probability of Exceedance')
        plt.legend(frameon=False, loc='center right')
        plt.grid(True)
        plt.tight_layout()

        # Save
        fname = os.path.join(self.output_folder, 'System Fragility.png')
        plt.savefig(fname)
        plt.close()

        # Component fragility curves
        num_piers = self.frag_comp_param['pier']['DLS-1'].shape[1]
        num_abutments = self.frag_comp_param['abutment']['DLS-1'].shape[1]
        num_bearings = self.frag_comp_param['bearing']['DLS-1'].shape[1]
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 12))
        for dls_i in range(0, 4):
            if dls_i == 0:
                row = 0
                col = 0
            elif dls_i == 1:
                row = 0
                col = 1
            elif dls_i == 2:
                row = 1
                col = 0
            elif dls_i == 3:
                row = 1
                col = 1

            for p_i in range(num_piers):
                theta = self.frag_comp_param['pier']['DLS-' + str(dls_i + 1)][0, p_i]
                beta = self.frag_comp_param['pier']['DLS-' + str(dls_i + 1)][1, p_i]
                p_occur = norm.cdf(np.log(self.iml_fit / theta) / beta)
                ax[row, col].plot(self.iml_fit, p_occur, c='black', label='Piers')

                if dls_i == 3 and self.pier_edp_type != 1:
                    theta = self.frag_comp_param['pier']['DLS-4S'][0, p_i]
                    beta = self.frag_comp_param['pier']['DLS-4S'][1, p_i]
                    p_occur = norm.cdf(np.log(self.iml_fit / theta) / beta)
                    ax[row, col].plot(self.iml_fit, p_occur, c='black', label='Piers-S', ls='--')

            for a_i in range(num_abutments):
                theta = self.frag_comp_param['abutment']['DLS-' + str(dls_i + 1)][0, a_i]
                beta = self.frag_comp_param['abutment']['DLS-' + str(dls_i + 1)][1, a_i]
                p_occur = norm.cdf(np.log(self.iml_fit / theta) / beta)
                ax[row, col].plot(self.iml_fit, p_occur, c='red', label='Abutments')

            for b_i in range(num_bearings):
                theta = self.frag_comp_param['bearing']['DLS-' + str(dls_i + 1)][0, b_i]
                beta = self.frag_comp_param['bearing']['DLS-' + str(dls_i + 1)][1, b_i]
                p_occur = norm.cdf(np.log(self.iml_fit / theta) / beta)
                ax[row, col].plot(self.iml_fit, p_occur, c='blue', label='Bearings')

            ax[row, col].set_xlim([0, max(self.iml_fit)])
            ax[row, col].set_ylim([0, 1])
            ax[row, col].set_title(f'DLS-{str(dls_i + 1)}', fontweight='bold')
            ax[row, col].grid(True)

        ax_main = fig.add_subplot(111, frameon=False)
        handles, labels = ax[row, col].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Shrink current axis's height by 10% on the bottom
        ax_main.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.12),
                       frameon=False, ncol=4)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.ylabel('Probability of Exceedance')
        plt.xlabel('Intensity Measure Level')
        plt.suptitle('Component Fragility Curves', fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 1.0])

        # Save
        fname = os.path.join(self.output_folder, 'component_fragility.png')
        plt.savefig(fname)
        plt.close()

    def get_loss_simplified(self):
        """
        Details
        -------
        Generates loss curves and calculates expected annual loss values following simplified approach (HAZUS).

        """

        # Damage ratios used to generate vulnerability curves
        damage_ratios = np.array([0.03, 0.08, 0.25, 1.0])

        # bridge replacement cost
        rep_cost = self.rep_cost
        # iml values of stripes
        msa_imls = self.input.msa_imls[:, 0]
        # probability of exceedance values of stripes
        msa_poes = self.input.msa_imls[:, 1]
        # increased sample space
        iml_fit = self.iml_fit
        # mean annual frequency of exceedance corresponding to iml_fit
        lambda_fit = self.lambda_fit
        # Fragility curve parameters for the current bridge
        poedls = self.frag_sys[:, 1:]

        # Array for storing mean return period of the IM level
        return_t = np.zeros([len(msa_imls), 1])
        # Array for storing the expected simplified loss per IM level (direct)
        se_loss = np.zeros([len(msa_imls), 1])
        # Array for storing the standard deviation of the simplified loss per IM level (direct)
        ssd_loss = np.zeros([len(msa_imls), 1])
        # Array for collapse given IML, required for mean annual frequency of collapse
        col_iml = np.zeros([len(msa_imls), 1])

        # Computation of Mean Annual Rates of Exceedance
        # TODO: this can be refined using fitted values, but I calculate EAL using poe values only
        for j in range(len(msa_imls)):
            # Calculate return period
            poe = msa_poes[j]
            pooc = 1 - poe
            return_t[j] = 1 / (1 - np.exp((np.log(pooc) / self.tw)))

            # Find the position of iml in iml_range vector
            posiml = min(np.where(iml_fit >= msa_imls[j])[0])

            # get the corresponding probability of exceedance values
            poedls_iml = poedls[posiml, :]
            col_iml[j] = poedls_iml[-1]
            # Compute probability of being each particular DLS
            temp = np.flip(poedls_iml)  # reverse the array based on columns
            temp = np.insert(temp, [0], 0)  # append zeros to the first column
            diff_poe = np.flip(abs(np.diff(temp)))  # get the absolute difference between each
            diff_poe[diff_poe < 0] = 0  # check if there is any negative value, if there are any, make them 0

            # Compute the Compound damage ratio
            drc = np.sum(diff_poe * damage_ratios)
            sigma_drc = np.sqrt(np.sum(diff_poe * (damage_ratios - drc) ** 2))

            # Compute estimated losses for the scenario earthquake
            exploss = drc * rep_cost
            sigmaloss = sigma_drc * rep_cost
            if exploss > rep_cost:  # sanity check
                exploss = rep_cost
            se_loss[j] = exploss
            ssd_loss[j] = sigmaloss

        iml_mafe = 1 / return_t  # Array for computation of IM level mean annual frequency of exceedance
        se_loss_std = se_loss + ssd_loss  # Expected loss + 1 standard deviation in loss
        # Work out with the Hazard Curve of the bridge
        # Find boundary IML in the hazard curve
        deltaIML = np.zeros([len(msa_imls) - 1, 1])
        for j in range(len(deltaIML)):
            deltaIML[j] = msa_imls[j] - msa_imls[j + 1]

        # Find boundary IML in the hazard curve
        deltaIML = np.diff(msa_imls)
        pointsIML = np.zeros(len(msa_imls) + 1)
        for j in range(len(pointsIML)):
            if j == 0:
                pointsIML[j] = msa_imls[j] - 0.5 * deltaIML[j]
            elif j == len(pointsIML) - 1:
                pointsIML[j] = msa_imls[j - 1] + 0.5 * deltaIML[j - 2]
            else:
                pointsIML[j] = msa_imls[j - 1] + 0.5 * deltaIML[j - 1]

        Lambda_haz_vals = np.zeros(len(msa_imls) + 1)
        for j in range(len(Lambda_haz_vals)):
            imlval = pointsIML[j]
            if imlval > max(iml_fit):
                imlval = max(iml_fit)
            if imlval < min(iml_fit):
                imlval = min(iml_fit)

            posimlval = min(np.where(iml_fit >= imlval)[0])
            Lambda_haz_vals[j] = lambda_fit[posimlval]

        delta_lambda_haz = np.zeros([len(msa_imls), 1])
        for j in range(len(delta_lambda_haz)):
            delta_lambda_haz[j] = abs(Lambda_haz_vals[j] - Lambda_haz_vals[j + 1])

        # Compute Mean Annualized Losses
        s_eal = sum(se_loss * delta_lambda_haz)
        col_mafe = sum(col_iml * delta_lambda_haz)
        s_ealr = s_eal * 100 / rep_cost

        # Save the results
        self.loss_simplified['exp_loss'] = se_loss
        self.loss_simplified['std_loss'] = se_loss_std
        self.loss_simplified['iml_mafe'] = iml_mafe
        self.loss_simplified['eal'] = s_eal
        self.loss_simplified['ealr'] = s_ealr
        self.loss_simplified['delta_lambda_haz'] = delta_lambda_haz
        self.loss_simplified['col_mafe'] = col_mafe

    def get_loss_comprehensive(self):
        """
        Details
        -------
        Generates loss curves and calculates expected annual loss values.

        """

        # The Function to Check for Prime Numbers
        def is_prime(number):
            if number > 1:
                for num in range(2, int(number ** 0.5) + 1):
                    if number % num == 0:
                        return False
                return True
            return False

        # Finding All Prime Numbers Between a range
        prime_numbers = []
        for num in range(90, 152):
            if is_prime(num):
                prime_numbers.append(num)

        # Comprehensive loss calculation parameters (FEMA P-58)
        num_realizations_list = [val**2 for val in prime_numbers]  # Number of realizations for simulated demands per IM Level
        # Dispersion on the repair cost for DLS-1,DLS-2,DLS-3 and replacement cost (DLS-4)
        beta_cost_dls = [0.4, 0.4, 0.4, 0.5]
        num_points_loss_curve = 150  # Number of points for the computation of the loss curve
        min_loss_curve_value = 10000  # Minimum value of loss for the loss curve
        per_max_loss_on_curve = 95  # Percentile of the loss that takes the values of maximum loss for construction of the loss curve

        # Modelling Uncertainty
        beta_mod = np.sqrt(self.beta_c ** 2 + self.beta_q ** 2)

        # Initialize the dictionary for the bridge
        self.loss_comprehensive = {'mean_valpn_coll_repcost': {}, 'nocoll_repcost': {}, 'coll_repcost': {}}
        # Number of components
        joints = list(self.input.EleIDsBearing.keys())
        bearing_per_joint = [len(self.input.EleIDsBearing[joint]) for joint in self.input.EleIDsBearing]
        num_bearings = sum(bearing_per_joint)
        num_piers = len(self.input.EleIDsPier)

        # get parameters to use
        rep_cost = self.rep_cost
        piers_dls1 = self.pier_dmdls1
        piers_dls2 = self.pier_dmdls2
        piers_dls3 = self.pier_dmdls3
        abut_dls1 = self.abut_dmdls1
        abut_dls2 = self.abut_dmdls2
        abut_dls3 = self.abut_dmdls3
        bearing_dls1 = self.bearing_dmdls1
        bearing_dls2 = self.bearing_dmdls2
        bearing_dls3 = self.bearing_dmdls3
        bent_idx_map = {}
        idx = 0
        for _, bent in enumerate(self.pg_input['bent']):
            idxs = []
            for _, pier in enumerate(self.pg_input['bent'][bent]):
                idxs.append(idx)
                idx += 1
            bent_idx_map[bent] = idxs

        abut_idx_map = [num_piers, num_piers + 1]
        bearing_idx_map = {}
        idx1 = num_piers + 2
        for j, joint in enumerate(joints):
            bearing_idx_map[joint] = [idx1 + idx for idx in range(bearing_per_joint[j])]
            idx1 += bearing_per_joint[j]

        iml_fit = self.iml_fit
        msa_imls = self.input.msa_imls[:, 0]
        msa_poes = self.input.msa_imls[:, 1]
        edp_pier = {}
        edp_abut = {}
        edp_bearing = {}
        for poe in msa_poes:
            if self.pier_edp_type == 1:
                edp_pier[poe] = self.input.msa_edps[poe]['mu_curv']
            elif self.pier_edp_type == 2:
                edp_pier[poe] = self.input.msa_edps[poe]['mu_disp']
            elif self.pier_edp_type == 3:
                edp_pier[poe] = self.input.msa_edps[poe]['drift_ratio']
            # TODO: Instead of retrieving SRSS of displacements, retrieve distinct edps for these components
            edp_abut[poe] = self.input.msa_edps[poe]['abut_disp']
            edp_bearing[poe] = self.input.msa_edps[poe]['bearing_disp']
        frag_str = self.frag_sys[:, 1:]
        delta_lambda_haz = self.loss_simplified['delta_lambda_haz']

        # initialize some parameters
        comp_loss_info = {}  # a dictionary to save loss info
        exp_loss_direct = np.zeros([len(msa_imls), 1])
        exp_loss_indirect = np.zeros([len(msa_imls), 1])
        std_loss_direct = np.zeros([len(msa_imls), 1])
        std_loss_indirect = np.zeros([len(msa_imls), 1])
        exp_loss = np.zeros([len(msa_imls), 1])
        std_loss = np.zeros([len(msa_imls), 1])
        exp_loss_nocoll = np.zeros([len(msa_imls), 1])
        exp_loss_coll = np.zeros([len(msa_imls), 1])
        nriml = np.zeros([len(msa_imls), 1])
        per95_loss = np.zeros([len(msa_imls), 1])
        per86_loss = np.zeros([len(msa_imls), 1])
        per50_loss = np.zeros([len(msa_imls), 1])
        per14_loss = np.zeros([len(msa_imls), 1])
        per05_loss = np.zeros([len(msa_imls), 1])
        # Computation of Simulated Demands for the Loss Calculation
        # Implementation of the Algorithm proposed in FEMA-P-58 Appendix G
        for i, poe in enumerate(msa_poes):
            # Note:
            # In numpy calculations columns are observations, rows are components
            # However, this is the opposite in MATLAB. e.g. np.cov and cov (MATLAB)
            pier = np.transpose(edp_pier[poe])
            abut = np.transpose(edp_abut[poe])
            bearing = np.zeros((pier.shape[1], num_bearings))

            idx1 = 0
            for j, joint in enumerate(joints):
                idx2 = idx1 + bearing_per_joint[j]
                bearing[:, idx1:idx2] = edp_bearing[poe][joint]
                idx1 += bearing_per_joint[j]
            bearing = np.transpose(bearing)
            edps_all = np.concatenate([pier, abut, bearing], axis=0)
            # Recover Demand Estimations from Structural Analysis
            # num_var: number of demand parameters or components
            # num_gm:  number of gm used at the current iml
            num_var, num_gm = edps_all.shape
            vbeta_m = np.ones([1, num_var]) * beta_mod
            # Taking natural logarithm of the EDPs Estimation from Structural Analysis
            lnEDPs = np.log(edps_all)
            # Computing mean of logarithmic demands
            lnEDPs_m = np.array([np.mean(lnEDPs, axis=1)]).transpose()  # calculate the mean edp for each component
            # Computing Covariance of the matrix of logarithmic demands
            lnEDPs_cov = np.cov(lnEDPs)
            lnEDPs_cov[lnEDPs_cov == 0] = 1e-30
            # Computing the rank of the covariance matrix
            lnEDPs_cov_rank = np.linalg.matrix_rank(lnEDPs_cov)
            # Inflating variance due to epistemic uncertainty in modelling
            sigma = np.array([np.sqrt(np.diag(lnEDPs_cov))]).transpose()
            sigmap2 = sigma ** 2
            sigma_t = sigma.transpose()
            R = lnEDPs_cov / (sigma * sigma_t)
            vbeta_m = vbeta_m.transpose()
            sigmap2 = sigmap2 + (vbeta_m * vbeta_m)
            sigma = np.sqrt(sigmap2)
            sigma2 = sigma * sigma.transpose()
            lnEDPs_cov_inflated = R * sigma2
            # Computing Eigenvalues and Eigenvectors of the covariance matrix
            eigenValues, eigenVectors = np.linalg.eigh(lnEDPs_cov_inflated)
            lnEDPs_std = np.array([np.sqrt(np.diag(lnEDPs_cov_inflated))])
            D2_total = eigenValues
            L_total = eigenVectors
            # Partition of L_total to L_use
            # Partition of D2_total to D2_use
            if lnEDPs_cov_rank >= num_var:
                L_use = L_total
                D2_use = D2_total
            else:
                L_use = L_total[:, num_var - lnEDPs_cov_rank:]
                D2_use = D2_total[num_var - lnEDPs_cov_rank:]
            # Find the square root of D2_use and call is D_use
            D_use = np.diag(D2_use ** 0.5)

            # Compute Simulated Demands
            for num_realizations in num_realizations_list:
                # Generate Standard Random Numbers
                if lnEDPs_cov_rank >= num_var:
                    U = do_sampling(num_var, num_realizations, sampling_type = 'LHS')  # Latin Hypercube Sampling
                    # U = do_sampling(num_var, num_realizations, sampling_type = 'MCS') # Monte Carlo Sampling
                else:
                    U = do_sampling(lnEDPs_cov_rank, num_realizations, sampling_type = 'LHS')  # Latin Hypercube Sampling
                    # U = do_sampling(lnEDPs_cov_rank, num_realizations, sampling_type = 'MCS') # Monte Carlo Sampling

                # Apply normalisation in case of Latin Hypercube Sampling
                U = norm(loc=0, scale=1).ppf(U)
                U = U.transpose()
                # Create Lambda = D_use . L_use
                Lambda = np.matmul(L_use, D_use)
                # Create realizations matrix
                Z = np.matmul(Lambda, U) + np.matmul(lnEDPs_m, np.ones([1, num_realizations]))

                # Simulated Demands
                if not Z.any():
                    Z = np.real(Z)

                lnEDPs_sim_m = np.mean(Z, axis=1)
                lnEDPs_sim_std = np.std(Z, axis=1)
                mean_ratio = lnEDPs_sim_m / lnEDPs_m.flatten()
                std_ratio = lnEDPs_sim_std / lnEDPs_std

                # Check mean and std ratios, if they are ok we are fine, we have good number of samples.
                max_mean_ratio = np.max(mean_ratio)
                min_mean_ratio = np.min(mean_ratio)
                max_std_ratio = np.max(std_ratio)
                min_std_ratio = np.min(std_ratio)
                if not (max_mean_ratio > 1.20 or min_mean_ratio < 0.80 or max_std_ratio > 1.20 or min_std_ratio < 0.80):
                    break

            edps_sim = np.exp(Z)
            nriml[i] = num_realizations

            # Estimation of repair cost per each realization
            num_pg = len(bent_idx_map) + len(abut_idx_map)
            coll_dir_cost = np.zeros([num_realizations, 1])
            nocoll_comp_repcost = np.zeros([num_realizations, num_pg])
            coll_indir_cost = np.zeros([num_realizations, 1])
            no_coll_indir_cost = np.zeros([num_realizations, 1])

            # Determine the probability of collapse
            posiml = min(np.where(iml_fit >= msa_imls[i])[0])
            p_coll = frag_str[posiml, 3]  # CDF

            # TODO: This is the real deal! Compute losses...
            # Note: Alternatively, realizations can be performed for each component separately,
            # and then the collapse case can be identified by checking if any component is in DLS4 or not.
            # Here collapse case is identified based on collapse fragility curves.
            for j in range(num_realizations):
                # Determine if there is structural collapse
                test_coll = np.random.uniform(0, 1)  # uniformly distributed random variable
                if test_coll <= p_coll:  # Replacement Cost if Collapse
                    coll_dir_cost[j] = np.random.lognormal(np.log(rep_cost) - 0.5 * beta_cost_dls[3] ** 2,
                                                          beta_cost_dls[3])
                    replacement_duration = np.random.lognormal(np.log(self.repair_duration['collapse'][1]) -
                                                               0.5 * self.repair_duration['collapse'][2] ** 2,
                                                               self.repair_duration['collapse'][2])
                    coll_indir_cost[j] = self.daily_indirect_loss[4, 1] * replacement_duration

                else:  # Repair Cost if Non-Collapse
                    func_pg = []  # get functionality values for each performance group
                    cwd_pg = []  # get cwd values for each performance group
                    dls_pg = []  # get dls for each performance group

                    demand = edps_sim[:, j]
                    for joint in joints:  # consider each joint as a performance group
                        func_sub = 100
                        func_bearing = 100
                        # 1-) Compute losses and DLS for abutments
                        if joint == 0 or joint == joints[-1]:
                            if joint == 0:
                                abut_idx = 0
                            elif joint == joints[-1]:
                                abut_idx = 1
                            # Compute losses for abutment
                            edp_comp = demand[abut_idx_map[abut_idx]]
                            poedls1 = normal_cdf((np.log(edp_comp / abut_dls1[abut_idx, 0])) / abut_dls1[abut_idx, 1])
                            poedls2 = normal_cdf((np.log(edp_comp / abut_dls2[abut_idx, 0])) / abut_dls2[abut_idx, 1])
                            poedls3 = normal_cdf((np.log(edp_comp / abut_dls3[abut_idx, 0])) / abut_dls3[abut_idx, 1])
                            if poedls3 > poedls2:
                                poedls2 = 0
                            if poedls2 > poedls1:
                                poedls1 = 0
                            P1 = 1 - poedls1
                            P2 = 1 - poedls2
                            P3 = 1 - poedls3
                            # uniformly distributed random variable
                            test_dls = np.random.uniform(0, 1)
                            if test_dls < P1:
                                dls_sub = 0
                                meancost = 1e-15
                                betacost = beta_cost_dls[0]
                            elif P1 <= test_dls < P2:
                                dls_sub = 1
                                meancost = self.pg_cost['abutment'][abut_idx, 0]
                                betacost = beta_cost_dls[0]
                                func_sub = self.repair_duration['abutment'][0, 0]
                            elif P2 <= test_dls < P3:
                                dls_sub = 2
                                meancost = self.pg_cost['abutment'][abut_idx, 1]
                                betacost = beta_cost_dls[1]
                                func_sub = self.repair_duration['abutment'][1, 0]
                            elif P3 <= test_dls <= 1:
                                dls_sub = 3
                                meancost = self.pg_cost['abutment'][abut_idx, 2]
                                betacost = beta_cost_dls[2]
                                func_sub = self.repair_duration['abutment'][2, 0]

                            cost_sub = np.random.lognormal(np.log(meancost) - 0.5 * betacost ** 2, betacost)

                            if func_sub == 100:
                                repair_duration_sub = 0
                            else:
                                row = np.where(self.repair_duration['abutment'][:, 0] == func_sub)[0][0]
                                repair_duration_sub = np.random.lognormal(np.log(self.repair_duration['abutment'][row, 1]) -
                                                                          0.5 * self.repair_duration['abutment'][row, 2] ** 2,
                                                                          self.repair_duration['abutment'][row, 2])

                        # 2-) Compute losses and DLS for bents
                        else:
                            dls_sub = 0
                            bent_tag = 'B' + str(joint)
                            edp_group = demand[bent_idx_map[bent_tag]]
                            for k in range(len(edp_group)):
                                edp_comp = edp_group[k]
                                poedls1 = normal_cdf(
                                    (np.log(edp_comp / piers_dls1[bent_tag][k, 0])) / piers_dls1[bent_tag][k, 1])
                                poedls2 = normal_cdf(
                                    (np.log(edp_comp / piers_dls2[bent_tag][k, 0])) / piers_dls2[bent_tag][k, 1])
                                poedls3 = normal_cdf(
                                    (np.log(edp_comp / piers_dls3[bent_tag][k, 0])) / piers_dls3[bent_tag][k, 1])
                                if poedls3 > poedls2:
                                    poedls2 = 0
                                if poedls2 > poedls1:
                                    poedls1 = 0
                                P1 = 1 - poedls1
                                P2 = 1 - poedls2
                                P3 = 1 - poedls3
                                # get uniformly distributed random variable
                                test_dls = np.random.uniform(0, 1)
                                # determine dls of the pier
                                if test_dls < P1:
                                    dls_tmp = 0
                                elif P1 <= test_dls < P2:
                                    dls_tmp = 1
                                elif P2 <= test_dls < P3:
                                    dls_tmp = 2
                                elif P3 <= test_dls <= 1:
                                    dls_tmp = 3
                                # update dls of the bent
                                dls_sub = max(dls_sub, dls_tmp)

                            # get losses for bent
                            if dls_sub == 0:
                                meancost = 1e-15
                                betacost = beta_cost_dls[0]
                            elif dls_sub == 1:
                                meancost = self.pg_cost['bent'][joint - 1, 0]
                                betacost = beta_cost_dls[0]
                                func_sub = self.repair_duration['bent'][0, 0]
                            elif dls_sub == 2:
                                meancost = self.pg_cost['bent'][joint - 1, 1]
                                betacost = beta_cost_dls[1]
                                func_sub = self.repair_duration['bent'][1, 0]
                            elif dls_sub == 3:
                                meancost = self.pg_cost['bent'][joint - 1, 2]
                                betacost = beta_cost_dls[2]
                                func_sub = self.repair_duration['bent'][2, 0]

                            cost_sub = np.random.lognormal(np.log(meancost) - 0.5 * betacost ** 2, betacost)
                            if func_sub == 100:
                                repair_duration_sub = 0
                            else:
                                row = np.where(self.repair_duration['bent'][:, 0] == func_sub)[0][0]
                                repair_duration_sub = np.random.lognormal(np.log(self.repair_duration['bent'][row, 1]) -
                                                                          0.5 * self.repair_duration['bent'][row, 2] ** 2,
                                                                          self.repair_duration['bent'][row, 2])

                        # 3-) Compute losses and DLS for bearing group
                        dls_bearing = 0  # initialize DLS of the bearing group on joints
                        if dls_sub == 3:  # abutment or bent will be reconstructed -> hence replace the bearings and fix the deck as well
                            meancost = self.pg_cost['bearing'][joint, 2] - self.temporal_support_costs[joint]
                            betacost = beta_cost_dls[2]
                            cost_bearing = np.random.lognormal(np.log(meancost) - 0.5 * betacost ** 2, betacost)
                            repair_duration_bearing = np.random.lognormal(np.log(self.repair_duration['bearing'][row, 1]) -
                                                                          0.5 * self.repair_duration['bearing'][row, 2] ** 2,
                                                                          self.repair_duration['bearing'][row, 2])
                            func_bearing = self.repair_duration['bearing'][2, 0]
                        else:  # substructure is not going to be reconstructed
                            edp_group = demand[bearing_idx_map[joint]]
                            for k in range(len(edp_group)):  # for bearings
                                edp_comp = edp_group[k]
                                poedls1 = normal_cdf(
                                    (np.log(edp_comp / bearing_dls1[joint][k, 0])) / bearing_dls1[joint][k, 1])
                                poedls2 = normal_cdf(
                                    (np.log(edp_comp / bearing_dls2[joint][k, 0])) / bearing_dls2[joint][k, 1])
                                poedls3 = normal_cdf(
                                    (np.log(edp_comp / bearing_dls3[joint][k, 0])) / bearing_dls3[joint][k, 1])
                                if poedls3 > poedls2:
                                    poedls2 = 0
                                if poedls2 > poedls1:
                                    poedls1 = 0
                                P1 = 1 - poedls1
                                P2 = 1 - poedls2
                                P3 = 1 - poedls3
                                # uniformly distributed random variable
                                test_dls = np.random.uniform(0, 1)
                                # determine dls of the bearing
                                if test_dls < P1:
                                    dls_tmp = 0
                                elif P1 <= test_dls < P2:
                                    dls_tmp = 1
                                elif P2 <= test_dls < P3:
                                    dls_tmp = 2
                                elif P3 <= test_dls <= 1:
                                    dls_tmp = 3
                                # update dls of the bearing group
                                dls_bearing = max(dls_bearing, dls_tmp)
                            # get losses for bearing
                            if dls_bearing == 0:
                                meancost = 1e-15
                                betacost = beta_cost_dls[0]
                            elif dls_bearing == 1:
                                meancost = self.pg_cost['bearing'][joint, 0]
                                betacost = beta_cost_dls[0]
                                func_bearing = self.repair_duration['bearing'][0, 0]
                            elif dls_bearing == 2:
                                meancost = self.pg_cost['bearing'][joint, 1]
                                betacost = beta_cost_dls[1]
                                func_bearing = self.repair_duration['bearing'][1, 0]
                            elif dls_bearing == 3:
                                meancost = self.pg_cost['bearing'][joint, 2]
                                betacost = beta_cost_dls[2]
                                func_bearing = self.repair_duration['bearing'][2, 0]
                            cost_bearing = np.random.lognormal(np.log(meancost) - 0.5 * betacost ** 2, betacost)

                            if func_bearing == 100:
                                repair_duration_bearing = 0
                            else:
                                row = np.where(self.repair_duration['bearing'][:, 0] == func_bearing)[0][0]
                                repair_duration_bearing = np.random.lognormal(np.log(self.repair_duration['bearing'][row, 1]) -
                                                                          0.5 * self.repair_duration['bearing'][row, 2] ** 2,
                                                                          self.repair_duration['bearing'][row, 2])

                        dls_pg.append(max(dls_sub, dls_bearing))
                        func_pg.append(min(func_bearing, func_sub))  # functionality
                        cwd_pg.append(repair_duration_sub + repair_duration_bearing)  # repair duration
                        nocoll_comp_repcost[j, joint] += cost_sub + cost_bearing

                    zipped_lists = zip(func_pg, cwd_pg)
                    sorted_pairs = sorted(zipped_lists)
                    tuples = zip(*sorted_pairs)
                    func_pg, cwd_pg = [list(x) for x in tuples]

                    # Compute the cost due to the planning of repair activities
                    if max(dls_pg) != 0:
                        row = max(dls_pg) - 1
                        duration = np.random.lognormal(np.log(self.repair_duration['planning'][row, 0]) -
                                                              0.5 * self.repair_duration['planning'][row, 1] ** 2,
                                                              self.repair_duration['planning'][row, 1])
                        functionality = min(func_pg)
                        row = np.where(self.daily_indirect_loss[:, 0] == functionality)[0][0]
                        no_coll_indir_cost[j] += duration * self.daily_indirect_loss[row, 1]

                    # Compute costs due to the repair activities
                    idx0 = 0
                    for x in range(0, num_pg, self.num_crews):
                        idx1 = idx0 + self.num_crews
                        duration = max(cwd_pg[idx0:idx1])
                        functionality = min(func_pg[idx0:idx1])
                        row = np.where(self.daily_indirect_loss[:, 0] == functionality)[0][0]
                        no_coll_indir_cost[j] += duration * self.daily_indirect_loss[row, 1]
                        idx0 = idx0 + self.num_crews

            no_coll_dir_cost = np.array([np.sum(nocoll_comp_repcost, axis=1)]).transpose()  # size = [nrx1]

            direct_cost = coll_dir_cost + no_coll_dir_cost
            indirect_cost = coll_indir_cost + no_coll_indir_cost
            exp_loss_coll[i] = np.mean(coll_dir_cost + coll_indir_cost)
            exp_loss_nocoll[i] = np.mean(no_coll_dir_cost + no_coll_indir_cost)
            exp_loss_direct[i] = np.mean(direct_cost)
            std_loss_direct[i] = np.std(direct_cost)
            exp_loss_indirect[i] = np.mean(indirect_cost)
            std_loss_indirect[i] = np.std(indirect_cost)
            exp_loss[i] = np.mean(direct_cost + indirect_cost)
            std_loss[i] = np.std(direct_cost + indirect_cost)
            per95_loss[i] = np.percentile(direct_cost + indirect_cost, 95)
            per86_loss[i] = np.percentile(direct_cost + indirect_cost, 86)
            per50_loss[i] = np.percentile(direct_cost + indirect_cost, 50)
            per14_loss[i] = np.percentile(direct_cost + indirect_cost, 14)
            per05_loss[i] = np.percentile(direct_cost + indirect_cost, 5)

            posvalncollcost = np.where(no_coll_dir_cost != 0)[0]
            valp_ncollcost = np.zeros([len(posvalncollcost), num_pg])
            for nn in range(len(posvalncollcost)):
                pos = posvalncollcost[nn]
                valp_ncollcost[nn, :] = nocoll_comp_repcost[pos, :]

            # Perform a single disaggregation of results for the No Collapse Cases (direct_loss)
            mean_valpn_coll_repcost = np.mean(valp_ncollcost, axis=0)

            self.loss_comprehensive['mean_valpn_coll_repcost'][poe] = mean_valpn_coll_repcost
            self.loss_comprehensive['nocoll_repcost'][poe] = no_coll_dir_cost + no_coll_indir_cost
            self.loss_comprehensive['coll_repcost'][poe] = coll_dir_cost + coll_indir_cost

            costvals, freq = ecdf(direct_cost + indirect_cost)
            invfreq = np.array([1 - freq]).transpose()
            loss_curve = np.concatenate([costvals, invfreq], axis=1)
            comp_loss_info['loss_curve_iml' + str(i + 1)] = loss_curve

        # Compute Mean Annualized Losses
        eal = sum(exp_loss * delta_lambda_haz)  # total cost
        eal_dir = sum(exp_loss_direct * delta_lambda_haz)  # direct loss only
        eal_indir = sum(exp_loss_indirect * delta_lambda_haz)  # indirect loss only
        eal_coll = sum(exp_loss_coll * delta_lambda_haz) # collapse cases only
        eal_nocoll = sum(exp_loss_nocoll * delta_lambda_haz) # no collapse cases
        # Compute Mean Annualized Loss Ratios
        ealr = eal * 100 / rep_cost
        ealr_dir = eal_dir * 100 / rep_cost
        ealr_indir = eal_indir * 100 / rep_cost
        ealr_coll = eal_coll * 100 / rep_cost
        ealr_nocoll = eal_nocoll * 100 / rep_cost

        # Computation of Expected Loss Annual Rate of Exceedance
        poss_max_loss = np.zeros([len(msa_imls), 1])
        for i in range(len(msa_imls)):
            loss_curve = comp_loss_info['loss_curve_iml' + str(i + 1)]
            loss_test = loss_curve[:, 0]
            pone_test = loss_curve[:, 1]
            poss_max_loss[i] = np.percentile(loss_test, per_max_loss_on_curve)

        max_loss = max(poss_max_loss)
        exp_loss_lc = np.zeros([num_points_loss_curve, 1])
        for i in range(num_points_loss_curve):
            if i == 0:
                exp_loss_lc[i] = min_loss_curve_value
            elif i == num_points_loss_curve - 1:
                exp_loss_lc[i] = max_loss
            else:
                exp_loss_lc[i] = min_loss_curve_value + i * (
                        max_loss - min_loss_curve_value) / num_points_loss_curve

        loss_mafe = np.zeros([num_points_loss_curve, 1])
        for i in range(num_points_loss_curve):
            eloss = exp_loss_lc[i]
            poneloss = np.zeros([len(msa_imls), 1])
            for j in range(len(msa_imls)):
                loss_curve = comp_loss_info['loss_curve_iml' + str(j + 1)]
                loss = loss_curve[:, 0]
                pone = loss_curve[:, 1]

                if max(loss) < eloss:
                    poneloss[j] = 0
                else:
                    posloss = min(np.where(loss >= eloss)[0])
                    poneloss[j] = pone[posloss]

            loss_mafe[i] = sum(poneloss * delta_lambda_haz)

        self.loss_comprehensive['loss_info'] = comp_loss_info
        self.loss_comprehensive['05_loss'] = per05_loss
        self.loss_comprehensive['14_loss'] = per14_loss
        self.loss_comprehensive['50_loss'] = per50_loss
        self.loss_comprehensive['86_loss'] = per86_loss
        self.loss_comprehensive['95_loss'] = per95_loss
        self.loss_comprehensive['exp_loss'] = exp_loss
        self.loss_comprehensive['std_loss'] = std_loss
        self.loss_comprehensive['exp_dir_loss'] = exp_loss_direct
        self.loss_comprehensive['std_dir_loss'] = std_loss_direct
        self.loss_comprehensive['exp_indir_loss'] = exp_loss_indirect
        self.loss_comprehensive['std_indir_loss'] = std_loss_indirect
        self.loss_comprehensive['eal'] = eal
        self.loss_comprehensive['ealr'] = ealr
        self.loss_comprehensive['eal_coll'] = eal_coll
        self.loss_comprehensive['ealr_coll'] = ealr_coll
        self.loss_comprehensive['eal_nocoll'] = eal_nocoll
        self.loss_comprehensive['ealr_nocoll'] = ealr_nocoll
        self.loss_comprehensive['eal_dir'] = eal_dir
        self.loss_comprehensive['ealr_dir'] = ealr_dir
        self.loss_comprehensive['eal_indir'] = eal_indir
        self.loss_comprehensive['ealr_indir'] = ealr_indir
        self.loss_comprehensive['exp_loss_lc'] = exp_loss_lc
        self.loss_comprehensive['loss_mafe'] = loss_mafe

    def plot_loss(self):
        """
        Details
        -------
        Plots loss curves using component based (FEMA P-58) approach.
        Must have run get_loss_comprehensive before using it.

        """
        # Number of components
        joints = list(self.input.EleIDsBearing.keys())

        # get parameters to use
        msa_imls = self.input.msa_imls[:, 0]
        msa_poes = self.input.msa_imls[:, 1]
        iml_mafe = self.loss_simplified['iml_mafe']
        comp_loss_info = self.loss_comprehensive['loss_info']
        comp_loss = self.loss_comprehensive['exp_loss']
        comp_loss_dir = self.loss_comprehensive['exp_dir_loss']
        comp_loss_indir = self.loss_comprehensive['exp_indir_loss']
        se_loss = self.loss_simplified['exp_loss']
        s_ealr = float(self.loss_simplified['ealr'])
        c_eal = self.loss_comprehensive['eal']
        c_ealr = self.loss_comprehensive['ealr']
        c_ealr_dir = self.loss_comprehensive['ealr_dir']
        c_ealr_indir = self.loss_comprehensive['ealr_indir']
        exp_loss_lc = self.loss_comprehensive['exp_loss_lc']
        loss_mafe = self.loss_comprehensive['loss_mafe']
        per05_loss = self.loss_comprehensive['05_loss']
        per14_loss = self.loss_comprehensive['14_loss']
        per50_loss = self.loss_comprehensive['50_loss']
        per86_loss = self.loss_comprehensive['86_loss']
        per95_loss = self.loss_comprehensive['95_loss']

        for i in range(len(msa_imls)):
            poe = msa_poes[i]
            mean_valpn_coll_repcost = self.loss_comprehensive['mean_valpn_coll_repcost'][poe]
            nocoll_repcost = self.loss_comprehensive['nocoll_repcost'][poe]
            coll_repcost = self.loss_comprehensive['coll_repcost'][poe]
            repcost = coll_repcost + nocoll_repcost
            tot_mean = sum(mean_valpn_coll_repcost)
            costvals, freq = ecdf(repcost)
            invfreq = np.array([1 - freq]).transpose()

            # Plot Mean contribution per component for no collapse cases
            width = 0.8
            performance_groups = []
            for joint in joints:
                if joint == joints[0]:
                    performance_groups.append('A1')
                elif joint == joints[-1]:
                    performance_groups.append('A2')
                else:
                    performance_groups.append('B' + str(joint))

            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
            ax[0].bar(performance_groups, mean_valpn_coll_repcost, width, edgecolor='black')
            ax[0].set_title(f'Mean Direct Loss per Performance Group Given No Collapse - iml = {str(msa_imls[i])}',
                            fontsize=11)
            ax[0].set_ylabel(r'$\mu_{L_{comp}}$' + ' [\u20AC]')
            ax[0].grid(True)
            for tick in ax[0].xaxis.get_major_ticks():
                tick.label.set_fontsize(11)
                tick.label.set_rotation('vertical')
            ax[1].bar(performance_groups, mean_valpn_coll_repcost / tot_mean * 100, width, color='blue',
                      edgecolor='black')
            ax[1].set_xlabel('Performance Group')
            ax[1].set_ylabel(r'$\mu_{L_{comp}}$/$\mu_{L_{tot}}$ [%]')
            ax[1].grid(True)
            for tick in ax[1].xaxis.get_major_ticks():
                tick.label.set_fontsize(11)
                tick.label.set_rotation('vertical')
            fname = os.path.join(self.output_folder, f'comp_loss_{str(msa_imls[i])}.png')
            plt.savefig(fname)

            # Plot Histogram of no collapse cases
            pnocoll_repcost = nocoll_repcost[nocoll_repcost != 0]
            nbins = int(round(np.sqrt(len(pnocoll_repcost))))
            plt.figure(figsize=(8, 6))
            if nbins == 0:
                plt.hist(pnocoll_repcost, edgecolor='black')
            else:
                plt.hist(pnocoll_repcost, nbins, edgecolor='black')

            plt.title(f'No Collapse Cases - IML = {str(msa_imls[i])}', fontsize=11)
            plt.xlabel('Repair Cost - [\u20AC]')
            plt.ylabel('Frequency')
            plt.grid(True)
            fname = os.path.join(self.output_folder, f'hist_nocoll_iml_{str(msa_imls[i])}.png')
            plt.savefig(fname)

            # Plot Histogram of collapse cases
            pcoll_repcost = coll_repcost[coll_repcost != 0]
            nbins = int(round(np.sqrt(len(pcoll_repcost))))
            plt.figure(figsize=(8, 6))
            if nbins == 0:
                plt.hist(pcoll_repcost, edgecolor='black')
            else:
                plt.hist(pcoll_repcost, nbins, edgecolor='black')

            plt.title(f'Collapse Cases - IML = {str(msa_imls[i])}', fontsize=11)
            plt.xlabel('Repair Cost - [\u20AC]')
            plt.ylabel('Frequency')
            plt.grid(True)
            fname = os.path.join(self.output_folder, f'hist_coll_iml_{str(msa_imls[i])}.png')
            plt.savefig(fname)

            # Plot Histogram of all cases
            nbins = int(round(np.sqrt(len(repcost))))
            plt.figure(figsize=(8, 6))
            if nbins == 0:
                plt.hist(repcost, edgecolor='black')
            else:
                plt.hist(repcost, nbins, edgecolor='black')

            plt.title(f'All Cases - IML = {str(msa_imls[i])}', fontsize=11)
            plt.xlabel('Repair Cost - [\u20AC]')
            plt.ylabel('Frequency')
            plt.grid(True)
            fname = os.path.join(self.output_folder, f'hist_all_iml_{str(msa_imls[i])}.png')
            plt.savefig(fname)

            # Plot Cumulative Distribution Function for the IML
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(costvals, freq, lw=2)
            ax[0].set_title(f'Cost CDF - IML = {str(msa_imls[i])}', fontsize=11)
            ax[0].set_ylabel('Probability of Exceedance', fontsize=11)
            ax[0].grid(True)
            ax[1].plot(costvals, invfreq, lw=2, c='blue')
            ax[1].set_ylabel('P(COST>=cost)', fontsize=11)
            ax[1].set_xlabel('Repair Cost - [\u20AC]')
            ax[1].grid(True)
            fname = os.path.join(self.output_folder, f'cost_cdf_iml_{str(msa_imls[i])}.png')
            plt.savefig(fname)
            plt.close('all')

        # Plot Loss curves per IM Level
        plt.figure(figsize=(8, 6))
        for i in range(len(msa_imls)):
            loss_curve = comp_loss_info['loss_curve_iml' + str(i + 1)]
            loss = loss_curve[:, 0]
            pone = loss_curve[:, 1]
            plt.plot(loss, pone, label=f'IML = {str(msa_imls[i])}')

        plt.title(f'Cost Loss Curves per IM Level', fontsize=11)
        plt.xlabel('Repair Cost - [\u20AC]')
        plt.ylabel('P(COST>=cost)')
        plt.legend(frameon=False)
        plt.grid(True)
        fname = os.path.join(self.output_folder, 'cost_cdf.png')
        plt.savefig(fname)

        # Plot IM Level vs Expected Loss
        plt.figure(figsize=(8, 6))
        plt.plot(msa_imls, comp_loss, c='black', ls='-', label=r'$\mu_{L,total|IM}$')
        plt.plot(msa_imls, comp_loss_dir, c='red', ls='-', label=r'$\mu_{L,direct|IM}$')
        plt.plot(msa_imls, comp_loss_indir, c='blue', ls='-', label=r'$\mu_{L,indirect|IM}$')
        plt.title('Intensity Based Loss (Comprehensive) Curves', fontsize=11)
        plt.xlabel(f'Intensity Measure Level')
        plt.ylabel('Expected Loss [\u20AC]')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(self.output_folder, 'comp_iml_loss.png')
        plt.savefig(fname)

        # Plot IM Loss Curve
        plt.figure(figsize=(8, 6))
        plt.loglog(comp_loss, iml_mafe, label='$\mu_{L,total}$', c='black')
        plt.loglog(comp_loss_dir, iml_mafe, label='$\mu_{L,direct}$', c='red')
        plt.loglog(comp_loss_indir, iml_mafe, label='$\mu_{L,indirect}$', c='blue')
        plt.grid(True)
        plt.legend()
        plt.title('Loss Curves (Comprehensive)')
        plt.ylabel(r'Mean Annual Frequency of Exceedance, $\lambda_{IM}$')
        plt.xlabel('Monetary Loss [\u20AC]')
        ax = plt.gca()
        ax.text(0.02, 0.22, f'EAL,total = {float(c_eal):.0f}\u20AC', transform=ax.transAxes, fontsize=10, style='italic')
        ax.text(0.02, 0.16, f'EALR,total = {float(c_ealr):.3f}%', transform=ax.transAxes, fontsize=10, style='italic')
        ax.text(0.02, 0.1, f'EALR,direct = {float(c_ealr_dir):.3f}%', transform=ax.transAxes, fontsize=10, style='italic')
        ax.text(0.02, 0.04, f'EALR,indirect = {float(c_ealr_indir):.3f}%', transform=ax.transAxes, fontsize=10, style='italic')
        fname = os.path.join(self.output_folder, 'comp_loss.png')
        plt.savefig(fname)

        # Plot IM Loss Curve
        plt.figure(figsize=(8, 6))
        plt.loglog(comp_loss, iml_mafe, label='$\mu_{L,total}$', c='black')
        plt.loglog(per05_loss, iml_mafe, label='$5^{th}$ percentile', c='blue')
        plt.loglog(per14_loss, iml_mafe, label='$14^{th}$ percentile', c='green')
        plt.loglog(per50_loss, iml_mafe, label='$50^{th}$ percentile', c='yellow')
        plt.loglog(per86_loss, iml_mafe, label='$86^{th}$ percentile', c='orange')
        plt.loglog(per95_loss, iml_mafe, label='$95^{th}$ percentile', c='red')
        plt.grid(True)
        plt.xlim([min(comp_loss), max(per95_loss)])
        plt.legend()
        plt.title('Loss Curves (Comprehensive)')
        plt.ylabel(r'Mean Annual Frequency of Exceedance, $\lambda_{IM}$')
        plt.xlabel('Monetary Loss [\u20AC]')
        fname = os.path.join(self.output_folder, 'comp_loss_percentile.png')
        plt.savefig(fname)

        # Plot Loss Curve
        plt.figure(figsize=(8, 6))
        plt.loglog(exp_loss_lc, loss_mafe, label='$\mu_{L}$')
        plt.title('Loss Curve (Comprehensive)')
        plt.xlabel('Expected Loss [\u20AC]')
        plt.ylabel(r'Mean Annual Frequency of Exceedance, $\lambda_{Loss}$')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(self.output_folder, 'comp_loss_lc.png')
        plt.savefig(fname)

        # Plot comparison of loss as a function of IM Level
        plt.figure(figsize=(8, 6))
        plt.plot(msa_imls, comp_loss_dir, label=r'$\mu_{L|IM}$-Comprehensive')
        plt.plot(msa_imls, se_loss, label=r'$\mu_{L|IM}$-Simplified')
        plt.title('Comparison of Direct Losses')
        plt.xlabel(f'Intensity Measure Level')
        plt.ylabel('Monetary Loss [\u20AC]')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(self.output_folder, 'loss_iml_comparison.png')
        plt.savefig(fname)

        # Plot Comparison of Loss Curves
        plt.figure(figsize=(8, 6))
        plt.loglog(comp_loss_dir, iml_mafe, label=r'$\mu_{L|IM}$-Comprehensive')
        plt.loglog(se_loss, iml_mafe, label=r'$\mu_{L|IM}$-Simplified')
        plt.title('Comparison of Direct Losses')
        plt.ylabel(r'Annual Rate of Exceedance - $\lambda_{IM}$')
        plt.xlabel('Monetary Loss [\u20AC]')
        plt.legend()
        plt.grid(True)
        ax = plt.gca()
        ax.text(0.02, 0.1, f'C-EALR = {float(c_ealr_dir):.3f} %', transform=ax.transAxes, fontsize=10, style='italic')
        ax.text(0.02, 0.04, f'S-EALR = {float(s_ealr):.3f} %', transform=ax.transAxes, fontsize=10, style='italic')
        fname = os.path.join(self.output_folder, 'loss_mafe_comparison.png')
        plt.savefig(fname)
        plt.close('all')

    def save(self, obj_name='EzLoss'):
        """
        Details
        -------
        Saves the variables of the current object as pickle file.

        """

        # save some info as pickle
        path_obj = os.path.join(self.output_folder, obj_name + '.pkl')
        obj = vars(copy.deepcopy(self))  # use copy.deepcopy to create independent obj
        with open(path_obj, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_all(self, plot=1):
        """
        Details
        -------
        Performs the risk assessment, in other words, runs all the methods.

        """

        self.get_damage_models()
        self.get_fragility()
        self.get_loss_simplified()
        self.get_loss_comprehensive()
        if plot == 1:
            self.plot_damage_models()
            self.plot_fragility()
            self.plot_loss()
        self.save()


def _hazard_fit(iml_range, poe_data, iml_data, t_inv, fit_option=0):
    """
    Details
    -------
    Fits a hazard curve for the given data

    Parameters
    ----------
    iml_range: numpy.ndarray (1xm)
        Intensity measure levels of interest to fit
    poe_data: numpy.ndarray (1xn)
        Probability of exceedance of im level in investigation time (t_inv) to fit a curve
    iml_data: numpy.ndarray (1xn)
        Intensity measure levels to fit a curve
    t_inv: int, float
        Investigation time (yrs) e.g. 50
    fit_option: int
        way to fit the hazard curve
        0: use splines (default: linear) --> use this if you have PSHA results --> mean annual frequency of exceedance
        1: use splines (default: linear) --> use this if you have PSHA results --> probability of exceedance
        2: use eqn. (7) - Vamvastikos 2012 (SAC/FEMA) --> use this if you do not have PSHA results
        3: use eqn. (9) - Vamvastikos 2012 (SAC/FEMA) --> use this if you do not have PSHA results

    Returns
    -------
    Lambda: numpy.ndarray (1xm)
        Mean annual frequency of exceedance of intensity measure levels of interest
    poe: numpy.ndarray (1xm)
        Mean annual frequency of exceedance of intensity measure levels of interest in t_inv years
    """

    def first_order_model(s, k0, k1):
        H = k0 * np.exp((-k1) * np.log(s))
        return H

    def second_order_model(im, k0, k1, k2):
        H = k0 * np.exp((-k1) * np.log(im) + (-k2) * np.log(im) ** 2)
        return H

    Lambda_data = -np.log(1 - poe_data) / t_inv  # Mean Annual frequency of exceedance
    Lambda_fit = []

    if fit_option == 0:
        poe = np.exp(interp1d(np.log(iml_data), np.log(poe_data), kind='linear')(np.log(iml_range)))
        Lambda = -np.log(1 - poe) / t_inv

    if fit_option == 1:
        Lambda = interp1d(iml_data, Lambda_data, kind='linear')(iml_range)
        poe = 1 - (1 - Lambda) ** t_inv

    if fit_option == 2:
        coef_opt, coef_cov = curve_fit(first_order_model, iml_data, Lambda_data)
        k0, k1 = coef_opt  # hazard coefficients
        for i in range(len(iml_range)):
            Lambda_fit.append(first_order_model(iml_range[i], k0, k1))
        Lambda = np.asarray(Lambda_fit)
        poe = 1 - np.exp(-t_inv * np.asarray(Lambda))

    if fit_option == 3:
        coef_opt, coef_cov = curve_fit(second_order_model, iml_data, Lambda_data)
        k0, k1, k2 = coef_opt  # hazard coefficients
        for i in range(len(iml_range)):
            Lambda_fit.append(second_order_model(iml_range[i], k0, k1, k2))
        Lambda = np.asarray(Lambda_fit)
        poe = 1 - np.exp(-t_inv * np.asarray(Lambda))
        # poe = 1-(1-Lambda)**t_inv

    return Lambda, poe


def _get_abut_dm_dls():
    """

    Details
    -------
    The method obtains damage models corresponding to each damage limit states for the abutments.
    The matrix of abutment damage model definition is 2x2 matrix containing the median (first column) and
    logarithmic standard deviation that define the onset of DLS-1, DLS-2, DLS-3, and DLS-4 corresponding
    to the slight, moderate and extensive damage states.

    Notes
    -----
    Note also that these definitions are, by the time being, quite rough since it seems that
    there is no reliable data for their accurate definition.

    References
    ----------
    Perdomo, C., Abarca, A., & Monteiro, R. (2020). Estimation of Seismic Expected Annual
    Losses for Multi-Span Continuous RC Bridge Portfolios Using a Component-Level Approach. Journal of Earthquake
    Engineering, 1–27. https://doi.org/10.1080/13632469.2020.1781710

    Parameters
    ----------

    Returns
    -------
    abut_dmdls1: numpy.ndarray
        abutment damage limit state 1 matrix
    abut_dmdls2: numpy.ndarray
        abutment damage limit state 2 matrix
    abut_dmdls3: numpy.ndarray
        abutment damage limit state 3 matrix
    abut_dmdls4: numpy.ndarray
        abutment damage limit state 4 matrix

    """

    abut_dmdls1 = np.array([[0.015, 0.45] for _ in range(2)])
    abut_dmdls2 = np.array([[0.070, 0.50] for _ in range(2)])
    abut_dmdls3 = np.array([[0.120, 0.50] for _ in range(2)])
    abut_dmdls4 = np.array([[2.000, 0.10] for _ in range(2)])

    return abut_dmdls1, abut_dmdls2, abut_dmdls3, abut_dmdls4


def _get_pier_dm_dls(sections_data, bent_data, edp):
    """
    Details
    -------
    The method obtains damage models corresponding to each damage limit states for the piers.
    The matrix of pier damage model definition is nx2 matrix containing the median (first column) and
    logarithmic standard deviation that define the onset of DLS-i. i = 1,2,3,4 corresponds
    to the slight, moderate and extensive damage states.
    With regard to the damage models used herein:
    1) The damage models are generated for solid circular columns
    2) The damage models are generated for columns which behave as cantilever
    3) The damage models are generated for columns with rebars which are not corroded

    Notes
    -----
    Only the comprehensive damage models were used.

    References
    ----------
    Perdomo, C., & Monteiro, R. (2020). Simplified damage models for circular section
    reinforced concrete bridge columns. Engineering Structures, 217, 110794.
    https://doi.org/10.1016/j.engstruct.2020.110794

    Parameters
    ----------
    sections_data: dictionary
        dictionary containing bent sections' data, reinforcement, diameter, concrete strength etc per section.
    bent_data: dictionary
        dictionary containing bent data, pier heights, pier ductility, pier axial load, pier section etc per bent.
    edp: int
        Engineering Design Parameter used for pier damage model
        edp=1 for curvature ductility
        edp=2 for displacement ductility
        edp=3 for drift ratio

    Returns
    -------
    pier_dmdls1: numpy.ndarray
        pier damage limit state 1 matrix
    pier_dmdls2: numpy.ndarray
        pier damage limit state 2 matrix
    pier_dmdls3: numpy.ndarray
        pier damage limit state 3 matrix
    pier_dmdls4: numpy.ndarray
        pier damage limit state 4 matrix
    pier_dmdls4S: numpy.ndarray
        pier damage limit state 4 (shear) matrix
    """

    # Add the input the ground motion database to use
    pickle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'damage_models', "pier_dm.pkl")
    with open(pickle_file, "rb") as file:
        dm_circular = pickle.load(file)

    pier_dmdls1 = {}
    pier_dmdls2 = {}
    pier_dmdls3 = {}
    pier_dmdls4 = {}
    pier_dmdls4S = {}

    if edp == 1:
        pier_dmdls4S = None
        edp_type = 'CurvatureDuctility'
    elif edp == 2:
        edp_type = 'DisplacementDuctility'
    elif edp == 3:
        edp_type = 'DriftRatio'

    for bent_tag in bent_data:
        pier_dmdls1[bent_tag] = np.zeros([len(bent_data[bent_tag]), 2])
        pier_dmdls2[bent_tag] = np.zeros([len(bent_data[bent_tag]), 2])
        pier_dmdls3[bent_tag] = np.zeros([len(bent_data[bent_tag]), 2])
        pier_dmdls4[bent_tag] = np.zeros([len(bent_data[bent_tag]), 2])
        pier_dmdls4S[bent_tag] = np.zeros([len(bent_data[bent_tag]), 2])
        for nP, pier_tag in enumerate(bent_data[bent_tag]):
            pier = bent_data[bent_tag][pier_tag]
            section = sections_data[pier['section']]
            if section['Type'] != 'Solid Circular':
                raise ValueError('Damage models for sections other than solid circular are not defined yet..')
            elif section['Type'] == 'Solid Circular':
                # Pier properties
                P = pier['AxialForce']
                ds = section['D'] - 2 * section['cover']  # Core diameter
                Acc = np.pi * ds ** 2 / 4  # Area of core of section enclosed by the center lines of the perimeter hoop
                Asl = section['number of bars'] * (
                        np.pi * section['dl'] ** 2) / 4  # Total area of longitudinal steel reinforcements
                Ag = np.pi * section['D'] ** 2 / 4  # Total area of concrete section
                rho_v = 4 * (np.pi * section['dh'] ** 2 / 4) / (ds * section['sh'])  # Ratio of the volume of transverse
                rho_vconf = max(0.45 * (Ag / Acc - 1) * section['Fce'] / section['Fyhe'],
                                0.12 * section['Fce'] / section['Fyhe'])  # code requirement by AASHTO

                # Required parameters to determine damage model
                D_pier = section['D']  # Pier diameter
                H_pier = pier['H']  # Pier height
                alr = P / (Ag * section['Fce'])  # Axial load ratio
                rho_long = Asl / Ag  # Ratio of area of longitudinal reinforcement to area of gross section
                rho_trans = rho_v / rho_vconf  # Ratio of available transverse reinforcement ratio to code requirement

                # The available values in the damage model database
                D_avail = np.unique(dm_circular[edp_type]['DLS-1']['D'])
                alr_avail = np.unique(dm_circular[edp_type]['DLS-1']['ALR'])
                rhol_avail = np.unique(dm_circular[edp_type]['DLS-1']['rho_l'])
                rhot_avail = np.unique(dm_circular[edp_type]['DLS-1']['rho_v/rho_v,conf'])

                # Find the position of values in the database which are closest to actual values
                test_D = abs(D_avail - D_pier)
                min_test_D = min(test_D)
                pos_test_D = max(np.where(test_D == min_test_D)[0])
                diam_dm = D_avail[pos_test_D]

                test_alr = abs(alr_avail - alr)
                min_test_alr = min(test_alr)
                pos_test_alr = max(np.where(test_alr == min_test_alr)[0])
                alr_dm = alr_avail[pos_test_alr]

                test_rhol = abs(rhol_avail - rho_long)
                min_test_rhol = min(test_rhol)
                pos_test_rhol = max(np.where(test_rhol == min_test_rhol)[0])
                rhol_dm = rhol_avail[pos_test_rhol]

                test_rhot = abs(rhot_avail - rho_trans)
                min_test_rhot = min(test_rhot)
                pos_test_rhot = max(np.where(test_rhot == min_test_rhot)[0])
                rhot_dm = rhot_avail[pos_test_rhot]

                # Damage model for shear failure mechanism
                if edp != 1:
                    # Ratio of steel yield strength to unconfined concrete compressive strength
                    fyfc = section['Fyle'] / section['Fce']
                    # Ratio of pier height to pier diameter
                    hd = H_pier / D_pier
                    hd_avail = np.unique(dm_circular[edp_type]['DLS-4S']['H/D'])
                    fyfc_avail = np.unique(dm_circular[edp_type]['DLS-4S']['fy/fc'])

                    if hd < min(hd_avail):
                        hd_dm1 = hd_avail[0]
                        hd_dm2 = hd_avail[1]
                        hd = hd_avail[0]
                    elif hd > max(hd_avail):
                        hd_dm1 = hd_avail[-2]
                        hd_dm2 = hd_avail[-1]
                        hd = hd_avail[-1]
                    else:
                        pos_hd_dm1 = min(np.where(hd_avail >= hd)[0])
                        pos_hd_dm2 = max(np.where(hd_avail < hd)[0])
                        hd_dm1 = hd_avail[pos_hd_dm1]
                        hd_dm2 = hd_avail[pos_hd_dm2]

                    if fyfc < min(fyfc_avail):
                        fyfc_dm1 = fyfc_avail[0]
                        fyfc_dm2 = fyfc_avail[1]
                        fyfc = fyfc_dm1 + 0.001
                    elif fyfc > max(fyfc_avail):
                        fyfc_dm1 = fyfc_avail[-2]
                        fyfc_dm2 = fyfc_avail[-1]
                        fyfc = fyfc_dm1 + 0.001
                    else:
                        pos_fyfc_dm1 = min(np.where(fyfc_avail >= fyfc)[0])
                        pos_fyfc_dm2 = max(np.where(fyfc_avail < fyfc)[0])
                        fyfc_dm1 = fyfc_avail[pos_fyfc_dm1]
                        fyfc_dm2 = fyfc_avail[pos_fyfc_dm2]

                    # Determine damage model parameters for shear failure - DLS-4S
                    mask_dm = (dm_circular[edp_type]['DLS-4S']['D'] == diam_dm) * \
                              (dm_circular[edp_type]['DLS-4S']['ALR'] == alr_dm) * \
                              (dm_circular[edp_type]['DLS-4S']['rho_l'] == rhol_dm) * \
                              (dm_circular[edp_type]['DLS-4S']['rho_v/rho_v,conf'] == rhot_dm)

                    mask_dm11 = (dm_circular[edp_type]['DLS-4S']['H/D'] == hd_dm1) * \
                                (dm_circular[edp_type]['DLS-4S']['fy/fc'] == fyfc_dm1)
                    pos_dm11 = [i for i, x in enumerate(mask_dm * mask_dm11) if x]
                    theta_dls4s_11 = float(dm_circular[edp_type]['DLS-4S']['theta'].iloc[pos_dm11])
                    beta_dls4s_11 = float(dm_circular[edp_type]['DLS-4S']['beta'].iloc[pos_dm11])

                    mask_dm12 = (dm_circular[edp_type]['DLS-4S']['H/D'] == hd_dm1) * \
                                (dm_circular[edp_type]['DLS-4S']['fy/fc'] == fyfc_dm2)
                    pos_dm12 = [i for i, x in enumerate(mask_dm * mask_dm12) if x]
                    theta_dls4s_12 = float(dm_circular[edp_type]['DLS-4S']['theta'].iloc[pos_dm12])
                    beta_dls4s_12 = float(dm_circular[edp_type]['DLS-4S']['beta'].iloc[pos_dm12])

                    mask_dm21 = (dm_circular[edp_type]['DLS-4S']['H/D'] == hd_dm2) * \
                                (dm_circular[edp_type]['DLS-4S']['fy/fc'] == fyfc_dm1)
                    pos_dm21 = [i for i, x in enumerate(mask_dm * mask_dm21) if x]
                    theta_dls4s_21 = float(dm_circular[edp_type]['DLS-4S']['theta'].iloc[pos_dm21])
                    beta_dls4s_21 = float(dm_circular[edp_type]['DLS-4S']['beta'].iloc[pos_dm21])

                    mask_dm22 = (dm_circular[edp_type]['DLS-4S']['H/D'] == hd_dm2) * \
                                (dm_circular[edp_type]['DLS-4S']['fy/fc'] == fyfc_dm2)
                    pos_dm22 = [i for i, x in enumerate(mask_dm * mask_dm22) if x]
                    theta_dls4s_22 = float(dm_circular[edp_type]['DLS-4S']['theta'].iloc[pos_dm22])
                    beta_dls4s_22 = float(dm_circular[edp_type]['DLS-4S']['beta'].iloc[pos_dm22])

                    if theta_dls4s_11 == 0:
                        theta_dls4s_11 = 100
                        beta_dls4s_11 = 0.9

                    if theta_dls4s_12 == 0:
                        theta_dls4s_12 = 100
                        beta_dls4s_12 = 0.9

                    if theta_dls4s_21 == 0:
                        theta_dls4s_21 = 100
                        beta_dls4s_21 = 0.9

                    if theta_dls4s_22 == 0:
                        theta_dls4s_22 = 100
                        beta_dls4s_22 = 0.9

                    points = np.array([[hd_dm1, fyfc_dm1], [hd_dm1, fyfc_dm2], [hd_dm2, fyfc_dm1], [hd_dm2, fyfc_dm2]])
                    theta_values = np.array([[theta_dls4s_11], [theta_dls4s_12], [theta_dls4s_21], [theta_dls4s_22]])
                    beta_values = np.array([[beta_dls4s_11], [beta_dls4s_12], [beta_dls4s_21], [beta_dls4s_22]])
                    theta_dls4s = float(griddata(points, theta_values, (hd, fyfc)))
                    beta_dls4s = float(griddata(points, beta_values, (hd, fyfc)))
                    if np.isnan(theta_dls4s) or np.isnan(beta_dls4s):
                        theta_dls4s = 1
                        beta_dls4s = 0.9

                    pier_dmdls4S[bent_tag][nP, 0] = theta_dls4s
                    pier_dmdls4S[bent_tag][nP, 1] = beta_dls4s

                # Determine damage model parameters for other DLS (Flexural Failure)
                for i in range(1, 5):
                    pierdls_name = 'pier_dmdls' + str(i)
                    mask_dm = (dm_circular[edp_type]['DLS-' + str(i)]['D'] == diam_dm) * \
                              (dm_circular[edp_type]['DLS-' + str(i)]['ALR'] == alr_dm) * \
                              (dm_circular[edp_type]['DLS-' + str(i)]['rho_l'] == rhol_dm) * \
                              (dm_circular[edp_type]['DLS-' + str(i)]['rho_v/rho_v,conf'] == rhot_dm)

                    if edp == 1:
                        pos_dm = [i for i, x in enumerate(mask_dm) if x]
                        theta_dls = dm_circular[edp_type]['DLS-' + str(i)]['theta'].iloc[pos_dm, :]
                        beta_dls = dm_circular[edp_type]['DLS-' + str(i)]['beta'].iloc[pos_dm, :]

                    elif edp in [2, 3]:
                        mask_dm1 = dm_circular[edp_type]['DLS-' + str(i)]['H/D'] == hd_dm1
                        pos_dm1 = [i for i, x in enumerate(mask_dm * mask_dm1) if x]
                        theta_dls1 = float(dm_circular[edp_type]['DLS-' + str(i)]['theta'].iloc[pos_dm1])
                        beta_dls1 = float(dm_circular[edp_type]['DLS-' + str(i)]['beta'].iloc[pos_dm1])
                        mask_dm2 = dm_circular[edp_type]['DLS-' + str(i)]['H/D'] == hd_dm2
                        pos_dm2 = [i for i, x in enumerate(mask_dm * mask_dm2) if x]
                        theta_dls2 = float(dm_circular[edp_type]['DLS-' + str(i)]['theta'].iloc[pos_dm2])
                        beta_dls2 = float(dm_circular[edp_type]['DLS-' + str(i)]['beta'].iloc[pos_dm2])
                        theta_dls = interp1d([hd_dm1, hd_dm2], [theta_dls1, theta_dls2])(hd)
                        beta_dls = interp1d([hd_dm1, hd_dm2], [beta_dls1, beta_dls2])(hd)

                    eval(pierdls_name)[bent_tag][nP, 0] = theta_dls
                    eval(pierdls_name)[bent_tag][nP, 1] = beta_dls

    return pier_dmdls1, pier_dmdls2, pier_dmdls3, pier_dmdls4, pier_dmdls4S


def _get_bearing_dm_dls(bearing_data):
    """
    Details
    -------
    The method obtains damage models corresponding to each damage limit states for the bearings.
    The bearing damage model definition is nx2 matrix containing the median (first column) and
    logarithmic standard deviation that define the onset of DLS-i. i = 1,2,3,4 corresponds
    to the slight, moderate and extensive damage states.

    Notes
    -----
    The damage models are retrieved for thin neoprene pad bearings:
        DLS-1; sliding (mu*P)/(G*A/L) + impact on shear key (50mm)
        DLS-2; excessive sliding --> deck realignment (150mm, from Nielson 2005)
        DLS-3; deck falls from pedestal --> bearing replacement + jacking up the bridge deck (300mm, from geometry)
        DLS-4; unseating -> based on the geometry (550mm, from geometry)

    Damage models for other specific types can be added here.

    References
    ----------
    Nielson, B. G. (2005). Analytical fragility curves for highway bridges in moderate seismic zones.
    Environmental Engineering, December, 400. https://doi.org/10.1016/j.engstruct.2017.03.041

    Parameters
    ----------
    bearing_data: dictionary
        dictionary containing bearing data

    Returns
    -------
    bearing_dmdls1: numpy.ndarray
        bearing damage limit state 1 matrix
    bearing_dmdls2: numpy.ndarray
        bearing damage limit state 2 matrix
    bearing_dmdls3: numpy.ndarray
        bearing damage limit state 3 matrix
    bearing_dmdls4: numpy.ndarray
        bearing damage limit state 4 matrix
    """

    bearing_dmdls1 = {}
    bearing_dmdls2 = {}
    bearing_dmdls3 = {}
    bearing_dmdls4 = {}
    # get units to prepare the input with m, kN, sec
    m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units(0)
    # Can make this bearing specific in the future.
    for joint in bearing_data:
        e_uns = bearing_data[joint][0][5]
        bearing_dmdls1[joint] = np.array([[50 * mm, 0.3] for _ in range(len(bearing_data[joint]))])
        bearing_dmdls2[joint] = np.array([[150 * mm, 0.2] for _ in range(len(bearing_data[joint]))])
        bearing_dmdls3[joint] = np.array([[300 * mm, 0.2] for _ in range(len(bearing_data[joint]))])
        bearing_dmdls4[joint] = np.array([[550 * mm, 0.1] for _ in range(len(bearing_data[joint]))])

    return bearing_dmdls1, bearing_dmdls2, bearing_dmdls3, bearing_dmdls4
