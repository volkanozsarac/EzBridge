from EzBridge import EzBridge
from EzBridge.Utility import Get_T0, RunTime

T0 = Get_T0()

# get units to prepare the input with m, kN, sec
m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = EzBridge.units()

################################################
############## INPUT STARTS HERE ###############
################################################

#  ----------------------------------------------------------------------------
#  INPUT for model generation 
#  ----------------------------------------------------------------------------
inputs = {
    'Mass': 'Lumped',

    'Deck': {
        'Type': 'Discontinuous',  # ('Continuous', 'Discontinuous')
        'Xs': [0, 30 * m, 60 * m],
        'Ys': [0, 0, 0, 0],
        'Zs': [10, 10 * m, 10 * m],
        'A': 9 * m ** 2,
        'Iy': 2.01383 * m ** 4,
        'Iz': 140.07 * m ** 4,
        'J': 0.5 * m ** 4,
        'E': 33500 * MPa,
        'G': 33500 * MPa / (2 * (1 + 0.2)),
        'wSDL': 8.5 * kN / m,
        'numEle': 7,
    },

    'LinkSlab': {  # note: Iy is reduced to 20%, Iz, J are reduced to 50%
        'cond': [1],  # if 1 link slab exist on joint, if 0 not
        'd': 0.4 * m,  # from deck centroid to link centroid
        'L': 1 * m,
        't': 0.2 * m,
        'w': 14 * m,
        'E': 31225 * MPa,
        'G': 31225 * MPa / (2 * (1 + 0.2))
    },

    'Bearing': {
        'dv': 0.85 * m,  # vertical distance from deck centroid to bearing top
        'dh': 0.93 * m,  # horizontal distance frodm cap cetroid to bearing
        'h': 11.5 * cm,  # height of bearings

        # Distributed bearing approach
        'N': 10,  # underneath each span end
        's': 1310 * mm,  # spacing of bearings (if more than 1)
        'Type': 'Elastic',  # Type of bearing
        'kx': 1755 * kN / m,
        'ky': 1e8 * kN / m,
        'kz': 1143752 * kN / m,
        'krx': 10 * kN / m,
        'kry': 10 * kN / m,
        'krz': 10 * kN / m

        # Lumped bearing approach
        # 'N'  : 1,
        # 's'  : 0*mm,
        # 'Type': 'Elastic',
        # 'kx' :10*1755*kN/m,
        # 'ky' : 1e8*kN/m, # Assume that there is non-sacrifical shear keys
        # 'kz' : 7*51200*kN/m,
        # 'krx': 1e10*kN/m,
        # 'kry': 10*kN/m,
        # 'krz': 1e10*kN/m

        # This is like assigning a release
        # 'N'  : 1,
        # 's'  : 0*mm,  
        # 'Type': 'Elastic',         
        # 'kx' : 1e10*kN/m,
        # 'ky' : 1e10*kN/m,
        # 'kz' : 1e10*kN/m,
        # 'krx': 1e10*kN/m,
        # 'kry': 10*kN/m,
        # 'krz': 1e10*kN/m

        # Elastomeric bearings
        # 'Type': 'ElastomericX',
        # 'Fy': 150*kN,
        # 'alpha': 0.15,
        # 'G': 0.8*MPa,
        # 'K': 2*GPa,
        # 'D1': 0*cm,
        # 'D2': 50*cm,
        # 'ts': 1*mm,
        # 'tr': 4*mm,
        # 'n': 40

        # Elastomeric bearings
        # 'Type': 'elastomericBearingBoucWen',  
        # 'Kvert': 1e7,
        # 'Kinit': 1000*kN/m,
        # 'Fb': 150*kN,
        # 'alpha1': 0.4,
        # 'alpha2': 0,
        # 'mu':1

        # Friction Type Isolators
        # 'N'  : 3,
        # 's'  : 3*m,
        # 'Type': 'singleFPBearing_coulomb',
        # 'R': 3.4*m,
        # 'mu': 0.03,
        # 'K': 1e7*kN
    },

    'BentCap': {
        'L': 12 * m,  # From deck centroid to bearing top
        'w': 2.6 * m,  # From deck centroid to bearing top
        'h': 2 * m,  # Height of pier cap
        'E': 0.5 * 31225 * MPa,
        'G': 0.5 * 31225 * MPa / (2 * (1 + 0.2))
    },

    'Bent': {
        'N': 2,  # Number of Piers per Bent
        'Dist': 8 * m,  # Spacing between the Piers
        'H': [12 * m],  # Pier Heights
        'Sections': [1],  # Define section id used for each bent (start from 1)
        'EleType': 1,
        # Fully Elastic element:
        # 0: Legendre - section
        # Lumped Plasticity (elastic interior):
        # 1: HingeRadau (springs)
        # 2: HingeRadau (fiber section)
        # Distributed plasticity (inelastic interior):
        # 3: HingeRadau (fiber sections)
        # 4: Lobatto with 5 gauss points (fiber sections)
        # Sections
        'RF': 0.5,  # Modulus reduction factor for EleType 0 case
        'SectionTypes': 'Circ',  # Define section tags used for each bent
        'D': [2.8 * m],  # Diameter of the circular column section
        'Fce': [39 * MPa],  # Expected nominal compressive strength of the concrete material
        'Fyle': [462 * MPa],  # Expected yield strength of longitudinal rebars
        'Fyhe': [462 * MPa],  # Expected yield strength of transverse rebars
        'cover': [50 * mm],  # Clear cover to centroid of stirrups
        'numBars': [92],  # Number of longitudinal bars
        'dl': [40 * mm],  # Nominal diameter of longitudinal bars
        'dh': [20 * mm],  # Nominal diameter of transversal bars
        's': [100 * mm],  # Vertical spacing between the centroid of spirals or hoops
        'TransReinfType': ['Hoops']  # Type of transversal steel, 'Hoops' or 'Spirals'
    },

    'Abutment': {
        # 'Type': 'Fixed',  # Abutment Type ('Fixed', 'SDC 2019')
        'Type': 'SDC 2019', # ('Fixed', 'SDC 2019')
        'spring': 2, # 2 or 1
        'gap': 100*mm, # in longitudinal direction
        'gapFlag': 0, # in longitudinal direction
        'height': 6.55*m,
        'width': 14*m,
        'breadth': 1*m,
    },

    'Foundation': {
        # 'Type': 'Fixed',  # Foundation Type (Fixed, Macro elements, Pile-Shaft, Group Pile)
        
        # 'Type': 'Macro Elements',  # Foundation Type (Fixed, Macro elements, Pile-Shaft, Group Pile)
        # 'Thickness': 1*m,  # Foundation Type (Fixed, Macro elements, Pile-Shaft, Group Pile)
        # 'Kx': [147.06e3*kN/m],  #  Elastic Spring Property, -dof X
        # 'Ky': [150.83e3*kN/m],  #  Elastic Spring Property, -dof Y
        # 'Kz': [2439.02e3*kN/m],  #  Elastic Spring Property, -dof Z
        # 'Krx': [50000.00e3*kN/m],  #  Elastic Spring Property, -dof rX
        # 'Kry': [12500.00e3*kN/m],  #  Elastic Spring Property, -dof rY
        # 'Krz': [50000.00e3*kN/m]   #  Elastic Spring Property, -dof rZ
        
        # 'Type': 'Pile-Shaft',  # Foundation Type (Fixed, Macro elements, Pile-Shaft, Group Pile)
        # 'EleType': 1,             # 0: Displacement-Based Beam column element with elastic section
        # # 1: Displacement-Based Beam column element with inelastic fiber section
        # # - Gauss Legendre with 2 gauss points
        # # 2: Force-Based Beam column element with inelastic fiber section
        # # - Gauss Lobatto with 3 gauss points
        # 'RF': 0.5,  # Modulus reduction factor for EleType 0 case
        # 'Sections': [1],  # Define section id used for piles underneath each bent (start from 1)
        # 'D': [3.2 * m],  # Pile diameter
        # 'Fce': [39 * MPa],  # Expected nominal compressive strength of the concrete material
        # 'Fyle': [462 * MPa],  # Expected yield strength of longitudinal rebars
        # 'Fyhe': [462 * MPa],  # Expected yield strength of transverse rebars
        # 'cover': [50 * mm],  # Clear cover to centroid of stirrups
        # 'numBars': [96],  # Number of longitudinal bars
        # 'dl': [40 * mm],  # Nominal diameter of longitudinal bars
        # 'dh': [20 * mm],  # Nominal diameter of transversal bars
        # 's': [100 * mm],  # Vertical spacing between the centroid of spirals or hoops
        # 'TransReinfType': ['Hoops'],  # Type of transversal steel, 'Hoops' or 'Spirals'

        'Type': 'Group Pile',  # Foundation Type (Fixed, Macro elements, Pile-Shaft, Group Pile)
        'EleType': 2,
        # 0: Displacement-Based Beam column element with elastic section
        # 1: Displacement-Based Beam column element with inelastic fiber section
        # - Gauss Legendre with 2 gauss points
        # 2: Force-Based Beam column element with inelastic fiber section
        # - Gauss Lobatto with 3 gauss points
        'RF': 0.5,  # Modulus reduction factor for EleType 0 case
        'Sections': [1],  # Define section id used for piles underneath each bent (start from 1)
        'D': [1 * m],  # Pile diameter
        'Fce': [39 * MPa],  # Expected nominal compressive strength of the concrete material
        'Fyle': [462 * MPa],  # Expected yield strength of longitudinal rebars
        'Fyhe': [462 * MPa],  # Expected yield strength of transverse rebars
        'cover': [50 * mm],  # Clear cover to centroid of stirrups
        'numBars': [30],  # Number of longitudinal bars
        'dl': [30 * mm],  # Nominal diameter of longitudinal bars
        'dh': [16 * mm],  # Nominal diameter of transversal bars
        's': [100 * mm],  # Vertical spacing between the centroid of spirals or hoops
        'TransReinfType': ['Hoops'],  # Type of transversal steel, 'Hoops' or 'Spirals'
        'cap_t': 2 * m,  # pile cap thickness
        'cap_A': 50 * m ** 2,  # pile cap area
        'nx': [2],  # number of rows in x dir
        'ny': [4],  # number of rows in y dir
        'sx': [3 * m],  # spacing in x dir
        'sy': [3 * m],  # spacing in y dir
    }
}

#  ----------------------------------------------------------------------------
#  INPUT for single NRHA: 
#  ----------------------------------------------------------------------------
# GMs = ['RSN1158_KOCAELI_DZC270.AT2', 'RSN1158_KOCAELI_DZC180.AT2', 'RSN1158_KOCAELI_DZC-UP.AT2']
GMs = ['RSN1158_KOCAELI_DZC270.txt', 'RSN1158_KOCAELI_DZC180.txt', 'RSN1158_KOCAELI_DZC-UP.txt']
GMcomponents = [1, 2]
GMfactors = [1, 1]
signal = '-accel'

# GMs = ['RSN1158_KOCAELI_DZC270.AT2']
# GMcomponents = [1]
# GMfactors=[1.0]

#  ----------------------------------------------------------------------------
#  INPUT for MSA (Multiple Stripes Analysis):
#  ----------------------------------------------------------------------------
gm_msa = {
    'Folder': 'P0',
    # GMFiles/MSA/Folder --> this is the folder containing input records for MSA, P0 accounts for processor name that is being used
    'MSAcomponents': [1, 2, 3],  # Ground motion components to use in the analysis
    'dts_file': "GMR_dts.txt",  # Time steps of ground motions to run for uniform excitation case
    'gm_H1_names_file': "GMR_H1_names.txt",  # Names of ground motions to run (dir 1) for uniform excitation case
    'gm_H2_names_file': "GMR_H2_names.txt",  # Names of ground motions to run (dir 2) for uniform excitation case
    'gm_V_names_file':  "GMR_V_names.txt",   # Names of ground motions to run (dir 3) for uniform excitation case
    'gm_multi_support': "GMR_multi_support.txt" # GMR folder names for multi-support excitation case
}

#  INPUT for IDA (incremental Dynamic Analysis): 
#  ----------------------------------------------------------------------------

# set the ground motion records to be used for each processor
# gmFol = os.path.join('IDA','P0') 

# IDAdir = 2 # Directions to apply ground motions
# im = {'im': 'SaT', 'Tstar': 1.4, 'xi': 0.05}
# htf = {'num_runs': 2,
#         'iml_init': 0.05,            # initial hunt ratio
#         'iml_hunt_incr': 0.20,       # hunt increment ratio
#         'iml_trace_incr': 0.10,      # trace increment ratio
#         'iml_min_trace': 0.0,       # minimum trace incr ratio
#         }
# gm_ida = {'gm_names_file':  "GMR_names.txt", # Names of ground motions to run
#           'dts_file':       "GMR_dts.txt",   # Time steps of ground motions to run
#           'durs_file':      "GMR_durs.txt",  # Durations of the ground motions to run
#          }

################################################
############### INPUT ENDS HERE ################
################################################

# Start the bridge object
Bridge = EzBridge.Main(model=inputs)

# Generate the model
Bridge.generate_model()

# Calculates cost etc.
Bridge.get_summary()

# Plot the model
Bridge.plot_model(show_node='yes')
Bridge.plot_sections()

# Modal analysis
# Bridge.modal(numEigen = 10)

# Plot mode shapes
# Bridge.plot_modeshape(1,200)
# Bridge.plot_modeshape(2,200)
# Bridge.plot_modeshape(3,200)

# Set some analysis configurations
# Bridge.analysis_config('Penalty', 'RCM', 'UmfPack')

# Gravity analysis
Bridge.gravity(pflag=1, load_type=1)

# Plot deformed shape
# Bridge.plot_deformedshape(scale = 50)

# Response spectrum analysis
# Bridge.rsa('Target_Spectrum.txt', 10, 0.02, 1)

# Activate user define recorders
# Bridge.set_recorders()

# Animation option for nspa and nrha
Bridge.animation_config(animate=1, Movie=0, FrameStep=5, scale=10, fps=50)

# Pushover analysis
# Bridge.nspa(scheme = 'Uniform', PushOption = 1, 
# ctrlNode = 2001, IOflag = 0, ctrlDOF = 2)

# Nonlinear response history analysis
# Bridge.nrha(excitation = 'Uniform' , signal = signal, GMs = GMs, GM_components=GMcomponents, GM_factors=GMfactors, DtFactor = 1.0, pFlag=1, GMdt = 0.005)
Bridge.nrha(excitation = 'Multi-Support', signal = signal, GMs = GMs, GM_components=GMcomponents, GM_factors=GMfactors, DtFactor = 1.0, pFlag=1, GMdt = 0.005)

# Multiple-stripe analysis
# Bridge.msa(gm_msa, excitation = 'Multi-Support', signal = signal)
# Bridge.msa(gm_msa, excitation = 'Uniform', signal = signal)

# Incremental dynamic analysis
# Bridge.ida_htf(htf, gm_ida, im, gmFol, IDAdir)

# Wipe the model
Bridge.wipe_model()

print(RunTime(T0))
