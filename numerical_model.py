from EzBridge import smat
from EzBridge.utils import get_current_time, get_run_time, get_units
import numpy as np
import pickle

T0 = get_current_time()

# get units to prepare the input with m, kN, sec
m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units(1)

# Bearing Realizations
realizations = np.loadtxt('realizations.txt')

# Geometry of the bearing
Lb = 700 * mm  # length of the bearing in longitudinal direction
Wb = 500 * mm  # width of the bearing in transverse direction
Hb = 20 * mm  # total height of the bearing

for i in range(100):

    # Material properties obtained via realizations
    Eb = 30 * ksi  # bearing young's modulus (from AASHTO)
    mub = realizations[i, 1]  # friction coefficient between rubber and concrete (from realization)
    Gb = realizations[i, 0] * MPa  # shear modulus of the rubber (from realization)

    # Mechanical properties
    Kh = Gb * (Wb * Lb) / Hb
    Kv = (30 * ksi) * (Wb * Lb) / Hb  # assume 30ksi for E (from AASHTO)
    Krotz = (30 * ksi) * (1/12 * Wb**3 * Lb) / Hb  # assume 30ksi for E
    Kroty = (30 * ksi) * (1/12 * Wb * Lb**3) / Hb  # assume 30ksi for E
    Ktor = (1.0 * ksi) * 0.18 * (Wb * Lb**3) / Hb  # assume 1 ksi for G

    ################################################
    ############## INPUT STARTS HERE ###############
    ################################################

    #  ----------------------------------------------------------------------------
    #  INPUT for model generation 
    #  ----------------------------------------------------------------------------
    input_data = {

        # Update bearing volume
        'bearing_volume': Lb * Wb * Hb,

        # General information to construct bridge model.
        'General': {

            # Type of mass matrix to use ('Lumped', 'Consistent')
            'Mass': 'Lumped',
            # Maximum deck element length per span
            'Deck Element Discretization': 8,
            # maximum pier element length
            'Pier Element Discretization': 8,
            # Superimposed dead load per length on span elements
            'wSDL': 0,
            # Define pier element type
            # 0: Elastic-Gross, 1: Elastic-Cracked, 2: Inelastic-Fiber-Lobatto, 3: Inelastic-Fiber-HingeRadau
            'Pier Element Type': 3,
            # Joint coordinates, x
            'Joint Xs': [0, 32, 65, 98, 131, 163],
            # Joint coordinates, y
            'Joint Ys': [0, 0, 0, 0, 0, 0],
            # Joint coordinates, z
            'Joint Zs': [0, 0, 0, 0, 0, 0],
            # Definition of span to span joint connections
            'Connections': ['con2', 'con2', 'con2', 'con2'],
            # Configuration of deck elements per span
            'Decks': ['deck1', 'deck1', 'deck1', 'deck1', 'deck1'],
            # Configuration of bearing elements at each abutment to span joint and span to span joint 
            # for example: abutment1-span1, span1-span2, span2-span3, span3-span4, span4-abutment2.
            'Bearings': ['bconf1', 'bconf2', 'bconf2', 'bconf2', 'bconf2', 'bconf1'],
            # Configuration of bentcaps at each span to span joint
            # for example: span1-span2, span2-span3, span3-span4.
            'Bent Caps': ['cap1', 'cap1', 'cap1', 'cap1'],
            # Configuration of bents at each span to span joint
            # for example: span1-span2, span2-span3, span3-span4.
            'Bents': ['bent1', 'bent2', 'bent3', 'bent4'],
            # Configuration of backfill soil springs at each abutment
            'Abutments': 'ab2',
            # Configuration of shear keys
            'Shear Keys': 'sk1',
        },

        #  ----------------------------------------------------------------------------
        #  EXAMPLE OF JOINT PROPERTY DEFINITIONS
        #  ----------------------------------------------------------------------------
        # deck is discontinuous
        'con2': {
            'Type': 'Discontinuous',
            'L': 80 * mm,  # length of the gap two between span ends on top of bents
            'w': 12.5 * m   # deck width which is used create rigid link nodes to simulate pounding between deck ends
        },

        #  ----------------------------------------------------------------------------
        #  EXAMPLE OF DECK A SECTION DEFINITION
        #  ----------------------------------------------------------------------------
        # z
        # |_y
        # Define properties of deck section with tag 'd1'
        'deck1': {
            'A': 10.185 * m ** 2,  # area
            'Iy': 4.4046 * m ** 4,  # Moment of inertia in y-dir
            'Iz': 135.15 * m ** 4,  # Moment of inertia in z-dir
            'J': 0.96885 * m ** 4,  # Polar moment of inertia
            'E': 30310.0 * MPa,  # Young's modulus
            'G': 12629.0 * MPa,  # Shear modulus
            'dv': 0.736 * m  # vertical distance from deck centroid to deck bottom
        },

        #  ----------------------------------------------------------------------------
        #  EXAMPLE BEARING CONFIGURATION AND BEARING TYPE DEFINITIONS
        #  ----------------------------------------------------------------------------
        # at abutment to span joints or continuous span-to-span joints
        'bconf1': {
            'dh': 0.95 * m,  # horizontal distance between the centre point of all bearings and span ends
            'spacing': 2500 * mm,  # constant horizontal spacing between bearings
            'h': Hb,  # constant bearing heights
            # list of bearing types underneath the span end
            'bearings': ['b1', 'b1', 'b1', 'b1', 'b1'],
        },

        # at span-to-span joints which are not continuous, there are bearings on both sides of joint
        'bconf2': {
            'dh': 0.91 * m,  # horizontal distance between the centre point of all bearings and span ends
            'spacing': 2500 * mm,  # constant horizontal spacing between bearings
            'h': Hb,  # constant bearing heights
            # list of bearing types underneath the span located on left side of the joint
            'bearingsL': ['b1', 'b1', 'b1', 'b1', 'b1'],
            # list of bearing types underneath the span located on right side of the joint
            'bearingsR': ['b1', 'b1', 'b1', 'b1', 'b1'],
        },

        'b1': {
            'model': 'flatSliderBearing',
            'weight': 0 * kN,  # bearing weight + concrete block under bearing
            # Materials or coefficients for bearing model
            'dir-1': 'mat1',  # translation along the vertical direction of bridge
            'dir-4': 'mat4',  # rotation about the vertical direction of bridge
            'dir-5': 'mat2',  # rotation about the transverse direction of bridge
            'dir-6': 'mat3',  # rotation about the longitudinal direction of bridge
            'k_init': Kh,
            # initial elastic stiffness in local shear direction
            'friction_model': 'frn1'  # friction model used by the bearing
        },

        #  ----------------------------------------------------------------------------
        #  MATERIAL DEFINITIONS USED BY BEARING ELEMENTS
        #  ----------------------------------------------------------------------------
        # https://openseespydoc.readthedocs.io/en/latest/src/uniaxialMaterial.html
        # MatTag: [Material Type, Material Arguments*]
        'mat1': ['ENT', Kv],
        'mat2': ['Elastic', Krotz],
        'mat3': ['Elastic', Kroty],
        'mat4': ['Elastic', Ktor],

        #  ----------------------------------------------------------------------------
        #  FRICTION DEFINITIONS USED BY BEARING ELEMENTS
        #  ----------------------------------------------------------------------------
        # https://openseespydoc.readthedocs.io/en/latest/src/frictionModel.html
        'frn1': ['Coulomb', mub],  # FrnTag: [Friction Model, Friction Arguments*]

        #  ----------------------------------------------------------------------------
        #  EXAMPLE BENT CAP DEFINITIONS
        #  ----------------------------------------------------------------------------
        'cap1': {
            'length': 11 * m,  # From deck centroid to bearing top
            'width': 3 * m,  # From deck centroid to bearing top
            'height': 2.2 * m,  # Height of pier cap
            'A': 6.0112 * m ** 2,  # Area
            'Iy': 2.4871 * m ** 4,  # Moment of inertia in y-dir
            'Iz': 4.4045 * m ** 4,  # Moment of inertia in z-dir
            'J': 4.4116 * m ** 4,  # Polar moment of inertia
            'E': 30310.0 * MPa,  # Young's modulus
            'G': 12629.0 * MPa  # Shear modulus
        },

        #  ----------------------------------------------------------------------------
        #  EXAMPLE BENT CONFIGURATION AND PIER SECTION DEFINITIONS
        #  ----------------------------------------------------------------------------
        'bent1': {
            'spacing': 5 * m,  # Horizontal spacing between the Piers
            'height': 9.75 * m,  # Pier Heights
            'sections': ['sec1']  # Section properties of each pier
        },

        'bent2': {
            'spacing': 5 * m,  # Horizontal spacing between the Piers
            'height': 13.4 * m,  # Pier Heights
            'sections': ['sec1']  # Section properties of each pier
        },

        'bent3': {
            'spacing': 5 * m,  # Horizontal spacing between the Piers
            'height': 12.35 * m,  # Pier Heights
            'sections': ['sec1']  # Section properties of each pier
        },

        'bent4': {
            'spacing': 5 * m,  # Horizontal spacing between the Piers
            'height': 10.22 * m,  # Pier Heights
            'sections': ['sec1']  # Section properties of each pier
        },

        'sec1': {
            'Type': 'Solid Circular',  # Section geometry
            'D': 2.6 * m,  # Diameter of the circular pier section
            'Fce': 29.1 * MPa,  # Expected nominal compressive strength of the concrete material
            'Fyle': 415.2 * MPa,  # Expected yield strength of longitudinal rebars
            'dl': 34 * mm,  # Nominal diameter of longitudinal bars
            'number of bars': 90,  # Number of longitudinal bars
            'cover': 40 * mm,  # Clear cover to centroid of stirrups

            # additional information for steel model, if these are not provided default values are used.
            'max_eps_steel': 0.1,  # e.g. rupture (Default)
            'min_eps_steel': -0.1,  # e.g. buckling (Default)
            'rs': 0.002,  # strain hardening ratio (Default)
            'Es': 200 * GPa,  # steel Youngs modulus (Default)
            'Ec': 5000 * MPa * 29.1 ** 0.5,  # concrete Youngs modulus (Default)

            # necessary information for confinement, if these are provided, confined concrete properties are being used.
            'Confinement': 'Mander_et_al_1988',  # confinement model to use ('Kent_Park_1971', 'Mander_et_al_1988')
            'Fyhe': 415.2 * MPa,  # Expected yield strength of transverse rebars
            'dh': 14 * mm,  # Nominal diameter of transversal bars
            'sh': 444 * mm,  # Vertical spacing between the centroid of spirals or hoops
            'Transverse Reinforcement Type': 'Hoops',  # Type of transversal steel, 'Hoops' or 'Spirals'
        },

        #  ----------------------------------------------------------------------------
        #  EXAMPLE ABUTMENT CONFIGURATIONS
        #  ----------------------------------------------------------------------------
        'ab2': {
            'Type': 'SDC 2019',
            'h': 3.5 * m,
            'w': 12 * m,
            'L': 6 * m,
            'NumPile': 8,  # Piles are used to define activate soil response at abutments
        },

        #  ----------------------------------------------------------------------------
        #  EXAMPLE SHEAR KEY CONFIGURATIONS
        #  ----------------------------------------------------------------------------
        'sk1': {
            'Type': 'Non-Sacrificial',  # type of the shear key
            'gapT': 50 * mm,  # transverse gap between shear key and bearings shear key
            'gapL': 80 * mm  # longitudinal gap between shear key and bearings shear key at abutments
        },
    }

    # Save the object
    with open('bridge_input.pkl', 'wb') as handle:
        pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #  ----------------------------------------------------------------------------
    #  INPUT for MSA (Multiple Stripes Analysis):
    #  ----------------------------------------------------------------------------
    gm_components = [1, 2]
    dt_file = "GMR_dts.txt"  # Time steps of ground motions to run for any case
    h1_names_file = "GMR_H1_names.txt"  # Names of ground motions to run (dir 1) for uniform excitation case
    h2_names_file = "GMR_H2_names.txt"  # Names of ground motions to run (dir 2) for uniform excitation case
    v_names_file = "GMR_V_names.txt"  # Names of ground motions to run (dir 3) for uniform excitation case

    ################################################
    ############### INPUT ENDS HERE ################
    ################################################
    Modes = 20

    # Start the bridge object
    Bridge = smat.main(model=input_data, output_dir='Outputs_' + str(i+1), const_opt=0)

    # Generate the model
    Bridge.generate_model(mphi=1)

    # Plot moment curvature analysis for piers
    Bridge.plot_mphi(save=1)

    # # Plot the model
    Bridge.plot_model(show_node='yes')
    # Bridge.plot_sections()

    # Set some analysis configurations
    Bridge.set_analysis_parameters('Transformation', 'RCM', 'UmfPack', alphaM=1e14, alphaS=1e14)

    # Gravity analysis
    Bridge.do_gravity(pflag=0)

    # Modal analysis
    Bridge.do_modal(num_eigen=Modes, pflag=1)

    # Retrieve the first mode period for damping assignment
    Ts = Bridge.modal_properties['Periods']
    Ms = Bridge.modal_properties['Modal Mass Ratios']
    Ms_dir1 = Ms[:, 0]
    inds1 = Ms_dir1.argsort()[::-1]
    Ms_dir1 = Ms_dir1[inds1]
    Ts_dir1 = Ts[inds1]
    T1_dir1 = Ts_dir1[0]
    Ms_dir2 = Ms[:, 1]
    inds2 = Ms_dir2.argsort()[::-1]
    Ms_dir2 = Ms_dir2[inds2]
    Ts_dir2 = Ts[inds2]
    T1_dir2 = Ts_dir2[0]
    damping_period = (T1_dir1 * T1_dir2) ** 0.5

    # set damping
    Bridge.set_damping(option=3, damping_periods=damping_period, xi=0.02)

    # Multiple-stripe analysis
    Bridge.do_msa(gm_components=gm_components, dt_file=dt_file, msa_dir='Test', h1_names_file=h1_names_file, h2_names_file=h2_names_file,
                  v_names_file=v_names_file, gm_angle=0, dc=10, t_free=0)

    print(get_run_time(T0))
