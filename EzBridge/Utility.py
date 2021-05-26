#  ----------------------------------------------------------------------------
#  Import Python Libraries
#  ----------------------------------------------------------------------------
import errno
import os
import shutil
import stat
import time
from datetime import date
import numpy as np


def program_info():
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    # Sample program info to print
    textstr = ("\
|-----------------------------------------------------------------------|\n\
|                                                                       |\n\
|    EzBridge                                                           |\n\
|    Ordinary RC Bridge Modeller                                        |\n\
|    Version: 1.0                                                       |\n\
|    Units: kN, m, sec                                                  |\n\
|    Date: %s                                                   |\n\
|                                                                       |\n\
|    Created on 10/01/2021                                              |\n\
|    Updated on 08/04/2021                                              |\n\
|    Author: Volkan Ozsarac                                             |\n\
|    Affiliation: University School for Advanced Studies IUSS Pavia     |\n\
|    Earthquake Engineering PhD Candidate                               |\n\
|                                                                       |\n\
|-----------------------------------------------------------------------|\n" % d1)

    return textstr


def RunTime(startTime):
    # Procedure to obtained elapsed time in Hr, Min, and Sec
    finishTime = time.time()
    timeSeconds = finishTime - startTime
    timeMinutes = int(timeSeconds / 60)
    timeHours = int(timeSeconds / 3600)
    timeMinutes = int(timeMinutes - timeHours * 60)
    timeSeconds = timeSeconds - timeMinutes * 60 - timeHours * 3600
    text = "Run time: %d hours: %d minutes: %.2f seconds" % (timeHours, timeMinutes, timeSeconds)
    return text


def Get_T0():
    return time.time()


def create_outdir(outdir_path):
    """  
    Parameters
    ----------
    outdir_path : str
        output directory to create.

    Returns
    -------
    None.
    """

    def handleRemoveReadonly(func, path, exc):
        excvalue = exc[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
            func(path)
        else:
            raise

    if os.path.exists(outdir_path):
        shutil.rmtree(outdir_path, ignore_errors=False, onerror=handleRemoveReadonly)
    os.makedirs(outdir_path)


def distance(coord1, coord2):
    dist = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2) ** 0.5
    return dist


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


def getCircProp(D):
    # D: Pier Diameter
    pi = 3.141592653589793
    A = pi * D ** 2 / 4
    J = pi * D ** 4 / 32
    Iy = pi * D ** 4 / 64
    Iz = pi * D ** 4 / 64

    return A, Iy, Iz, J


def ReadRecord(inFilename, outFilename=None):
    """
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and
    time iterval of the recording.
    Parameters:
    ------------
    inFilename : string (location and name of the input file)
    outFilename : string (location and name of the output file)

    Output:
    ------------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.

    """
    try:
        with open(inFilename, 'r') as inFileID:
            content = inFileID.readlines()
        counter = 0
        desc, row4Val, acc_data = "", "", []
        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4Val = x
                if row4Val[0][0] == 'N':
                    val = row4Val.split()
                    npts = float(val[(val.index('NPTS=')) + 1].rstrip(','))
                    dt = float(val[(val.index('DT=')) + 1])
                else:
                    val = row4Val.split()
                    npts = float(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value)
                    acc_data.append(a)
                inp_acc = np.asarray(acc_data)
                time = []
                for i in range(0, len(acc_data)):
                    t = i * dt
                    time.append(t)
            counter = counter + 1
        if outFilename is not None:
            np.savetxt(outFilename, inp_acc, fmt='%1.4e')

        inFileID.close()
        return dt, int(npts), desc, time, inp_acc
    except IOError:
        print("processMotion FAILED!: File is not in the directory")


def def_units(pFlag):
    #  ------------------------------------------------------------------------------------------------------------
    #  DEFINITION OF UNITS
    #  ------------------------------------------------------------------------------------------------------------
    # Basic Units
    m = 1.0
    kN = 1.0
    sec = 1.0

    # Length
    mm = m / 1000.0
    cm = m / 100.0
    inch = 25.4 * mm
    ft = 12.0 * inch

    # Force
    N = kN / 1000.0
    kip = kN * 4.448221615

    # Mass (tonnes)
    tonne = kN * sec ** 2 / m
    kg = N * sec ** 2 / m

    # Stress (kN/m2 or kPa)
    Pa = N / (m ** 2)
    kPa = Pa * 1.0e3
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9
    ksi = 6.8947573 * MPa
    psi = 1e-3 * ksi

    if pFlag == 1:
        print('Values are returned for the following units:\n\
m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, degrees\n')
    return m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi


def sdof_ltha(Ag, dt, T, xi, m):
    """
    Details
    -------
    This script will carry out linear time history analysis for SDOF system
    It currently uses Newmark Beta Method
    
    References
    ---------- 
    Chopra, A.K. 2012. Dynamics of Structures: Theory and 
    Applications to Earthquake Engineering, Prentice Hall.
    N. M. Newmark, “A Method of Computation for Structural Dynamics,”
    ASCE Journal of the Engineering Mechanics Division, Vol. 85, 1959, pp. 67-94.
    
    Notes
    -----
    * Linear Acceleration Method: Gamma = 1/2, Beta = 1/6
    * Average Acceleration Method: Gamma = 1/2, Beta = 1/4
    * Average acceleration method is unconditionally stable,
      whereas linear acceleration method is stable only if dt/Tn <= 0.55
      Linear acceleration method is preferable due to its accuracy.
    
    Parameters
    ----------
    Ag: numpy.ndarray    
        Acceleration values
    dt: float
        Time step [sec]
    T:  float, numpy.ndarray
        Considered period array e.g. 0 sec, 0.1 sec ... 4 sec
    xi: float
        Damping ratio, e.g. 0.05 for 5%
    m:  float
        Mass of SDOF system
        
    Returns
    -------
    u: numpy.ndarray       
        Relative displacement response history
    v: numpy.ndarray   
        Relative velocity response history
    ac: numpy.ndarray 
        Relative acceleration response history
    ac_tot: numpy.ndarray 
        Total acceleration response history
    """

    # Get the length of acceleration history array
    n1 = max(Ag.shape)
    # Get the length of period array
    n2 = max(T.shape)
    T = T.reshape((1, n2))

    # Assign the external force
    p = -m * Ag

    # Calculate system properties which depend on period
    fn = np.ones(T.shape)
    fn = 1 / T  # frequency
    wn = np.ones(T.shape)
    wn = 2 * np.pi * fn  # circular natural frequency
    k = np.ones(T.shape)
    k = m * wn ** 2  # actual stiffness
    c = np.ones(T.shape)
    c = 2 * m * wn * xi  # actual damping coefficient

    # Newmark Beta Method coefficients
    Gamma = np.ones((1, n2)) * (1 / 2)
    # Use linear acceleration method for dt/T<=0.55
    Beta = np.ones((1, n2)) * 1 / 6
    # Use average acceleration method for dt/T>0.55
    Beta[np.where(dt / T > 0.55)] = 1 / 4

    # Compute the constants used in Newmark's integration
    a1 = Gamma / (Beta * dt)
    a2 = 1 / (Beta * dt ** 2)
    a3 = 1 / (Beta * dt)
    a4 = Gamma / Beta
    a5 = 1 / (2 * Beta)
    a6 = (Gamma / (2 * Beta) - 1) * dt
    kf = k + a1 * c + a2 * m
    a = a3 * m + a4 * c
    b = a5 * m + a6 * c

    # Initialize the history arrays
    u = np.zeros((n1, n2))  # relative displacement history
    v = np.zeros((n1, n2))  # relative velocity history
    ac = np.zeros((n1, n2))  # relative acceleration history
    ac_tot = np.zeros((n1, n2))  # total acceleration history

    # Set the Initial Conditions
    u[0] = 0
    v[0] = 0
    ac[0] = (p[0] - c * v[0] - k * u[0]) / m
    ac_tot[0] = ac[0] + Ag[0]

    for i in range(n1 - 1):
        dpf = (p[i + 1] - p[i]) + a * v[i] + b * ac[i]
        du = dpf / kf
        dv = a1 * du - a4 * v[i] - a6 * ac[i]
        da = a2 * du - a3 * v[i] - a5 * ac[i]

        # Update history variables
        u[i + 1] = u[i] + du
        v[i + 1] = v[i] + dv
        ac[i + 1] = ac[i] + da
        ac_tot[i + 1] = ac[i + 1] + Ag[i + 1]

    return u, v, ac, ac_tot


def get_pyParam_sand(pyDepth, sigV, phiDegree, b, pEleLength, LSwitch=1, puSwitch=1, kSwitch=2, gwtSwitch=1):
    """
    ###########################################################
    #                                                         #
    # Procedure to compute ultimate lateral resistance, p_u,  #
    #  and displacement at 50% of lateral capacity, y50, for  #
    #  p-y springs representing cohesionless soil.            #
    #                                                         #
    ###########################################################
    
    References
    ----------
    American Petroleum Institute (API) (1987). Recommended Practice for Planning, Designing and
    Constructing Fixed Offshore Platforms. API Recommended Practice 2A(RP-2A), Washington D.C,
    17th edition.
    
    Brinch Hansen, J. (1961). “The ultimate resistance of rigid piles against transversal forces.”
    Bulletin No. 12, Geoteknisk Institute, Copenhagen, 59.
    
    Boulanger, R. W., Kutter, B. L., Brandenberg, S. J., Singh, P., and Chang, D. (2003). Pile 
    Foundations in liquefied and laterally spreading ground during earthquakes: Centrifuge experiments
    and analyses. Center for Geotechnical Modeling, University of California at Davis, Davis, CA.
    Rep. UCD/CGM-03/01.
    
    Reese, L.C. and Van Impe, W.F. (2001), Single Piles and Pile Groups Under Lateral Loading.
    A.A. Balkema, Rotterdam, Netherlands.
        
    Parameters
    ----------
    pyDepth : float
        depth of spring.
    sigV : float
        effective overburden pressure.
    phiDegree : float
        angle of friction.
    b : float
        Pile diameter.
    pEleLength : float
        length of pile element.
    LSwitch : int
        static or cyclic loading conditions.
        cyclic --> 1
        static --> 2
    puSwitch : int
         pult is defined per API recommendations (Reese and Van Impe, 2001 or API, 1987) for --> 1
         OR per the method of Brinch Hansen (1961) for --> 2
    kSwitch : int
        variation in coefficent of subgrade reaction with depth for p-y curves.
        API linear variation (default)   --> 1
        modified API parabolic variation --> 2
    gwtSwitch : int
        effect of ground water on subgrade reaction modulus for p-y curves.
        above gwt --> 1
        below gwt --> 2

    Returns
    -------
    None.

    """
    # need to be careful at ground surface (don't want to have zeros)
    if pyDepth == 0:
        pyDepth = 0.01
        sigV = 0.01

    phi = phiDegree * (np.pi / 180)
    zbRatio = pyDepth / b
    # -------API recommended method-------
    # obtain loading-type coefficient A for given depth-to-diameter ratio zb
    # from linear relationship given in API
    if LSwitch == 1:  # cyclic loading
        A = 0.9
    else:  # static loading
        A = max(3 - 0.8 * zbRatio, 0.9)

    # ----------------------------------------------------------
    #  define ultimate lateral resistance, pult 
    # ----------------------------------------------------------
    if puSwitch == 1:
        # pult is defined per API recommendations (Reese and Van Impe, 2001 or API, 1987)

        # define common terms same as LpliePlus, Boulanger et al. 2003, pp. 5-6
        alpha = phi / 2
        beta = np.pi / 4 + phi / 2
        K0 = 0.4

        # terms for Equation (3.8), Reese and Van Impe (2001)
        tan_1 = np.tan(np.pi / 4 - phi / 2)
        tan_2 = np.tan(phi)
        tan_3 = np.tan(beta - phi)
        sin_1 = np.sin(beta)
        cos_1 = np.cos(alpha)
        tan_4 = np.tan(beta)
        tan_5 = np.tan(alpha)
        c1 = K0 * tan_2 * sin_1 / (tan_3 * cos_1) + (tan_4 / tan_3) * tan_4 * tan_5 + K0 * tan_4 * (
                tan_2 * sin_1 - tan_5)

        Ka = tan_1 ** 2
        c2 = tan_4 / tan_3 - Ka

        # terms for Equation (3.10), Reese and Van Impe (2001)
        c3 = Ka * (tan_4 ** 8 - 1) + K0 * tan_2 * tan_4 ** 4

        # Equation (3.8), Reese and Van Impe (2001)
        pst = sigV * (pyDepth * c1 + b * c2)

        # Equation (3.10), Reese and Van Impe (2001)
        psd = b * sigV * c3

        # pult is the lesser of pst and psd. At surface, an arbitrary value is defined
        pu = A * min(pst, psd)

    else:
        # pressure at ground surface
        Kqo = np.exp((np.pi / 2 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(np.pi / 4 + phi / 2) - \
              np.exp(-(np.pi / 2 - phi) * np.tan(phi)) * np.cos(phi) * np.tan(np.pi / 4 - phi / 2)
        Kco = (1 / np.tan(phi)) * (
                np.exp((np.pi / 2 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(np.pi / 4 + phi / 2) - 1)

        # pressure at great depth
        dcinf = 1.58 + 4.09 * (np.tan(phi)) ** 4
        Nc = (1 / np.tan(phi)) * (np.exp(np.pi * np.tan(phi))) * (np.tan(np.pi / 4 + phi / 2) ** 2 - 1)
        Ko = 1 - np.sin(phi)
        Kcinf = Nc * dcinf
        Kqinf = Kcinf * Ko * np.tan(phi)

        # pressure at an arbitrary depth
        aq = Kqo / (Kqinf - Kqo) * (Ko * np.sin(phi)) / np.sin(np.pi / 4 + phi / 2)
        KqD = (Kqo + Kqinf * aq * zbRatio) / (1 + aq * zbRatio)

        # ultimate lateral resistance
        pu = sigV * KqD * b

    # PySimple1 material formulated with pult as a force, not force/length, multiply by trib. length
    pult = pu * pEleLength
    # ----------------------------------------------------------
    #  define displacement at 50% lateral capacity, y50
    # ----------------------------------------------------------
    # values of y50 depend of the coefficent of subgrade reaction, k, which can be defined in several ways.
    # API (1987) recommended subgrade modulus for given friction angle, approximate values obtained from figure 6.8.7-1
    # ph = [28.8, 29.5, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]
    #
    # # subgrade modulus above the water table
    # if gwtSwitch == 1:
    #     k = [10, 23, 45, 61, 80, 100, 120, 140, 160, 182, 215, 250, 275]
    #
    # else:
    #     k = [10, 20, 33, 42, 50, 60, 70, 85, 95, 107, 122, 141, 155]
    #
    # dataNum = 13
    # for i in range(dataNum):
    #     if ph[i] <= phiDegree and phiDegree <= ph[i + 1]:
    #         khat = (k[i + 1] - k[i]) / (ph[i + 1] - ph[i]) * (phiDegree - ph[i]) + k[i]
    #
    # # change units from (lb/in^3) to (kN/m^3)
    # k_SIunits = khat * 271.45

    # subgrade modulus above the water table
    if gwtSwitch == 1:
        ck0 = -5739.582666
        ck1 = -3857.636939
        ck2 = 146.2629258
    # subgrade modulus below the water table
    else:
        ck0 = -15337.63982
        ck1 = -1240.567342
        ck2 = 66.60706369

    k_SIunits = ck0 + phiDegree * (ck1 + ck2 * phiDegree)
    k_SIunits = max(k_SIunits, 10)

    # overburden pressure correction
    if kSwitch == 2:
        # Equation (5-16), Boulanger et al. (2003)
        cSigma = (50 / sigV) ** 0.5
        # Equation (5-15), Boulanger et al. (2003)
        k_SIunits = cSigma * k_SIunits

    # define y50 based on pult and subgrade modulus k    
    # based on API (1987) recommendations, p-y curves are described using tanh functions.    
    # when half of full resistance has been mobilized, p(y50)/pult = 0.5
    # y50 = 0.5 * (pu / A) / (k_SIunits * pyDepth) * np.arctanh(0.5)
    y50 = pu / (k_SIunits * pyDepth) * np.arctanh(0.5/A)
    y = np.logspace(-4, 0, num=20)
    P = pu * np.tanh(k_SIunits * pyDepth / pu * y)

    strain_stress = []
    for i in range(len(y)):
        strain_stress.append(y[i])
        strain_stress.append(P[i])

    return y50, pult, np.array(strain_stress)


def get_pyParam_clay(pyDepth, sigV, cu, eps50, b, pEleLength, clay='soft clay', mSwitch=1):
    # p-y curves for soft clay with free water (Matlock, 1970)
    if mSwitch == 1:
        # Emprical constant
        J = 0.5

        pu1 = (3 + sigV / cu + J * pyDepth / b) * cu * b
        pu2 = 9 * cu * b
        pu = min(pu1, pu2)
        y50 = 2.5 * eps50 * b
        pult = pu * pEleLength
        y = np.logspace(-4, 0, num=20)
        if clay == 'soft clay' or clay == 'medium clay':
            P = 0.5 * pu * (y / y50) ** (1 / 3)
        elif clay == 'stiff clay':
            P = 0.5 * pu * (y / y50) ** (1 / 4)
        P[P > pult] = pult

    # elif mSwitch = 2:
    # p-y curves for stiff clay with free water (Reese, et al., 1975)
    # TODO: do it later

    strain_stress = []
    for i in range(len(y)):
        strain_stress.append(y[i])
        strain_stress.append(P[i])

    return y50, pult, np.array(strain_stress)


def get_qzParam(phi, b, sigV, G):
    """
    ###########################################################
    #                                                         #
    #  Procedure to compute ultimate tip resistance, qult,    #
    #  displacement at 50% mobilization of qult, z50, for     #
    #  use in q-z curves for cohesionless soil.               #
    #                                                         #
    #   Created by:  Chris McGann                             #
    #                Pedro Arduino                            #
    #                University of Washington                 #
    #                                                         #
    ###########################################################

    References:
        Meyerhof G.G. (1976). Bearing capacity and settlement of pile foundations.
        J. Geotech. Eng. Div., ASCE, 102(3), 195-228.

        Vijayvergiya, V.N. (1977). Load-movement characteristics of piles.
        Proc., Ports 77 Conf., ASCE, New York.

        Kulhawy, F.H. ad Mayne, P.W. (1990). Manual on Estimating Soil Properties for
        Foundation Design. Electrical Power Research Institute. EPRI EL-6800,
        Project 1493-6 Final Report.

    Args:
        phi: Angle of friction in degrees
        b: pile diameter
        sigV: effective overburden pressure stress at the pile tip
        G: shear modulus of the soil layer at the pile tip

    Returns:
        z50: strain at q50
        qult: ultimate bearing capacity
    """

    # required constants
    sin_4 = np.sin(phi * (np.pi / 180))
    Ko = 1 - sin_4

    # ultimate tip pressure can be computed by qult = Nq*sigV after Meyerhof (1976)
    # where Nq is a bearing capacity factor
    phi = phi * (np.pi / 180)

    # rigidity index
    tan_7 = np.tan(phi)
    Ir = G / (sigV * tan_7)

    # bearing capacity factor
    tan_8 = np.tan(np.pi / 4 + phi / 2)
    sin_5 = np.sin(phi)
    pow_4 = tan_8 ** 2
    pow_5 = Ir ** ((4 * sin_5) / (3 * (1 + sin_5)))
    exp_4 = np.exp(np.pi / 2 - phi)
    Nq = (1 + 2 * Ko) * (1 / (3 - sin_5)) * exp_4 * pow_4 * pow_5

    # tip resistance
    qu = Nq * sigV

    # QzSimple1 material formulated with qult as force, not stress, multiply by area of pile tip
    pow_6 = b ** 2
    qult = qu * np.pi * pow_6 / 4

    # the q-z curve of Vijayvergiya (1977) has the form, q(z) = qult*(z/zc)^(1/3)
    # where zc is critical tip deflection given as ranging from 3-9% of the
    # pile diameter at the tip.

    # assume zc is 5% of pile diameter
    zc = 0.05 * b

    # based on Vijayvergiya (1977) curve, z50 = 0.125*zc
    z50 = 0.125 * zc

    return z50, qult


def get_tzParam(phi, b, sigV, pEleLength):
    """
    ###########################################################
    #                                                         #
    # Procedure to compute ultimate resistance, tult, and     #
    # displacement at 50% mobilization of tult, z50, for      #
    # use in t-z curves for cohesionless soil.                #
    #                                                         #
    #   Created by:  Chris McGann                             #
    #                Peter Mackenzie-Helnwein                 #
    #                University of Washington                 #
    #                                                         #
    ###########################################################

    References:
        Mosher, R.L. (1984). “Load transfer criteria for numerical analysis of
        axial loaded piles in sand.” U.S. Army Engineering and Waterways
        Experimental Station, Automatic Data Processing Center, Vicksburg, Miss.

        Kulhawy, F.H. (1991). "Drilled shaft foundations." Foundation engineering
        handbook, 2nd Ed., Chap 14, H.-Y. Fang ed., Van Nostrand Reinhold, New York

    Args:
        phi: Angle of friction in degrees
        b: pile diameter
        sigV: effective overburden pressure stress at the pile tip
        pEleLength: length of the pile element

    Returns:


    """

    # Compute tult based on tult = Ko*sigV*pi*dia*tan(delta), where
    #   Ko    is coeff. of lateral earth pressure at rest,
    #         taken as Ko = 0.4
    #   delta is interface friction between soil and pile,
    #         taken as delta = 0.8*phi to be representative of a
    #         smooth precast concrete pile after Kulhawy (1991)
    delta = 0.8 * phi * np.pi / 180

    # if z = 0 (ground surface) need to specify a small non-zero value of sigV
    if sigV == 0.0:
        sigV = 0.01

    tan_9 = np.tan(delta)
    tu = 0.4 * sigV * np.pi * b * tan_9

    # TzSimple1 material formulated with tult as force, not stress, multiply by tributary length of pile
    tult = tu * pEleLength

    # # Mosher (1984) provides recommended initial tangents based on friction angle
    # # values are in units of psf/in
    # kf = [6000, 10000, 10000, 14000, 14000, 18000]
    # fric = [28, 31, 32, 34, 35, 38]
    #
    # # determine kf for input value of phi, linear interpolation for intermediate values
    # dataNum = len(fric)
    # if phi < fric[0]:
    #     k = kf[0]
    # elif phi > fric[5]:
    #     k = kf[5]
    # else:
    #     for i in range(dataNum):
    #         if fric[i] <= phi <= fric[i + 1]:
    #             k = ((kf[i + 1] - kf[i]) / (fric[i + 1] - fric[i])) * (phi - fric[i]) + kf[i]
    #
    # # need to convert kf to units of kN/m^3
    # kSIunits = k * 1.885

    # alternatively, use the regression in kN/m^3 (Peter Mackenzie-Helnwein, 2018)
    kSIunits = 2304 * phi - 53408
    if kSIunits < 1.e-4:
        kSIunits = 1.e-4

    # based on a t-z curve of the shape recommended by Mosher (1984), z50 = tult/kf
    z50 = tult / kSIunits

    return z50, tult
