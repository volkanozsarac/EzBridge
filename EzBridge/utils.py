"""
This module contains utility functions.

Author: Volkan Ozsarac, Earthquake Engineering PhD Candidate
Affiliation: University School for Advanced Studies IUSS Pavia
e-mail: volkanozsarac@iusspavia.it
"""

#  ----------------------------------------------------------------------------
#  Import Python Libraries
#  ----------------------------------------------------------------------------
import errno
import os
import shutil
import stat
import time
from datetime import date, datetime
import numpy as np
import numba as nb
from scipy import stats
from scipy.optimize import curve_fit, minimize


def program_info(print_flag=0):
    """
    -------------------
    PROGRAM INFORMATION
    -------------------
    The method saves the basic software information as string

    Parameters
    ----------
    print_flag: int, optional (The default is 0)
        If 1, prints the output on the screen

    Returns
    -------
    text: str
        String describes the software

    """
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    # Sample program info to print
    text = ("|-----------------------------------------------------------------------|\n" +
            "|    EzBridge: Program to perform dynamic analysis                      |\n" +
            "|    and direct seismic loss assessment of RC bridges                   |\n" +
            "|    having piers with Circular Sections                                |\n" +
            "|    Version: 1.0                                                       |\n" +
            f"|    Date: {d1}                                                   |\n" +
            "|                                                                       |\n" +
            "|    Created on 10/01/2021                                              |\n" +
            "|    Updated on 09/03/2022                                              |\n" +
            "|    Author: Volkan Ozsarac                                             |\n" +
            "|    Affiliation: University School for Advanced Studies IUSS Pavia     |\n" +
            "|    Earthquake Engineering PhD Candidate                               |\n" +
            "|                                                                       |\n" +
            "|-----------------------------------------------------------------------|\n")

    if print_flag == 1:
        print(text)

    return text


def get_run_time(start_time, print_flag=0):
    """
    -----------
    PASSED TIME
    -----------
    The method computes the total amount of time passed since start_time

    Parameters
    ----------
    print_flag: int, optional (The default is 0)
        If 1, prints the output on the screen
    start_time: float
        Reference time to calculate passed time until now. time.perf_counter()

    Returns
    -------
    text: str
        String describing total amount of time passed since start_time in hr,min,sec

    """

    # Procedure to obtained elapsed time in Hr, Min, and Sec
    finish_time = time.perf_counter()
    time_seconds = finish_time - start_time
    time_minutes = int(time_seconds / 60)
    time_hours = int(time_seconds / 3600)
    time_minutes = int(time_minutes - time_hours * 60)
    time_seconds = time_seconds - time_minutes * 60 - time_hours * 3600
    text = "Run time: %d hours: %d minutes: %.2f seconds" % (time_hours, time_minutes, time_seconds)

    if print_flag == 1:
        print(text)

    return text


def get_distance(coord1, coord2):
    """
    ---------------------------
    DISTANCE BETWEEN TWO POINTS
    ---------------------------

    Parameters
    ----------
    coord1: list, numpy.ndarray
        Coordinates of the first point
    coord2: list, numpy.ndarray
        Coordinates of the second point

    Returns
    -------
    distance: float
        The distance between two points on 3-D cartesian system

    """

    dist = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2) ** 0.5
    return dist


def get_current_time():
    """
    ------------
    CURRENT TIME
    ------------

    Parameters
    ----------

    Returns
    -------
    time.perf_counter()
        The point where the time starts in computer

    """

    return time.perf_counter()


def create_dir(output_path):
    """
    ----------------
    CREATE DIRECTORY
    ----------------
    The method creates an empty directory. It will delete it if the directory exists, and recreate it.

    Parameters
    ----------
    output_path : str
        output directory to create.

    Returns
    -------

    """

    def handle_remove_readonly(func, path, exc):
        exc_value = exc[1]
        if func in (os.rmdir, os.remove) and exc_value.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
            func(path)
        else:
            raise

    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=False, onerror=handle_remove_readonly)
    os.makedirs(output_path)


def read_nga_record(in_file_name, out_file_name=None):
    """
    --------------------------
    READ RECORD IN PEER FORMAT
    --------------------------
    This method processes the acceleration history for NGA data file (.AT2 format)
    to a single column value.

    Parameters
    ----------
    in_file_name : str
        Path to the input file
    out_file_name : str
        Path to the output file

    Returns
    -------
    desc: str
        Description of the earthquake (e.g., name, year, etc.)
    npts: int
        Total number of recorded points (acceleration data)
    dt: float
        Time interval of recorded points
    time: numpy.ndarray (n x 1)
        Time array, same length with npts
    inp_acc: numpy.ndarray (n x 1)
        Acceleration array, same length with time unit usually in (g) unless stated as other.
    """

    try:
        with open(in_file_name, 'r') as inFileID:
            content = inFileID.readlines()
        counter = 0
        desc, row4_val, acc_data = "", "", []
        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4_val = x
                if row4_val[0][0] == 'N':
                    val = row4_val.split()
                    npts = int(val[(val.index('NPTS=')) + 1].rstrip(','))
                    dt = float(val[(val.index('DT=')) + 1])
                else:
                    val = row4_val.split()
                    npts = int(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value)
                    acc_data.append(a)
                inp_acc = np.asarray(acc_data)
                time_list = []
                for i in range(0, len(acc_data)):
                    t = i * dt
                    time_list.append(t)
            counter = counter + 1
        if out_file_name is not None:
            np.savetxt(out_file_name, inp_acc, fmt='%1.4e')

        inFileID.close()
        return dt, npts, desc, time_list, inp_acc
    except IOError:
        print("processMotion FAILED!: File is not in the directory")


def get_units(print_flag=0):
    """
    -------------------
    DEFINITION OF UNITS
    -------------------
    This method defines units and their conversions to the basic units: m, kN, sec

    Parameters
    ----------
    print_flag; int
        Flag to print allowed units on the screen

    Returns
    -------
    m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi
        Value of the all allowed units in terms of basic units (m, kN, sec)
    """

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

    if print_flag == 1:
        print('Values are returned for the following units:')
        print('m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi')
    return m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi


def find_nearest(array, value):
    """
    Details
    -------
    Finds the row where the nearest value is located

    Parameters
    ----------
    array : numpy.ndarray
        array in which the row of the closest value is sought
    value: float, int
        value used to find closest value in array

    Returns
    -------
    idx: int, list, tuple
        index of the closest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def mle_fit(points, trials, observations, initial_guess=None):
    """
    Details
    -------
    Applies maximum likelihood estimation to determine lognormal distribution parameters.
    Here, log-normal distribution represents the fragility curve.

    References
    ----------
    Baker, J. W. (2015). Earthquake Spectra, 31(1), 579-599

    Parameters
    ----------
    points: numpy.ndarray (1xn)
        Points or intensity measure levels (IML) of interest
    trials: numpy.ndarray
        Number of trials or ground motions at IMLs
    observations: numpy.ndarray (1xn)
        number of observations or collapses observed at IMLs
    initial_guess: list, optional (The default is None)
        Initial guess for the log-normal distribution parameters [theta,beta]

    Returns
    -------
    sol.x: numpy.ndarray
        Logarithmic mean and standard deviation for the fitted fragility curve

    Example
    -------
    Same as on Baker's video: https://www.youtube.com/watch?v=Q8e0_d81a40
    mle_fit(im=[0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0], trials = 40, obs = [0,0,0,4,6,13,12,16])
    output: array([1.07611623, 0.42923779])
    """

    if initial_guess is None:
        initial_guess = [0.8, 0.4]

    if len(trials if type(trials) is list else [trials]) == 1:
        trials = np.repeat(trials, len(points))

    def func(x):
        theta, beta = x
        # poes = observations / trials
        # evaluates the probability calculated under the initial assumption of theta and beta
        poes_fitted = stats.lognorm.cdf(points, beta, scale=theta)
        # calculates the likelihood of the parameters being correct
        likelihood = stats.binom.pmf(k=observations, n=trials, p=poes_fitted)
        likelihood[likelihood <= 0] = 1e-200  # do not assign zero, assign zero like value
        # sum of the logs of the likelihood to be minimized
        return -np.sum(np.log(likelihood))

    # this finds a solution, bounds prevent negative values, check documentation for other settings
    try:
        sol = minimize(func, x0=initial_guess, method='SLSQP', bounds=((0, None), (0, None)))
    except:
        initial_guess = [1.0, 0.5]
        sol = minimize(func, x0=initial_guess, method='SLSQP', bounds=((0, None), (0, None)))

    return sol.x


def log_normal_distribution(points, theta, beta):
    """
    Parameters
    ----------
    points: numpy.ndarray (1xn)
        Points for which probability values in a log-normal distribution are obtained
    theta: float
        Median of log-normal distribution
    beta: float
        Logarithmic standard deviation of log-normal distribution

    Returns
    -------
    prob: numpy.ndarray (1xn)
        Probabilities
    """

    prob = stats.norm.cdf((np.log(points / theta)) / beta)
    return prob


def fit_log_normal_distribution(points, prob_occur):
    """
    Details
    -------
    Uses non-linear least squares to determine log-normal distribution parameters for given empirical data.
    Here, log-normal distribution represents the fragility curve.

    Parameters
    ----------
    points: numpy.ndarray
        Points or intensity measure levels (IML) of interest
    prob_occur: numpy.ndarray
        Probability of occurrences or collapses

    Returns
    -------
    theta: float
        Median of log-normal distribution
    beta: float
        Logarithmic standard deviation of log-normal distribution

    """

    parameters, cov = curve_fit(log_normal_distribution, points, prob_occur)
    theta, beta = parameters  # distribution parameters
    return theta, beta


def ecdf(data):
    """
    Details
    -------
    Computes empirical cumulative distribution function

    Parameters
    ----------
    data: numpy.ndarray
        array of data where ecdf is being calculated

    Returns
    -------
    x: numpy.ndarray
        sorted data values
    y: numpy.ndarray
        probabilities
    """

    x = np.sort(data, axis=0)
    n = x.size
    y = np.arange(1, n + 1) / n

    return x, y


def normal_cdf(x):
    """
    Details
    -------
    Computes cumulative distribution function (CDF) for standard normal distribution

    Parameters
    ----------
    x: numpy.ndarray
        array of data where CDF for standard normal distribution is being calculated

    Returns
    -------
    probs: numpy.ndarray
        probability values of CDF for standard normal distribution given x
    """

    if np.array(x).ndim == 0:
        x = np.array([x])
    
    probs = numba_cdf(x)

    return probs


@nb.njit
def numba_cdf(x):
    """
    Details
    -------
    Computes cumulative distribution function (CDF) for standard normal distribution

    Parameters
    ----------
    x: numpy.ndarray
        array of data where CDF for normal distribution is being calculated

    Returns
    -------
    probs: numpy.ndarray
        probability values of CDF for standard normal distribution given x
    """

    val = x / (2 ** 0.5)
    z = np.abs(val)
    t = 1. / (1. + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t *
                   (1.00002368 + t *
                    (.37409196 + t *
                     (.09678418 + t *
                      (-.18628806 + t *
                       (.27886807 + t *
                        (-1.13520398 + t *
                         (1.48851587 + t *
                          (-.82215223 + t * .17087277)))))))))

    probs = 1 - 0.5 * r

    probs[x < 0] = 1 - 0.5 * (2. - r[x < 0])

    return probs


def do_sampling(num_variables, num_samples, sampling_type):
    """
    Details
    -------
    Used to perform sampling based on Monte Carlo Simulation or Latin Hypecube Sampling

    References
    ----------
    Olsson, A., Sandberg, G., & Dahlblom, O. (2003). On Latin hypercube sampling for structural reliability analysis.
    In Structural Safety (Vol. 25, Issue 1, pp. 47–68). Elsevier BV. https://doi.org/10.1016/s0167-4730(02)00039-5
    Olsson, A. M. J., & Sandberg, G. E. (2002). Latin Hypercube Sampling for Stochastic Finite Element Analysis.
    In Journal of Engineering Mechanics (Vol. 128, Issue 1, pp. 121–125).
    American Society of Civil Engineers (ASCE). https://doi.org/10.1061/(asce)0733-9399(2002)128:1(121)
    About: LatinHypercube class
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html#scipy.stats.qmc.LatinHypercube

    Parameters
    ----------
    num_variables: int
        number of variables
    num_samples: int
        number of realizations
    sampling_type: str
        type of sampling.
        Monte Carlo Sampling: 'MCS'
        Latin Hypercube Sampling: 'LHS'

    Returns
    -------
    sample: numpy.ndarray (num_samples x num_variables)
        Array which contains randomly generated numbers between 0 and 1
    """
    # Not really required, but will ensure different realizations each time
    seed = int(datetime.today().strftime("%H%M%S"))
    if sampling_type == 'MCS':
        # Do Monte Carlo Sampling without any grid
        np.random.seed(seed)
        sample = np.random.uniform(size=[num_variables, num_samples]).T
    elif sampling_type == 'LHS':
        # # Limits of each grid
        # lower_limits = np.arange(0, num_samples) / num_samples
        # upper_limits = np.arange(1, num_samples + 1) / num_samples
        # # Do Monte Carlo Sampling with predefined grids based on number of realizations
        # sample = np.random.uniform(low=lower_limits, high=upper_limits, size=[num_variables, num_samples])
        # # Shuffle each row separately to remove unwanted correlation of sampling plan
        # for x in sample:
        #     np.random.shuffle(x)

        # Use scipy
        sampler = stats.qmc.LatinHypercube(d=num_variables, seed=seed, strength=2)
        # sampler = stats.qmc.LatinHypercube(d=num_variables, seed=seed, strength=2, optimization='random-cd')
        sample = sampler.random(n=num_samples)

    return sample
