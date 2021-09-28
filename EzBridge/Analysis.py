import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from matplotlib import gridspec


def SinglePush(dref, mu, ctrlNode, dispDir, nSteps, IOflag=1):
    """
    Procedures to carry out a non-cylic pushover of a model

    Args:
        dref: Reference displacement to which cycles are run. Corresponds to yield or equivalent other, such as 1mm
        mu: Multiple of dref to which the push is run. So pushover can be run to a specified ductility or displacement
        ctrlNode: Node to control with the displacement integrator.
        dispDir: DOF the loading is applied.
        nSteps: Number of steps.
        IOflag: Option to print details on screen. 2 for print of each step, 1 for basic info (default), 0 for off

    Returns:
        LoadFactor: Load factor
        DispCtrlNode: Displacement of control node
    """

    # Procedures to carry out a pushover of a model
    # About Test Types
    # NormUnbalance                 Dont use with Penalty constraints
    # NormDispIncr
    # EnergyIncr                    Dont use with Penalty constraints
    # RelativeNormUnbalance         Dont use with Penalty constraints
    # RelativeNormDispIncr          Dont use with Lagrange constraints
    # RelativeTotalNormDispIncr     Dont use with Lagrange constraints
    # RelativeEnergyIncr            Dont use with Penalty constraints
    # FixedNumIter                  Performs a fixed number of iterations without testing for convergence

    LoadFactor = [0]
    DispCtrlNode = [0]

    testType = {1: 'NormUnbalance', 2: 'NormDispIncr', 3: 'EnergyIncr', 4: 'RelativeNormUnbalance',
                5: 'RelativeNormDispIncr', 6: 'RelativeTotalNormDispIncr', 7: 'RelativeEnergyIncr', 8: 'FixedNumIter'}

    # Algorithm Types
    algoType = {1: 'Newton', 2: 'ModifiedNewton', 3: 'KrylovNewton', 4: 'Broyden', 5: 'NewtonLineSearch', 6: 'BFGS'}

    # Integrator Types
    intType = {1: 'DisplacementControl', 2: 'LoadControl', 3: 'Parallel DisplacementControl',
               4: 'Minimum Unbalanced Displacement Norm', 5: 'Arc-Length Control'}

    tolInit = 1.0e-7  # Set the initial Tolerance, so it can be referred back to
    iterInit = 50  # Set the initial Max Number of Iterations

    # test(testType, *testArgs)    e.g. test('NormUnbalance', tol, iter, pFlag=0, nType=2, maxincr=-1)
    ops.test(testType[2], tolInit, iterInit)

    # algorithm(algoType, *algoArgs)   e.g. algorithm('Newton', secant=False, initial=False, initialThenCurrent=False)
    ops.algorithm(algoType[1])
    disp = dref * mu
    dU = disp / (1.0 * nSteps)
    ops.integrator(intType[1], ctrlNode, dispDir, dU)
    ops.analysis('Static')

    # Print values
    if IOflag >= 1:
        print("SinglePush: Push node %d to %.3f m" % (ctrlNode, mu * dref))

    # Set the initial values to start the while loop
    ok = 0.0
    step = 1.0
    loadf = 1.0
    # This feature of disabling the possibility of having a negative loading has been included.
    # This has been adapted from a similar script by Prof. Garbaggio

    while step <= nSteps and ok == 0 and loadf > 0:
        ok = ops.analyze(1)
        loadf = ops.getTime()
        temp = ops.nodeDisp(ctrlNode, dispDir)

        # Print the current displacement
        if IOflag >= 2:
            print("Pushed", ctrlNode, "in", dispDir, "to", temp, "with", loadf)

        # If the analysis fails, try the following changes to achieve convergence
        # Analysis will be slower in here though...
        if ok != 0:
            print("Trying relaxed convergence ...")
            ops.test(testType[2], tolInit * 0.01, iterInit * 50)
            ok = ops.analyze(1)
            ops.test(testType[2], tolInit, iterInit)

        if ok != 0:
            print("Trying Newton with initial then current ...")
            ops.test(testType[2], tolInit * 0.01, iterInit * 50)
            ops.algorithm(algoType[1], 'initialThenCurrent')
            ok = ops.analyze(1)
            ops.algorithm(algoType[1])
            ops.test(testType[2], tolInit, iterInit)

        if ok != 0:
            print("Trying ModifiedNewton with initial ..")
            ops.test(testType[2], tolInit * 0.01, iterInit * 50)
            ops.algorithm(algoType[2], 'initial')
            ok = ops.analyze(1)
            ops.algorithm(algoType[1])
            ops.test(testType[2], tolInit, iterInit)

        if ok != 0:
            print("Trying KrylovNewton ...")
            ops.test(testType[2], tolInit * 0.01, iterInit * 50)
            ops.algorithm(algoType[3])
            ok = ops.analyze(1)
            ops.algorithm(algoType[1])
            ops.test(testType[2], tolInit, iterInit)

        if ok != 0:
            print("Perform a Hail Mary ...")
            ops.test(testType[8], tolInit, iterInit)
            ok = ops.analyze(1)

        temp = ops.nodeDisp(ctrlNode, dispDir)
        loadf = ops.getTime()
        step += 1.0

        LoadFactor.append(loadf)
        DispCtrlNode.append(temp)

    if ok != 0:
        print("DispControl Analysis is FAILED")
        print('-------------------------------------------------------------------------')
    # answer = input("Do you wish to continue y/n ?") # not recommended in parameter study
    # if answer == "y":
    #     # Do nothing.
    # elif answer == "n":
    #     break # as it interrupts batch file
    else:
        print("DispControl Analysis is SUCCESSFUL")
        print('-------------------------------------------------------------------------')

    if loadf <= 0:
        print("Stopped because of Load factor below zero:", loadf)

    return LoadFactor, DispCtrlNode


def CyclicPush(dref, mu, numCycles, ctrlNode, dispDir, dispIncr, IOflag=0):
    """
    Procedure to carry out a cyclic pushover of a model
    Args:
        dref:         Reference displacement to which cycles are run. Corresponds to yield or equivalent other.
        mu:           Multiple of dref to which the cycles is run.
        numCycles:    No. of cycles. Valid options either 1,2,3,4,5,6
        ctrlNode:     Node to control with the displacement integrator
        dispDir:      Direction the loading is applied.
        dispIncr:     Number of displacement increments.
        IOflag:       Option to print cycle details on screen. 0 for off, 1 for on
    Returns:
        LoadFactor: Load factor
        DispCtrlNode: Displacement of control node
    """

    LoadFactor = [0]
    DispCtrlNode = [0]

    testType = {1: 'NormUnbalance', 2: 'NormDispIncr', 3: 'EnergyIncr', 4: 'RelativeNormUnbalance',
                5: 'RelativeNormDispIncr', 6: 'RelativeTotalNormDispIncr', 7: 'RelativeEnergyIncr', 8: 'FixedNumIter'}

    # Algorithm Types
    algoType = {1: 'Newton', 2: 'ModifiedNewton', 3: 'KrylovNewton', 4: 'Broyden', 6: 'ModifiedNewton',
                7: 'NewtonLineSearch'}

    # Integrator Types
    intType = {1: 'DisplacementControl', 2: 'LoadControl', 3: 'Parallel DisplacementControl',
               4: 'Minimum Unbalanced Displacement Norm', 5: 'Arc-Length Control'}

    tolInit = 1.0e-7  # Set the initial Tolerance, so it can be referred back to
    iterInit = 500  # Set the initial Max Number of Iterations

    # test(testType, *testArgs)    e.g. test('NormUnbalance', tol, iter, pFlag=0, nType=2, maxincr=-1)
    ops.test(testType[2], tolInit, iterInit)

    # algorithm(algoType, *algoArgs)   e.g. algorithm('Newton', secant=False, initial=False, initialThenCurrent=False)
    ops.algorithm(algoType[1])

    # Create the list of displacements
    if numCycles == 1:
        dispList = [dref * mu, -2 * dref * mu, dref * mu]
        dispNoMax = 3
    elif numCycles == 2:
        dispList = [dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu]
        dispNoMax = 6
    elif numCycles == 3:
        dispList = [dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu]
        dispNoMax = 9
    elif numCycles == 4:
        dispList = [dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu]
        dispNoMax = 12
    elif numCycles == 5:
        dispList = [dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu]
        dispNoMax = 15
    elif numCycles == 6:
        dispList = [dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu,
                    dref * mu, -2 * dref * mu, dref * mu]
        dispNoMax = 18
    else:
        print("ERROR: Value for numCycles not a valid choice. Choose between 1 and 6")
        print('-------------------------------------------------------------------------')

        # Print values
    if IOflag >= 1:
        print("CyclicPush: %s cycles to mu = %s at %s" % (numCycles, mu, ctrlNode))

    # Carry out loading
    for d in range(1, dispNoMax + 1, 1):
        numIncr = dispIncr
        dU = dispList[d - 1] / (1.0 * numIncr);
        ops.integrator(intType[1], ctrlNode, dispDir, dU)
        ops.analysis('Static')

        for l in range(0, numIncr, 1):
            # print("Analysis step:", l)
            ok = ops.analyze(1)
            LoadFactor.append(ops.getTime())
            DispCtrlNode.append(ops.nodeDisp(ctrlNode, dispDir))
            if ok != 0:
                print("DispControl Analysis is FAILED")
                print("Analysis failed at cycle: %s and dispIncr: %s" % (d, l))
                print('-------------------------------------------------------------------------')
                break

    if ok == 0:
        print("DispControl Analysis is SUCCESSFUL")
        print('-------------------------------------------------------------------------')

    return LoadFactor, DispCtrlNode


def SinglePush2(obj, dref, mu, ctrlNode, ctrlDOF, nSteps, IOflag=0):
    """
    Procedure to carry out a non-cylic pushover of a model
    The analysis strategy is slighly improved compared to the SinglePush routine
    It can be used to plot real time displaced shape
    Args:
        obj          Class object for the which contains information of the bridge
        dref:        Reference displacement to which cycles are run. Corresponds to yield or equivalent other, such as 1mm
        mu:          Multiple of dref to which the push is run. So pushover can be run to a specified ductility or displacement
        ctrlNode:    Node to control with the displacement integrator
        ctrlDOF:     DOF the loading is applied
        nSteps:      Number of steps
        IOflag:      Option to print details on screen (default = 0)
                     1; print analysis info at each analysis step
                     2: plot the deformed shape and pushover curve

    Returns:

    """

    if IOflag >= 2:
        scale = 10
        plt.figure(figsize=(16, 6))
        plt.tight_layout()
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 5])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], projection='3d')
        ax1.grid(True)
        ax1.set_xlabel('$u_{ctrl}$')
        ax1.set_ylabel('Load Factor, $\gamma$')

    print("SinglePush: Push node %d to %.3f m in DOF: %d" % (ctrlNode, mu * dref, ctrlDOF))
    LoadFactor = [0]
    DispCtrlNode = [0]
    # Test Types
    test = {0: 'EnergyIncr', 1: 'NormDispIncr', 2: 'RelativeEnergyIncr', 3: 'RelativeNormUnbalance',
            4: 'RelativeNormDispIncr', 5: 'NormUnbalance'}
    # Algorithm Types
    algorithm = {0: 'Newton', 1: 'KrylovNewton', 2: 'SecantNewton', 3: 'RaphsonNewton', 4: 'PeriodicNewton', 5: 'BFGS',
                 6: 'Broyden', 7: 'NewtonLineSearch'}
    # Integrator Types
    integrator = {0: 'DisplacementControl', 1: 'LoadControl', 2: 'Parallel DisplacementControl',
                  3: 'Minimum Unbalanced Displacement Norm', 4: 'Arc-Length Control'}

    tol = 1e-12  # Set the tolerance to use during the analysis
    iterInit = 10  # Set the initial max number of iterations
    maxIter = 1000  # Set the max number of iterations to use with other integrators
    disp = dref * mu
    dU = disp / (1.0 * nSteps)
    ops.test(test[0], tol, iterInit)  # lets start with energyincr as test
    ops.algorithm(algorithm[0])
    ops.integrator(integrator[0], ctrlNode, ctrlDOF, dU)
    ops.analysis('Static')

    # Set the initial values to start the while loop
    ok = 0.0
    step = 1.0
    loadf = 1.0
    # This feature of disabling the possibility of having a negative loading has been included.
    # This has been adapted from a similar script by Prof. Garbaggio

    i = 1  # counter for the tests, if the analysis fails starts with new test directly
    j = 0  # counter for the algorithm

    current_test = test[0]
    current_algorithm = algorithm[0]

    while step <= nSteps and ok == 0 and loadf > 0:
        ok = ops.analyze(1)

        # If the analysis fails, try the following changes to achieve convergence
        while ok != 0:
            if j == 7:  # this is the final algorithm to try, if the analysis did not converge
                j = 0;
                i += 1  # reset the algorithm to use
                if i == 6:  # we have tried everything
                    break

            j += 1  # change the algorithm

            if j < 3:
                ops.algorithm(algorithm[j], '-initial')

            else:
                ops.algorithm(algorithm[j])

            ops.test(test[i], tol, maxIter)
            ok = ops.analyze(1)
            current_test = test[i];
            current_algorithm = algorithm[j]

        temp1 = ops.nodeDisp(ctrlNode, 1)  # disp in dir 1
        temp2 = ops.nodeDisp(ctrlNode, 2)  # disp in dir 2
        temp = (temp1 ** 2 + temp2 ** 2) ** 0.5  # SRSS of disp in two dirs
        loadf = ops.getTime()
        step += 1.0
        LoadFactor.append(loadf)
        DispCtrlNode.append(temp)

        # Print the current displacement
        if IOflag >= 1:
            print('Current Test:', current_test, '| Current Algorithm:', current_algorithm,
                  "| Control Disp:", "{:.3f}".format(ops.nodeDisp(ctrlNode, ctrlDOF)),
                  "| Load Factor:", "{:.0f}".format(loadf))
        if IOflag >= 2:
            if step != 2:
                lines.pop(0).remove()
            ax2.clear()
            lines = ax1.plot(DispCtrlNode, LoadFactor, color='red')
            obj.plot_deformedshape(ax2, scale)
            plt.pause(0.001)

    if ok != 0:
        print("Displacement Control Analysis is FAILED")
        print('-------------------------------------------------------------------------')

    else:
        print("Displacement Control Analysis is SUCCESSFUL")
        print('-------------------------------------------------------------------------')

    if loadf <= 0:
        print("Stopped because of Load factor below zero:", loadf)
        print('-------------------------------------------------------------------------')

    return LoadFactor, DispCtrlNode


def nrha_single(obj, Dt, Tmax, Dc, log, pflag=0):
    """
    ----------------------------------------------------------------
    -- Script to Conduct 3D Non-linear Response History Analysis ---
    ----------------------------------------------------------------
    This procedure is a simple script that executes the NRHA of a 3D model. It
    requires that the model has the dynamic analysis objects defined and just the
    'analyze' of a regular OpenSees model needs to be executed. Therefore, the model
    needs to be built to the point where a modal analysis can be conducted. The
    ground motion timeSeries and pattern need to be setup and the constraints,
    numberer and system analysis objects also need to be predefined.

    When conducting the NRHA, this proc will try different options to achieve
    convergence and finish the ground motion. This allows for a relatively robust
    analysis algorithm to be implemented with a single command.

    In addition, the analysis requires that a deformation capacity be specified
    to define a limit that upon exceedance, will stop the NRHA and mark the
    analysis as a collapse case. It monitors the current deformation of a number
    of specified nodes and flags collapse based on their deforormation. This
    prevents the models getting 'stuck' trying to converge a model that has
    essentially collapsed, or would be deemed a collapse case in post processing.
    These are defined in terms of the pier drifts. For 3D analysis, the SRSS
    of absolute maximum drift in either direction is used.
    Other definitions are possible but not yet implemented.

    Lastly, a log file identifier is also required in the input. This will log
    all of the essential information about the maximum pier drifts. This script
    was developed for analysing bridges so the deformation capacity typically
    corresponds to a drift capacity and the top and bottom nodes would typically
    correspond to the centreline nodes of the bridge pier nodes.
    Args:
        Dt:       Analysis time step
        Tmax:     Length of the record (including padding of 0's)
        Dc:       Drift capacity for pier drift (%)
        log:      File handle of the logfile
        pflag:    Flag to print stuff if necessary

    Returns:

    """

    if pflag == 3:  # plotting option
        plt.figure(figsize=(19.2, 10.8))
        ax = plt.subplot(1, 1, 1, projection='3d')
        scale = 40                      
                      
    algorithms = [
        ['Newton', '-initial'],
        ['KrylovNewton'],  
        ['ModifiedNewton', '-initial'],
        ['NewtonLineSearch', '-InitialInterpolated', 0.8], 
        ['Broyden', 8]
                ]
    
    # Define the Initial Analysis Parameters
    testType = 'NormDispIncr'  # Set the initial test type (default)
    tolInit = 1.0e-6  # Set the initial Tolerance, so it can be referred back to (default)
    iterInit = 20  # Set the initial Max Number of Iterations (default)
    
    # Parameters required in Newmark Integrator, no numerical dissiaption
    # gamma = 1/2, beta = 1/4 --> Average Acceleration Method; unconditionally stable
    # gamma = 1/2, beta = 1/6 --> Linear Acceleration Method; conditionally stable dt/T <0.551
    # gamma = 0.5
    # beta = 0.25
    # ops.integrator('Newmark', gamma, beta)

    # Hail Mary... Bless the analysis with thy damping!
    # Parameters required in Hilber-Hughes-Taylor (HHT) integrator
    # alpha = 1.0 = Newmark Method. smaller alpha means greater numerical damping. 
    # 0.67<alpha<1.0 # recommended. Leave beta and gamma as default for unconditional stability.
    # alpha = 0.8
    # ops.integrator('HHT', alpha)

    # Parameters for GeneralizedAlpha integrator.
    # alphaF and alphaM are defined differently that in the paper, we use alpha_F = (1-alpha_f) and 
    # alpha_M=(1-alpha_m) where alpha_f and alpha_m are those used in the paper.
    # 1. Like Newmark and all the implicit schemes, the unconditional stability of this method 
    # applies to linear problems. There are no results showing stability of this method over the
    # wide range of nonlinear problems that potentially exist. Experience indicates that the time 
    # step for implicit schemes in nonlinear situations can be much greater than those for explicit schemes.
    # 2. alphaM = 1.0, alphaF = 1.0 produces the Newmark Method.
    # 3. alphaM = 1.0 corresponds to the HHT method.
    # 4. The method is second-order accurate provided gamma=0.5+alphaM-alphaF, beta=(1+alphaM-alphaF)**2/4
    # These are optional parameters that can be used, but default values satisfy this condition
    # 5. The method is unconditionally stable provided alphaM >= alphaF >= 0.5, beta >= 0.25+0.5*(alphaM-alphaF)
    # The following relationships minimize low-frequency damping and maximize high-frequency damping
    # pinf = 1 no dissipation of high frequency response 
    # pinf = 0 annihilation of high-frequency response
    pinf = 0.4
    alpha_m = (2*pinf-1)/(pinf+1)
    alpha_f = pinf/(1+pinf)
    alphaM = 1-alpha_m
    alphaF = 1-alpha_f
    ops.integrator('GeneralizedAlpha', alphaM, alphaF)

    # Set up analysis parameters
    cIndex = 0  # Initially define the control index (-1 for non-converged, 0 for stable, 1 for global collapse)
    controlTime = 0.0  # Start the controlTime
    ok = 0  # Set the convergence to 0 (initially converged)
    mflr = 0  # Set the initial pier collapse location
    Mdrft = 0.0  # Set initially the maximum of all pier drifts (SRSS)
    Dtt = Dt/1 # analysis time step

    # Set up the pier drift and acceleration values
    h = []
    mdrftX = []
    mdrftY = []
    mdrft = []
    
    forces = []
    disp = []
    forces2 = []
    disp2 = []

    tNode = []
    bNode = []

    for bent in obj.EleIDsBent:
        for pier in bent:
            eleNodes = ops.eleNodes(pier)
            bNode.append(eleNodes[0])
            tNode.append(eleNodes[1])
            
    for i in range(len(tNode)):
        # Find the coordinates of the nodes in Global Z (3)
        top2 = ops.nodeCoord(tNode[i], 3)
        bot2 = ops.nodeCoord(bNode[i], 3)
        dist = top2 - bot2

        # This means we take the distance in Z (3) in my coordinates systems at least. This is X-Y/Z| so X=1 Y=2 Z=3.
        # (gli altri vacca gare)
        h.append(dist)  # Current pier height
        mdrftX.append(0.0)  # We will populate the lists with zeros initially
        mdrftY.append(0.0)
        mdrft.append(0.0)
        if dist == 0: print("WARNING: Zerolength found in drift check")

    # Run the actual analysis now
    while cIndex == 0 and controlTime <= Tmax and ok == 0:
        ops.test(testType, tolInit, iterInit, 0, 2)
        ops.algorithm(*algorithms[0])
        ops.analysis('Transient')

        # Gradually increase the time increment size
        if Dtt == Dt/4:
            Dtt = Dt/2
        elif Dtt == Dt/2:
            Dtt = Dt/1

        # Do the analysis
        ok = ops.analyze(1, Dtt)  # Run a step of the analysis
        controlTime = ops.getTime()  # Update the control time
        if pflag > 1: print("Completed %.2f of %.2f seconds" % (controlTime, Tmax))

        # If the analysis fails, try the following changes to achieve convergence
        # Analysis will be slower in here though...
        
        # Reduce analysis the time increment size
        if ok != 0 and Dtt>Dt/2:
            print(" ~~~ Failed at %.2f - Reduced timestep by half..." % controlTime)
            Dtt = Dt/2
            ok = ops.analyze(1, Dtt)
            if ok == 0:
                print(' ~~~ The reduced timestep has worked ', Dtt)

        if ok != 0 and Dtt>Dt/4:
            print(" ~~~ Failed at %.2f - Reduced timestep by quarter..." % controlTime)
            Dtt = Dt/4
            ok = ops.analyze(1, Dtt)
            if ok == 0:
                print(' ~~~ The reduced timestep has worked: ', Dtt)       

        # Change the solution algorithm
        if ok != 0:
            ops.test(testType, iterInit*5, 0, 2) # lets increase the number of iterations for other algorithms than regular newton
            for algorithmType in algorithms[1:]:
                print(" ~~~ Failed at %.2f - Moving to the next algorithm..." % controlTime)
                ops.algorithm(*algorithmType)
                ok = ops.analyze(1, Dtt)
                if ok == 0:
                    print('The algorithm has worked: ', algorithmType)
                    break
        
        if ok != 0:
            # Next change both algorithm and tolerance to achieve convergence if this doesn't work
            # in bocca al lupo....
            print(" ~~~ Failed at %.2f - Relaxing the convergence criteria and increasing maximum number of iterations..." % controlTime)
            ops.test(testType, tolInit * 0.1, iterInit * 50, 0, 2)
            for algorithmType in algorithms:
                print(" ~~~ Failed at %.2f - Moving to the next algorithm..." % controlTime)
                ops.algorithm(*algorithmType)
                ok = ops.analyze(1, Dtt)
                if ok == 0:
                    print(' ~~~ The relaxed convergence criteria has worked.',)
                    break

        # Shit...  Failed to converge, exit the analysis.
        if ok != 0:
            print(" ~~~ Failed at %.2f - exit the analysis......" % controlTime)
            ops.wipe()
            cIndex = -1

        if ok == 0:
            # Check the pier drifts
            for i in range(len(tNode)):
                tNode_dispX = ops.nodeDisp(tNode[i], 1)  # Current top node disp in X
                tNode_dispY = ops.nodeDisp(tNode[i], 2)  # Current top node disp in Y
                bNode_dispX = ops.nodeDisp(bNode[i], 1)  # Current bottom node disp in X
                bNode_dispY = ops.nodeDisp(bNode[i], 2)  # Current bottom node disp in Y
                cHt = h[i]  # Current pier height
                cdrftX = 100.0 * abs(
                    tNode_dispX - bNode_dispX) / cHt  # Current pier drift in X at the current pier in %
                cdrftY = 100.0 * abs(
                    tNode_dispY - bNode_dispY) / cHt  # Current pier drift in X at the current pier in %
                cdrft = ((cdrftX ** 2) + (cdrftY ** 2)) ** 0.5  # SRSS of two drift components
                if cdrftX >= mdrftX[i]: mdrftX[i] = cdrftX
                if cdrftY >= mdrftY[i]: mdrftY[i] = cdrftY
                if cdrft >= mdrft[i]: mdrft[i] = cdrft
                if cdrft > Mdrft: Mdrft = cdrft; mflr = i + 1  # Update the current maximum pier drift and where it is

            # Test gap element
            # forces.append(ops.basicForce(obj.EleIDsGap[0][3]))
            # disp.append(ops.basicDeformation(obj.EleIDsGap[0][3]))
            # forces2.append(ops.basicForce(obj.EleIDsBearing[0][3])[2])
            # disp2.append(ops.basicDeformation(obj.EleIDsBearing[0][3])[2])

            if Mdrft >= Dc: 
                cIndex = 1 
                Mdrft = Dc
                ops.wipe()  # Set the state of the model to local collapse (=1)

        if pflag == 3:
            ax.clear()
            obj.plot_deformedshape(ax, scale)
            ax.text2D(0.10, 0.85, "Time: " + "{:.2f}".format(controlTime) + "/" + "{:.2f}".format(Tmax) + ' sec',
                      transform=ax.transAxes, fontweight="bold")
            plt.pause(0.001)
    if cIndex == -1:
        Analysis = "Analysis is FAILED to converge at %.3f of %.3f" % (controlTime, Tmax)
    if cIndex == 0:
        Analysis = "Analysis is SUCCESSFULLY completed\nPeak Pier Drift: %.2f%% at Pier %d" % (Mdrft, mflr)
    if cIndex == 1:
        Analysis = "Analysis is STOPPED, peak column drift ratio, %d%%, is exceeded, global COLLAPSE is observed" % Dc

    if ops.getNP() == 1:
        print(Analysis)

    if pflag > 0:
        # Create some output
        f = open(log, "w+")
        f.write(Analysis + '\n')  # Print to the logfile the analysis state

        # Print to the max interpier drifts
        f.write("Peak Pier DriftX: ")
        for i in range(len(mdrftX)):
            f.write("%.2f " % mdrftX[i])
        f.write("%\n")
        f.write("Peak Pier DriftY: ")
        for i in range(len(mdrftY)):
            f.write("%.2f " % mdrftY[i])
        f.write("%\n")
        f.write("Peak Pier Drift: ")
        for i in range(len(mdrft)):
            f.write("%.2f " % mdrft[i])
        f.write("%")
        f.close()
        
    # Test gap element
    # plt.figure()
    # plt.plot(disp,forces)
    # plt.plot(disp2,forces2)
    
    return Mdrft, cIndex, mflr, Analysis


def nrha_multiple(obj, Dt, Tmax, Dc, log, pflag=0):
    """
    ----------------------------------------------------------------
    -- Script to Conduct 3D Non-linear Response History Analysis ---
    ----------------------------------------------------------------

    The analysis procedure is essentially the same with previous procedure.
    However, unlike previous procedure, it returns multiple engineering demand parameters
    instead of Single one which is the previous pier drift ratio

    Args:
        obj:      bridge object
        Dt:       Analysis time step
        Tmax:     Length of the record (including padding of 0's)
        Dc:       Drift capacity for pier drift (%)
        log:      File handle of the logfile
        pflag:    Flag to print stuff if necessary

    Returns:

    """

    # For 3D fiber sections dofs are: [P,Mz,My,T]
    # If the section is aggregated with shear springs then dofs are: [P,Vz,Vy,Mz,My,T]

    # Initialize the engineering demand parameters to calculate
    h = []  # pier heights
    mdrft1 = []  # maximum drift in dir 1
    mdrft2 = []  # maximum drift in dir 2
    mdrft = []  # maximum drift SRSS

    mudisp1 = []  # maximum displacement ductility in 1
    mudisp2 = []  # maximum displacement ductility in 2
    mudisp = []  # maximum displacement ductility SRSS

    muK1 = []  # maximum curvature ductility in Mz   y
    muK2 = []  # maximum curvature ductility in My   |_z
    muK = []  # maximum curvature ductility SRSS

    tNode = []  # Pier element top Nodes
    bNode = []  # Pier element bottom Nodes
    
    EleIDs = [] # Pier element IDs
    Ky1 = [] # Pier element yield curvatures in dir 1
    Ky2 = [] # Pier element yield curvatures in dir 2
    Dispy1 = [] # Pier element yield displacements in dir 1
    Dispy2 = [] # Pier element yield displacements in dir 2

    abut1 = 0		# Displacement of start abutment node
    abut2 = 0		# Displacement of end abutment node

    for i in range(len(obj.EleIDsBent)):
        for j in range(len(obj.EleIDsBent[i])):
            eleNodes = ops.eleNodes(obj.EleIDsBent[i][j])
            bNode.append(eleNodes[0])
            tNode.append(eleNodes[1])
            EleIDs.append(obj.EleIDsBent[i][j])
            h.append(obj.BentHeight[i][j])
            Ky1.append(obj.BentKy1[i][j])
            Ky2.append(obj.BentKy2[i][j])
            Dispy1.append(obj.BentDispy1[i][j])
            Dispy2.append(obj.BentDispy2[i][j])             
            mdrft1.append(0.0)
            mdrft2.append(0.0)
            mdrft.append(0.0)
            mudisp1.append(0.0)
            mudisp2.append(0.0)
            mudisp.append(0.0)
            muK1.append(0.0)
            muK2.append(0.0)
            muK.append(0.0)
                
    algorithms = [
        ['Newton', '-initial'],
        ['KrylovNewton'],  
        ['ModifiedNewton', '-initial'],
        ['NewtonLineSearch', '-InitialInterpolated', 0.8], 
        ['Broyden', 8]
                ]
    
    # Define the Initial Analysis Parameters
    testType = 'NormDispIncr'  # Set the initial test type (default)
    tolInit = 1.0e-6  # Set the initial Tolerance, so it can be referred back to (default)
    iterInit = 20  # Set the initial Max Number of Iterations (default)
    
    # Parameters required in Newmark Integrator, no numerical dissiaption
    # gamma = 1/2, beta = 1/4 --> Average Acceleration Method; unconditionally stable
    # gamma = 1/2, beta = 1/6 --> Linear Acceleration Method; conditionally stable dt/T <0.551
    # gamma = 0.5
    # beta = 0.25
    # ops.integrator('Newmark', gamma, beta)

    # Hail Mary... Bless the analysis with thy damping!
    # Parameters required in Hilber-Hughes-Taylor (HHT) integrator
    # alpha = 1.0 = Newmark Method. smaller alpha means greater numerical damping. 
    # 0.67<alpha<1.0 # recommended. Leave beta and gamma as default for unconditional stability.
    # alpha = 0.67
    # ops.integrator('HHT', alpha)

    # Parameters for GeneralizedAlpha integrator.
    # alphaF and alphaM are defined differently that in the paper, we use alpha_F = (1-alpha_f) and 
    # alpha_M=(1-alpha_m) where alpha_f and alpha_m are those used in the paper.
    # 1. Like Newmark and all the implicit schemes, the unconditional stability of this method 
    # applies to linear problems. There are no results showing stability of this method over the
    # wide range of nonlinear problems that potentially exist. Experience indicates that the time 
    # step for implicit schemes in nonlinear situations can be much greater than those for explicit schemes.
    # 2. alphaM = 1.0, alphaF = 1.0 produces the Newmark Method.
    # 3. alphaM = 1.0 corresponds to the HHT method.
    # 4. The method is second-order accurate provided gamma=0.5+alphaM-alphaF, beta=(1+alphaM-alphaF)**2/4
    # These are optional parameters that can be used, but default values satisfy this condition
    # 5. The method is unconditionally stable provided alphaM >= alphaF >= 0.5, beta >= 0.25+0.5*(alphaM-alphaF)
    # The following relationships minimize low-frequency damping and maximize high-frequency damping
    # pinf = 1 no dissipation of high frequency response 
    # pinf = 0 annihilation of high-frequency response
    pinf = 0.4
    alpha_m = (2*pinf-1)/(pinf+1)
    alpha_f = pinf/(1+pinf)
    alphaM = 1-alpha_m
    alphaF = 1-alpha_f
    ops.integrator('GeneralizedAlpha', alphaM, alphaF)

    # Set up analysis parameters
    cIndex = 0  # Initially define the control index (-1 for non-converged, 0 for stable, 1 for global collapse)
    controlTime = 0.0  # Start the controlTime
    ok = 0  # Set the convergence to 0 (initially converged)
    mflr = 0  # Set the initial pier collapse location
    Mdrft = 0.0  # Set initially the maximum of all pier drifts (SRSS)
    Dtt = Dt/1 # analysis time step

    # Run the actual analysis now
    while cIndex == 0 and controlTime <= Tmax and ok == 0:
        ops.test(testType, tolInit, iterInit, 0, 2)
        ops.algorithm(*algorithms[0])
        ops.analysis('Transient')

        # Gradually increase the time increment size
        if Dtt == Dt/4:
            Dtt = Dt/2
        elif Dtt == Dt/2:
            Dtt = Dt/1

        # Do the analysis
        ok = ops.analyze(1, Dtt)  # Run a step of the analysis
        controlTime = ops.getTime()  # Update the control time
        if pflag > 1: print("Completed %.2f of %.2f seconds" % (controlTime, Tmax))

        # If the analysis fails, try the following changes to achieve convergence
        # Analysis will be slower in here though...
        
        # Reduce analysis the time increment size
        if ok != 0 and Dtt>Dt/2:
            print(" ~~~ Failed at %.2f - Reduced timestep by half..." % controlTime)
            Dtt = Dt/2
            ok = ops.analyze(1, Dtt)
            if ok == 0:
                print(' ~~~ The reduced timestep has worked ', Dtt)

        if ok != 0 and Dtt>Dt/4:
            print(" ~~~ Failed at %.2f - Reduced timestep by quarter..." % controlTime)
            Dtt = Dt/4
            ok = ops.analyze(1, Dtt)
            if ok == 0:
                print(' ~~~ The reduced timestep has worked: ', Dtt)       

        # Change the solution algorithm
        if ok != 0:
            ops.test(testType, iterInit*5, 0, 2) # lets increase the number of iterations for other algorithms than regular newton
            for algorithmType in algorithms[1:]:
                print(" ~~~ Failed at %.2f - Moving to the next algorithm..." % controlTime)
                ops.algorithm(*algorithmType)
                ok = ops.analyze(1, Dtt)
                if ok == 0:
                    print('The algorithm has worked: ', algorithmType)
                    break
        
        if ok != 0:
            # Next change both algorithm and tolerance to achieve convergence if this doesn't work
            # in bocca al lupo....
            print(" ~~~ Failed at %.2f - Relaxing the convergence criteria and increasing maximum number of iterations..." % controlTime)
            ops.test(testType, tolInit * 0.1, iterInit * 50, 0, 2)
            for algorithmType in algorithms:
                print(" ~~~ Failed at %.2f - Moving to the next algorithm..." % controlTime)
                ops.algorithm(*algorithmType)
                ok = ops.analyze(1, Dtt)
                if ok == 0:
                    print(' ~~~ The relaxed convergence criteria has worked.',)
                    break

        # Shit...  Failed to converge, exit the analysis.
        if ok != 0:
            print(" ~~~ Failed at %.2f - exit the analysis......" % controlTime)
            ops.wipe()
            cIndex = -1

        if ok == 0:

            for i in range(len(EleIDs)):
                tNode_disp1 = ops.nodeDisp(tNode[i], 1)  # Current top node disp in 1
                tNode_disp2 = ops.nodeDisp(tNode[i], 2)  # Current top node disp in 2
                bNode_disp1 = ops.nodeDisp(bNode[i], 1)  # Current bottom node disp in 1
                bNode_disp2 = ops.nodeDisp(bNode[i], 2)  # Current bottom node disp in 2

                # Get the pier displacement ductility and check if it is maximum
                cdisp1 = abs(tNode_disp1 - bNode_disp1) / Dispy1[i]
                cdisp2 = abs(tNode_disp2 - bNode_disp2) / Dispy2[i]
                cdisp = ((cdisp1 ** 2) + (cdisp2 ** 2)) ** 0.5
                if cdisp1 >= mudisp1[i]: mudisp1[i] = cdisp1
                if cdisp2 >= mudisp2[i]: mudisp2[i] = cdisp2
                if cdisp >= mudisp[i]: mudisp[i] = cdisp

                # Get the pier curvature ductility and check if it is maximum
                ccurv1 = ops.sectionDeformation(EleIDs[i], 1, 1) / Ky1[i]
                ccurv2 = ops.sectionDeformation(EleIDs[i], 1, 2) / Ky2[i]
                ccurv = ((ccurv1 ** 2) + (ccurv2 ** 2)) ** 0.5
                if ccurv1 >= muK1[i]: muK1[i] = ccurv1
                if ccurv2 >= muK2[i]: muK2[i] = ccurv2
                if ccurv >= muK[i]: muK[i] = ccurv

                # Check the pier drifts, and save
                cdrft1 = 100.0 * abs(tNode_disp1 - bNode_disp1) / h[i]  # Current pier drift in 1 at the current pier in %
                cdrft2 = 100.0 * abs(tNode_disp2 - bNode_disp2) / h[i]  # Current pier drift in 2 at the current pier in %
                cdrft = ((cdrft1 ** 2) + (cdrft2 ** 2)) ** 0.5  # SRSS of two drift components
                if cdrft1 >= mdrft1[i]: mdrft1[i] = cdrft1
                if cdrft2 >= mdrft2[i]: mdrft2[i] = cdrft2
                if cdrft >= mdrft[i]: mdrft[i] = cdrft
                if cdrft > Mdrft: Mdrft = cdrft; mflr = i + 1  # Update the current maximum pier drift and where it is

            if obj.model['Abutment_BackFill']['Type'] != 'None':
                # Abutment Displacements
                for i in range(len(obj.EleIDsAB1)):
                    disps1 = ops.basicDeformation(obj.EleIDsAB1[i])
                    abut1_temp = ((disps1[0] ** 2) + (disps1[1] ** 2)) ** 0.5
                    if abut1_temp > abut1: abut1 = abut1_temp

                    disps2 = ops.basicDeformation(obj.EleIDsAB2[i])
                    abut2_temp = ((disps2[0] ** 2) + (disps2[1] ** 2)) ** 0.5
                    if abut2_temp > abut2: abut2 = abut2_temp

            if Mdrft >= Dc: cIndex = 1; Mdrft = Dc; ops.wipe()  # Set the state of the model to local collapse (=1)

    if cIndex == -1:
        Analysis = "Analysis is FAILED to converge at %.3f of %.3f" % (controlTime, Tmax)
    if cIndex == 0:
        Analysis = "Analysis is SUCCESSFULLY completed\nPeak Pier Drift: %.2f%% at Pier %d" % (Mdrft, mflr)
    if cIndex == 1:
        Analysis = "Analysis is STOPPED, peak column drift ratio, %d%%, is exceeded, global COLLAPSE is observed" % Dc

    if ops.getNP() == 1:
        print(Analysis)

    return Mdrft, cIndex, mflr, mdrft, mudisp, muK, Analysis, [abut1, abut2]
