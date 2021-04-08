# In this script unitary costs for the materials are given
# Reference:
# A.N.A.S. S.p.A. [2018] LISTINO PREZZI 2018, NC-MS.2018 – REV.0, Nuove Costruzioni –
# Manutenzione Straordinaria, Direzione Ingegneria e Verifiche, Roma, Italy. 
# (in Italian) available at http://www.stradeanas.it/it/elenco-prezzi

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

def PSTEELCOST(): # post-tensioning steel costs (EUR/kg)
    costpsteel=2.59 # ANAS unitary price list B.05.057 and B.05.060
    return costpsteel

def RAILING(): #railing barrier costs (EUR/kg)
    costrail=2.15 # ANAS unitary price list B.05.017c
    return costrail

def ASPHALT(): # asphalt costs (EUR/m3)
    costasphalt=95 # Assumed
    return costasphalt

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

def PILECOST(DPile): # pile construction cost (EUR/m)
    if DPile==1.5:
        costpilecons=305.15+176.61 # ANAS Unitary price list B.02.035.d and B.02.046.d 
    elif DPile==1.0:
        costpilecons=151.73 # ANAS Unitary price list B.02.035.b 
    elif DPile==0.8:
        costpilecons=108.03 # ANAS Unitary price list B.02.035.a
    return costpilecons

def FILL(): # landfill cost (EUR/m3)
    # Assumed
    fillcost=70
    return fillcost

def EXCA(): # land excavation costs (EUR/m3)
    exccost=3.25 # ANAS Unitary price list A.01.001
    return exccost

def DEMOL(t): # demolition costs (EUR/m3)
    if t==1:
        demolcost=99.68 # ANAS A.03.008, demolition of decks
    elif t==2:
        demolcost=25.3 # ANAS A.03.019, Demolition of substrucure elements
    elif t==3:
        demolcost=180.2 # ANAS A.03.007, demolition of foundation elements
    return demolcost

def CLNCOST(): # Unitary cost for cleaning and superficial treatment (EUR/m2) 
    cleaningcost=21.18 # ANAS Unitary price list B.09.212
    return cleaningcost

def CONPATCHCOST(): # Unitary cost for concrete patch (EUR/m3)
    concretepatchcost=178.76  # ANAS Unitary price list B.04.003
    return concretepatchcost

def CRACKSEALCOST(): # Unitary cost for crack sealing (EUR/m)
    concretesealingcost = 191 # Assumed
    return concretesealingcost

def DEMOLCOVERCOST(): # Unitary cost of demolition of cover concrete (EUR/m3)
    conccoverdemolcost=289.84 # ANAS Unitary price list ANAS A.03.007
    return conccoverdemolcost

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
        