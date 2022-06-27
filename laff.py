from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from regex import P
from scipy.odr import ODR, Model, RealData
import scipy.integrate as integrate
import laffmodels
import numpy as np
# from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")

filepath = 'data/grb210112a.qdp' # data filepath
riseRequirement = 2 # how much data should rise to flag as potential flare
decayRequirement = 4 # variable affecting how easy it is to end decay
showFlares = True # show the excluded flare data?
showComponents = False # show the individual model components?
showRatio = True # if not, show residuals
showPlot = True # in case you just want output without plotting the result

#####################
# USEFUL FUNCTIONS
#####################

# returns the parameter value from the data
def tableValue(input_data,index,column):
    return input_data['%s' % column].iloc[index]

# take a list and remove all duplicates
def uniqueList(duplicate_list):
    unique_list = list(set(duplicate_list))
    unique_list.sort()
    return unique_list

# function to apply the ODR fitter
def modelFit(datapoints,laffmodel,inputpar):
    model = Model(laffmodel)
    odr = ODR(datapoints, model, inputpar)
    odr.set_job(fit_type=0)
    output = odr.run()

    if output.info != 1:
        i =1
        while output.info != 1 and i < 100:
            output = odr.restart()
            i += 1
    return output, output.beta

###############################################################
# LOAD DATA
###############################################################

# load data by table id (.qdp files are split between observation types)
table_0 = Table.read(filepath, format='ascii.qdp', table_id=0)
table_1 = Table.read(filepath, format='ascii.qdp', table_id=1)
table_2 = Table.read(filepath, format='ascii.qdp', table_id=2)
table_0['tableID'], table_1['tableID'], table_2['tableID'] = 0, 1, 2

# combine tables, sort by time and reset indexes
data = vstack([table_0, table_1, table_2]).to_pandas()
data = data.sort_values(by=['col1'])
data = data.reset_index(drop=True)
data = data.rename(columns={'col1': 'time', 'col1_perr': 'time_perr', \
    'col1_nerr': 'time_nerr', 'col2': 'flux', 'col2_perr': 'flux_perr', \
    'col2_nerr': 'flux_nerr'})
data['flare'] = False

###############################################################
# IDENTIFY POTENTIAL FLARES
###############################################################

# look through the next few flux values and see if they generally increase
def flareFinder(data,index):
    ahead = []
    for i in range(8):
        try:
            ahead.append(tableValue(data,index+i,"flux"))
        except:
            pass
    counter = 0
    for check in ahead:
        if tableValue(data,index,"flux") + (tableValue(data,index,"flux_perr")*riseRequirement) < check:
            counter += 1
    if counter >= 6:
        return True
    else:
        counter = 0
        return False

possible_flares = []

# if enough increase, mark as a possible flare
for index in data.index[data.time < 2000]:
    if flareFinder(data,index) == True:
        possible_flares.append(index)

possible_flares = uniqueList(possible_flares)

###############################################################
# REFINE FLARE START POSITION
###############################################################

index_start = []

# for each possible peak, look for a nearby minima as the flare start
for peak in possible_flares:
    values_start = []
    for n in range(-10, 1):
        if (peak+n) >= 0:
            values_start.append([tableValue(data,peak+n,"flux"),n])
    toAdjust = min(values_start)[1]
    index_start.append(peak + toAdjust)

index_start = uniqueList(index_start)

###############################################################
# FIND FLARE PEAK
###############################################################

index_peak = []

# for each start, filter through next values looking for the peak
for start in index_start:
    values_peak = []
    for n in range(0,50):
        values_peak.append([tableValue(data,start+n,"flux"),n])
    toAdjust = max(values_peak)[1]
    index_peak.append(start+toAdjust)

index_peak = uniqueList(index_peak)

###############################################################
# FIND DECAY END
###############################################################

possible_decay = []

# if gradient is increasing mark this as end of decay
for peak in index_peak:
    i = peak
    endDecay = False
    while endDecay == False:

        def grad_Peak(data,iter,peak):
            delta_flux = tableValue(data,iter,"flux") - tableValue(data,peak,"flux")
            delta_time = tableValue(data,iter,"time") - tableValue(data,peak,"time")
            return delta_flux/delta_time

        def grad_Next(data,iter):
            delta_flux = tableValue(data,iter+1,"flux") - tableValue(data,iter,"flux")
            delta_time = tableValue(data,iter+1,"time") - tableValue(data,iter,"time")
            return delta_flux/delta_time

        i += 1
        condition = 0

        for interval in data.index[(i):(i+10)]:
            current_Peak = grad_Peak(data,interval,peak)
            current_Next = grad_Next(data,interval)

            if endDecay == False:
                if (tableValue(data,interval,"time") > 2000) or (interval in index_start):
                    possible_decay.append(interval-1)
                    endDecay = True

            if endDecay == False:
                if current_Peak < current_Next:
                    if current_Peak > grad_Peak(data,interval-1, peak) and current_Next > grad_Next(data,interval-1):
                        condition += 1
                if condition > decayRequirement:
                    possible_decay.append(interval)
                    endDecay = True

possible_decay = uniqueList(possible_decay)

###############################################################
# REFINE DECAY END POSITION
###############################################################

index_decay = []

# 
for end in possible_decay:
    values_end = []
    n = 0
    while (end+n) not in index_peak:
        values_end.append([tableValue(data,end+n,"flux"),n])
        n += -1
    toAdjustDecay = min(values_end)[1]
    index_decay.append(end + toAdjustDecay)

index_decay = uniqueList(index_decay)

###############################################################
# FLARE ASSIGNING
###############################################################

# package the flare data nicely
flareIndexes = []
for start, peak, decay in zip(index_start, index_peak, index_decay):
    flareIndexes.append([start,peak,decay])
flareIndexes = flareIndexes[0:-1] # temporarily remove the last flare


# assign flares to table
for start, peak, decay in flareIndexes:
    rise_start = data.index >= start
    decay_end = data.index < decay
    data['flare'][rise_start & decay_end] = True

flares = data.flare == True

###############################################################
# POWERLAW FITTING
###############################################################

# define data
data_noflare = RealData(data.time[~flares], data.flux[~flares], data.time_perr[~flares], data.flux_perr[~flares])

# visually estimate the breaks
b1, b2, b3, b4, b5 = 120, 200, 630, 9500, 120000
a1, a2, a3, a4, a5, a6 = 1, 1, 1, 1, 1, 1
norm = 1e-7

# fit through 5 breaks
brk1_fit, brk1_param = modelFit(data_noflare, laffmodels.powerlaw_1break, [a1, a2, b1, norm])
brk2_fit, brk2_param = modelFit(data_noflare, laffmodels.powerlaw_2break, [a1, a2, a3, b1, b2, norm])
brk3_fit, brk3_param = modelFit(data_noflare, laffmodels.powerlaw_3break, [a1, a2, a3, a4, b2, b3, b4, norm])
brk4_fit, brk4_param = modelFit(data_noflare, laffmodels.powerlaw_4break, [a1, a2, a3, a4, a5, b2, b3, b4, b5, norm])
brk5_fit, brk5_param = modelFit(data_noflare, laffmodels.powerlaw_5break, [a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, norm])

# determine the best fit - WIP
bknpower_model = laffmodels.powerlaw_4break
bknpower_param = brk4_param

residuals = data.flux - bknpower_model(bknpower_param, np.array(data.time))

###############################################################
# FLARE FITTING
###############################################################

flareFits = []

# fit the flares
for start, peak, decay in flareIndexes:
    data_flare = RealData(data.time[start:decay+1], residuals[start:decay+1], data.time_perr[start:decay+1], data.flux_perr[start:decay+1])
    flare_fit, flare_param = modelFit(data_flare, laffmodels.flare_gaussian, [tableValue(data,peak,"flux"), tableValue(data,peak,"time"),tableValue(data,decay,"time")-tableValue(data,start,"time")])
    flareFits.append(flare_param)

###############################################################
# TOTAL MODEL FIT
###############################################################

# function that defines the whole model
def prepareOverallModel(mod_powerlaw, params_powerlaw, mod_flares, params_flares):

    model_components = mod_powerlaw, mod_flares, len(params_flares)

    model_params = []
    
    for item in params_powerlaw:
        model_params.append(item)

    for item2 in params_flares:
        for item in item2:
            model_params.append(item)

    return model_components, model_params

model_components, model_params = prepareOverallModel(bknpower_model, bknpower_param, laffmodels.flare_gaussian, flareFits)


# def finalModel(beta, x):
#     model_params = []
#     for woo in beta:
#         for poo in woo:
#             model_params.append(poo)

#     return print(*model_params)

def finalModel(beta, x):

    possibleModels = laffmodels.powerlaw_1break, laffmodels.powerlaw_2break, laffmodels.powerlaw_3break, laffmodels.powerlaw_4break, laffmodels.powerlaw_5break

    powerlawmodel = model_components[0]
    flaremodel = laffmodels.flare_gaussian
    flarecount = model_components[2]
    powerlawcompcount = len(beta) - (3*flarecount)

    totalModel = powerlawmodel(beta[0:powerlawcompcount], x) + flaremodel(beta[powerlawcompcount:powerlawcompcount+3],x)

    return totalModel
    # print(beta)

# totalmodel = lambda x : the funcrion
# then fit this function

modelmodelmodel = finalModel(model_params, np.array(data.time))

###############################################################
# STATISTICS
###############################################################

# final model adjusted to data
finalModel = bknpower_model(bknpower_param, np.array(data.time))
for flare in flareFits:
    finalModel += laffmodels.flare_gaussian(flare, np.array(data.time))

# final model across a range
constant_range = np.logspace(1.7,6, num=2000)
finalRange = bknpower_model(bknpower_param, constant_range)
for flare in flareFits:
    finalRange += laffmodels.flare_gaussian(flare, constant_range)


# R^2 statistic
ss_res = np.sum((data.flux - finalModel) ** 2)
ss_tot = np.sum((data.flux - np.mean(data.flux)) ** 2)
r2 = 1 - (ss_res/ss_tot)

# chi-square statistic
chi2 = np.sum(((data.flux - finalModel) ** 2)/(data.flux_perr**2))
dof = len(data.time) - len(bknpower_param) - (3 * 3)

# calculate residual (ratio not additive)
ratiores = np.sum(data.flux/finalModel)

###############################################################
# FLUENCE
###############################################################

# fluence - whole model
x = constant_range

func_powerlaw = lambda x: bknpower_model(bknpower_param, x)

func_flare = []

def createFinalFlare(flarefits):
    return lambda x: laffmodels.flare_gaussian(flarefits, x)

for flare in flareFits:
    func_flare.append(createFinalFlare(flare))

def calculateFluence(powerlaw_func, flare_funclist, start, stop):
    comp_powerlaw = integrate.quad(powerlaw_func, start, stop)[0]
    comp_flares = [integrate.quad(flare, start, stop)[0] for flare in flare_funclist]
    tot = comp_powerlaw + np.sum(comp_flares)
    
    return comp_powerlaw, comp_flares, tot

###############################################################
# PRINTING
###############################################################
doPrint = True

N = len(bknpower_param)
def printLine():
    print("==============================================")

if doPrint == True:
    printLine()
    print("LAFF")

    # print powerlaw parameters
    printLine()
    N = len(bknpower_param)
    print("Powerlaw Params")
    print("Indices >", [round(params, 2) for params in bknpower_param[0:int(N/2)]])
    print("Breaks\t>", [round(params, 2) for params in bknpower_param[int(N/2):int(N-1)]])
    print("Norm\t>", [float("{:.2e}".format(bknpower_param[-1]))])

    count = 0
    # print flare times
    printLine()
    print("Flare Times (start, peak, end)")
    for count, (start, peak, decay) in enumerate(flareIndexes, start=1):
        print("Flare",count,">",[round(tableValue(data,start,"time"),2), round(tableValue(data,peak,"time"),2), round(tableValue(data,decay,"time"),2)])

    # print statistics
    printLine()
    print("Statistics")
    print("R^2\t\t\t>",round(r2,2))
    print("Chi-square\t\t>",round(chi2,2),"for",dof, "dof")
    print("Reduced chi-square\t>",round(chi2/dof,3))
    print("Data/model residuals\t>",round(ratiores,2))

    printLine()

    print("Fluence")

    print('---')
    print("Full range")
    val = calculateFluence(func_powerlaw, func_flare, tableValue(data,0,"time"), tableValue(data,-1,"time"))
    print("Powerlaw >", val[0])
    print("Gaussian >", *[(str(x)+"\n\t  ") for x in list(val[1][:-1])],val[1][-1])
    print("Total\t >", val[2])
    
    count = 0
    for count, (start, peak, decay) in enumerate(flareIndexes, start=1):
        print('---')
        print("Flare",count)
        val = calculateFluence(func_powerlaw, func_flare, tableValue(data,start,"time"), tableValue(data,decay,"time"))
        print("Powerlaw >", val[0])
        print("Gaussian >", *[(str(x)+"\n\t  ") for x in list(val[1][:-1])],val[1][-1])
        print("Total\t >", val[2])

    printLine()

###############################################################
# PLOTTING
###############################################################

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios=[2, 1])
axes = gs.subplots(sharex=True)

# plot the lightcurve data
axes[0].errorbar(data.time, data.flux, xerr=[-data.time_nerr, data.time_perr], yerr=[-data.flux_nerr, data.flux_perr], \
    marker='', linestyle='None', capsize=0)
axes[0].errorbar(data.time[flares], data.flux[flares], xerr=[-data.time_nerr[flares], data.time_perr[flares]], yerr=[-data.flux_nerr[flares], data.flux_perr[flares]], \
        marker='', linestyle='None', capsize=0, color='red')

# plot the model
axes[0].plot(constant_range, finalRange)

# plot data/model ratio
axes[1].scatter(data.time, data.flux/finalModel, marker='.')
axes[1].axhline(y=1, color='r', linestyle='--')

# plot the underlying components
# axes[0].plot(constant_range, bknpower_model(bknpower_param, constant_range))
# for flare in flareFits:
#     axes[0].plot(constant_range, laffmodels.flare_gaussian(flare, constant_range), linestyle='--', linewidth=0.5)

# plot vertical lines at powerlaw breaks
for broken in bknpower_param[int(N/2):int(N-1)]:
    axes[0].axvline(broken, color='darkgrey', linestyle='--', linewidth=0.5)

# axes[0].scatter(data.time, modelmodelmodel, marker='X')

# plot properties
axes[0].loglog()
# axes[0].set_ylim(1e-14, 1e-7)
axes[0].set_ylabel("Observed Flux Density (Jy)")
axes[1].set_ylabel("Ratio")
axes[1].set_xlabel("Time since BAT Trigger (s)")

if showPlot == True:
    plt.show()