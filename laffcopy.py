from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.odr import ODR, Model, RealData
import laffmodels
import numpy as np

import warnings
warnings.filterwarnings("ignore")

filepath = 'data/grb210112a.qdp' # data filepath
riseRequirement = 2 # how much data should rise to flag as potential flare
decayRequirement = 4 # variable affecting how easier it is to end decay
showFlares = True # plot the flare data?
showComponents = False # show the individual model components?
showRatio = True # if not, show residuals

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

# for each possible peak, look for a nearby minima as the flare start
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

# assign flares to table
for start, peak, decay in zip(index_start, index_peak, index_decay):
    rise_start = data.index >= start
    decay_end = data.index < decay
    data['flare'][rise_start & decay_end] = True

flares = data.flare == True

###############################################################
# POWERLAW FITTING
###############################################################

# define data
modeldata = RealData(data.time[~flares], data.flux[~flares], data.time_perr[~flares], data.flux_perr[~flares])

# visually estimate the breaks
b1, b2, b3, b4, b5 = 120, 200, 630, 9500, 120000
a1, a2, a3, a4, a5, a6 = 1, 1, 1, 1, 1, 1
norm = 1e-7

# fit through 5 breaks
brk1_fit, brk1_param = modelFit(modeldata, laffmodels.powerlaw_1break, [a1, a2, b1, norm])
brk2_fit, brk2_param = modelFit(modeldata, laffmodels.powerlaw_2break, [a1, a2, a3, b1, b2, norm])
brk3_fit, brk3_param = modelFit(modeldata, laffmodels.powerlaw_3break, [a1, a2, a3, a4, b1, b2, b3, norm])
brk4_fit, brk4_param = modelFit(modeldata, laffmodels.powerlaw_4break, [a1, a2, a3, a4, a5, b1, b2, b3, b4, norm])
brk5_fit, brk5_param = modelFit(modeldata, laffmodels.powerlaw_5break, [a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, norm])

# determine the best fit - WIP
best_model = laffmodels.powerlaw_5break
best_param = brk5_param







# a range to plot the model across (so we don't have problems with gaps in the data)
constant_range = np.logspace(1.7, 6, num=2000)

# best fitting model
powerlaw_param = brk5_param
powerlaw_model = laffmodels.powerlaw_5break

plaw_model_range = powerlaw_model(powerlaw_param, constant_range)
plaw_model_data = powerlaw_model(powerlaw_param, np.array(data.time))

# residuals
residuals = data.flux - plaw_model_data
modelsum = plaw_model_data

###############################################################
# FLARE FITTING
###############################################################

flare_fits = []

# for each flare, automatically fit a gaussian
for start, peak, decay in zip(index_start, index_peak, index_decay):

    flaredata = RealData(data.time[start:decay+1], residuals[start:decay+1], data.time_perr[start:decay+1], data.flux_perr[start:decay+1])

    model_flare = Model(laffmodels.flare_gaussian)
    odr_flare = ODR(flaredata, model_flare, [tableValue(data,peak,"flux"),tableValue(data,peak,"time"),tableValue(data,decay,"time")-tableValue(data,start,"time")])
    odr_flare.set_job(fit_type=0)
    output_flare = odr_flare.run()

    if output_flare.info != 1:
        i = 1
        while output_flare.info != 1 and i < 100:
            output_flare = odr_flare.restart()
            i += 1

    # append the fit information
    flare_fits.append(output_flare.beta)

    # and add to residuals and modelsum
    residuals = residuals - laffmodels.flare_gaussian(output_flare.beta, np.array(data.time))
    modelsum = modelsum + laffmodels.flare_gaussian(output_flare.beta, np.array(data.time))

###############################################################
# STATISTICS
###############################################################

# adding the flare components to base model
finalModel = plaw_model_range
for flare in flare_fits:
    finalModel = finalModel + laffmodels.flare_gaussian(flare, constant_range)

finalFittedModel = plaw_model_data
for flare in flare_fits:
    finalFittedModel = finalFittedModel + laffmodels.flare_gaussian(flare, np.array(data.time))

# statistics
dof = len(data.time) - len(powerlaw_param)

###############################################################
# PRINTING
###############################################################

break_number = 3
doprint = False

if doprint == True:
    print("LAFF COMPLETE")
    print("=====================================")
    print("POWERLAW PARAMETERS")
    print("---")
    print("Number of breaks :", break_number)
    print("---")
    i = 0
    while i <= break_number:
        print("Index",i+1,">","{:.2f}\t".format(powerlaw_param[i]))
        i += 1
    print("---")
    while i <= 2*break_number:
        print("Break",i-break_number,">","{:.2f}\t".format(powerlaw_param[i]))
        i += 1
    print("---")
    print("Norm",">",powerlaw_param[i])
    print("=====================================")
    print("FLARES")
    print("---")
    print("Number of flares :",len(index_peak))
    print("---")
    i = 1
    for start, peak, decay, fit in zip(index_start, index_peak, index_decay, flare_fits):
        print("Flare",i)
        print("Start  >", "{:.2f}\t".format(tableValue(data,start,"time")))
        print("Peak   >", "{:.2f}\t".format(tableValue(data,peak,"time")))
        print("End    >", "{:.2f}\t".format(tableValue(data,decay,"time")))
        print("Fit Parameters")
        print("Height >", fit[0])
        print("Centre >", "{:.2f}\t".format(fit[1]))
        print("Width  >", "{:.2f}\t".format(fit[2]))
        print("---")
        i += 1

###############################################################
# PLOTTING
###############################################################

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios=[2, 1])
axes = gs.subplots(sharex=True)

# plot the main lightcurve
if showFlares == True:
    axes[0].errorbar(data.time, data.flux, xerr=[-data.time_nerr, data.time_perr], yerr=[-data.flux_nerr, data.flux_perr], \
        marker='', linestyle='None', capsize=0)
    axes[0].errorbar(data.time[flares], data.flux[flares], xerr=[-data.time_nerr[flares], data.time_perr[flares]], yerr=[-data.flux_nerr[flares], data.flux_perr[flares]], \
        marker='', linestyle='None', capsize=0, color='red')
else:
    axes[0].errorbar(data.time[~flares], data.flux[~flares], xerr=[-data.time_nerr[~flares], data.time_perr[~flares]], yerr=[-data.flux_nerr[~flares], data.flux_perr[~flares]], \
        marker='', linestyle='None', capsize=0)

# plot the underlying model on the lightcurve
axes[0].plot(constant_range, finalModel)

N = len(powerlaw_param)
# vertical lines on the powerlaw breaks
for broken in powerlaw_param[int(N/2):int(N-1)]:
    axes[0].axvline(broken, color='darkgrey', linestyle='--', linewidth=0.5)

# ratio if true, residuals if not
if showRatio == True:
    axes[1].scatter(data.time, data.flux/modelsum, marker='.')
    axes[1].axhline(y=1, color='r', linestyle='--')
else:
    axes[1].errorbar(data.time, residuals, xerr=[-data.time_nerr, data.time_perr], yerr=[-data.flux_nerr, data.flux_perr], \
     marker='', linestyle='None', capsize=0)

# plot the underlying model components before addition
if showComponents == True:
    axes[0].plot(constant_range, plaw_model_range)
    for flare in flare_fits:
        axes[0].plot(constant_range, laffmodels.flare_gaussian(flare,constant_range))

axes[0].set_ylim(1e-14, 1e-7)
axes[0].loglog()

axes[0].set_ylabel("Observed Flux Density (Jy)")
axes[1].set_ylabel("Ratio")
axes[1].set_xlabel("Time since BAT Trigger (s)")

# ss_res = np.sum((data.flux - finalFittedModel) ** 2)
# ss_tot = np.sum((data.flux - np.mean(data.flux)) ** 2 )
# r2 = 1 - (ss_res/ss_tot)
# print(r2)

plt.show()


# with the removed flare data
# plot a simple powerlaw
# then powerlaw with 1 break
# ...
# up to powerlaw with 5(?) breaks
# evaluate each one
# figure out which is the best fit






# should i include the first and last point within the flares in the fitting?

# be careful after removing flare data, it may cause errors when looking for certain positions
# maybe i should keep a copy of the full dataset always so i can go back to it if neccesary?


# make the lightcurve fitter a function and just run for each model?

# print fit errors

# optimise the break finder
# cycle through all the data and regularly put a break it
# test all these breaks to see which improves the fit the most
# whichever is best, add this as best fit
# keep that perm, and add the next break

# get res var of the fit + flares