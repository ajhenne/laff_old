from pyexpat import model
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
import laffmodels
import numpy as np
from tabulate import tabulate

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

b1, b2, b3, b4, b5 = 189, 213, 1300, 5100, 150000 # grb 210112a
a1, a2, a3, a4, a5, a6 = 1, 1, 1, 1, 1, 1
norm = 1e-7

# define data
modeldata = RealData(data.time[~flares], data.flux[~flares], data.time_perr[~flares], data.flux_perr[~flares])

###############################################################
# 1 BREAKS

model_1break = Model(laffmodels.powerlaw_1break)
odr_1break = ODR(modeldata, model_1break, [b1, a1, a2, norm])
odr_1break.set_job(fit_type=0)
output_1break = odr_1break.run()

if output_1break.info != 1:
    i = 1
    while output_1break.info != 1 and i < 100:
        output_1break = odr_1break.restart()
        i += 1

###############################################################
# 2 BREAKS

model_2break = Model(laffmodels.powerlaw_2break)
odr_2break = ODR(modeldata, model_2break, [b1, b2, a1, a2, a3, norm])
odr_2break.set_job(fit_type=0)
output_2break = odr_2break.run()

if output_2break.info != 1:
    i = 1
    while output_2break.info != 1 and i < 100:
        output_2break = odr_2break.restart()
        i += 1

###############################################################
# 3 BREAKS

model_3break = Model(laffmodels.powerlaw_3break)
odr_3break = ODR(modeldata, model_3break, [b1, b2, b3, a1, a2, a3, a4, norm])
odr_3break.set_job(fit_type=0)
output_3break = odr_3break.run()

if output_3break.info != 1:
    i = 1
    while output_3break.info != 1 and i < 100:
        output_3break = odr_3break.restart()
        i += 1

###############################################################
# 4 BREAKS

model_4break = Model(laffmodels.powerlaw_4break)
odr_4break = ODR(modeldata, model_4break, [b1, b2, b3, b4, a1, a2, a3, a4, a5, norm])
odr_4break.set_job(fit_type=0)
output_4break = odr_4break.run()

if output_4break.info != 1:
    i = 1
    while output_4break.info != 1 and i < 100:
        output_4break = odr_4break.restart()
        i += 1

###############################################################
# 5 BREAKS

model_5break = Model(laffmodels.powerlaw_5break)
odr_5break = ODR(modeldata, model_5break, [a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, norm])
odr_5break.set_job(fit_type=0)
output_5break = odr_5break.run()

if output_5break.info != 1:
    i = 1
    while output_5break.info != 1 and i < 100:
        output_5break = odr_5break.restart()
        i += 1



# a range to plot the model across (so we don't have problems with gaps in the data)
constant_range = np.logspace(1.7, 6, num=2000)

# best fitting model
bestFit_range = laffmodels.powerlaw_5break(output_5break.beta, constant_range)
bestFit_model = laffmodels.powerlaw_5break(output_5break.beta, np.array(data.time))

# residuals
residuals = data.flux - bestFit_model
modelsum = bestFit_model

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
# PRINTING
###############################################################

placeholder_breakno = 5

print("LAFF COMPLETE")
print("=====================================")
print("POWERLAW")
print("Number of breaks: ", placeholder_breakno)
for param in output_5break.beta:
    print(param)
print("=====================================")
# print the number of breaks found for best fit model
# and the fit parameters

# print how many flares found
# and what times the start, peak and decay is
# then print the fit for each flare


 # print all params
for param in output_5break.beta:
    print(param)

print("{:.2f}\t".format(tableValue(data,start,"time")),"{:.2f}\t".format(tableValue(data,peak,"time")),"{:.2f}".format(tableValue(data,decay,"time")))



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

for flare in flare_fits:
    bestFit_range = bestFit_range + laffmodels.flare_gaussian(flare, constant_range)
axes[0].plot(constant_range, bestFit_range)

# ratio if true, residuals if not
if showRatio == True:
    axes[1].scatter(data.time, data.flux/modelsum, marker='.')
    axes[1].axhline(y=1, color='r', linestyle='--')
else:
    axes[1].errorbar(data.time, residuals, xerr=[-data.time_nerr, data.time_perr], yerr=[-data.flux_nerr, data.flux_perr], \
     marker='', linestyle='None', capsize=0)


 # plot the underlying model components before addition
if showComponents == True:
    axes[0].plot(constant_range, laffmodels.powerlaw_5break(output_5break.beta, constant_range))
    for flare in flare_fits:
        axes[0].plot(constant_range, laffmodels.flare_gaussian(flare,constant_range))

axes[0].set_ylim(1e-14, 1e-7)
axes[0].loglog()

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

