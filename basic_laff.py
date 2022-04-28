from astropy.table import Table, vstack
import matplotlib.pyplot as plt
# only 2 imports

filepath = 'data/grb210112a.qdp' # data filepath
riseRequirement = 2 # how much data should rise to flag as potential flare
decayRequirement = 4 # variable affecting how easier it is to end decay
showFlares = True # plot the flare data?

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
# PLOTTING
###############################################################

# print flare times to console
print("START\t PEAK\t END")  
for start, peak, decay in zip(index_start, index_peak, index_decay):
    print("{:.2f}\t".format(tableValue(data,start,"time")),"{:.2f}\t".format(tableValue(data,peak,"time")),"{:.2f}".format(tableValue(data,decay,"time")))
  

if showFlares == True:
    plt.errorbar(data.time, data.flux, xerr=[-data.time_nerr, data.time_perr], yerr=[-data.flux_nerr, data.flux_perr], \
    marker='', linestyle='None', capsize=0 )

    for start, peak, decay in zip(index_start, index_peak, index_decay):
        rise_start  = data.index >= start
        rise_end    = data.index <= peak
        decay_start = data.index > peak
        decay_end   = data.index <= decay
        plt.axvspan(tableValue(data,start,"time"), tableValue(data,decay,"time"), color='grey', alpha=0.5)
        plt.errorbar(data.time[rise_start & rise_end], data.flux[rise_start & rise_end], xerr=[-data.time_nerr[rise_start & rise_end], data.time_perr[rise_start & rise_end]], yerr=[-data.flux_nerr[rise_start & rise_end], data.flux_perr[rise_start & rise_end]], \
        marker='', linestyle='None', capsize=0, color='green')
        plt.errorbar(data.time[decay_start & decay_end], data.flux[decay_start & decay_end], xerr=[-data.time_nerr[decay_start & decay_end], data.time_perr[decay_start & decay_end]], yerr=[-data.flux_nerr[decay_start & decay_end], data.flux_perr[decay_start & decay_end]], \
        marker='', linestyle='None', capsize=0, color='red')
    
else:
    for start, decay in zip(index_start, index_decay):
        plt.axvspan(tableValue(data,start,"time"), tableValue(data,decay,"time"), color='grey', alpha=0.5)

    # remove flares from the full dataset
    def removeData(data,start_index,decay_index):
        unfilteredData = data
        for start, decay in zip(start_index, decay_index):
            unfilteredData = unfilteredData[~unfilteredData['time'].between(tableValue(data,start+1,"time"),tableValue(data,decay-1,"time"))]
        data_subset = unfilteredData
        return data_subset

    data = removeData(data,index_start,index_decay)

    plt.errorbar(data.time, data.flux, xerr=[-data.time_nerr, data.time_perr], yerr=[-data.flux_nerr, data.flux_perr], \
    marker='', linestyle='None', capsize=0 )









plt.loglog()
plt.show()

# should i include the first and last point within the flares in the fitting?

# be careful after removing flare data, it may cause errors when looking for certain positions
# maybe i should keep a copy of the full dataset always so i can go back to it if neccesary?