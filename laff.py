from cmd import Cmd
import matplotlib.pyplot as plt
from numpy import True_
from pytest import param
import laffmodels
from lmfit import Model, Parameters
from lmfit.confidence import conf_interval
from astropy.table import Table, vstack
from textwrap import dedent
import os
import glob

plt.ion()

########################################################
# RUN LAFF PROMPT
########################################################

class laff(Cmd):

    prompt = 'LAFF > '
    intro="""\
================================================================================
Lightcurve and Flare Fitter (LAFF) -- last updated 13/12/2021
--------------------------------------------------------------------------------
The function of this program is to read Swift XRT lightcurve files and identify
potential flares. These can be ignored so that we can fit a model to the
underlying lightcurve, and then fit a model to the flares themselves.
--------------------------------------------------------------------------------
Type 'help' or '?' to list commands.
--------------------------------------------------------------------------------
*KNOWN BUGS*
- When using graphical commands, allow for the plot to finish updating before
  typing another  command.
- Don't click on/manipulate the plotting window.
================================================================================\
"""
    # INITIALISING VARIABLES
    
    # data (and reduced data variable)
    data         = None
    data_plot    = None
    data_fit     = None
    startIndex   = None
    peakIndex    = None
    decayIndex   = None
    currentModel = None
    
    # toggable plotting parameters
    logScale     = True
    showFlares   = True

    # toggable data parameters
    useFlares          = True
    detectionThreshold = 3

    # alternate command names
    cmd_data  = ('data', 'd', 'dt', 'da')
    cmd_flare = ('flares', 'flare', 'fl', 'f')

    def do_data(self, filepath):
        self.data       = None
        self.data_plot  = None
        self.data_fit   = None
        self.startIndex = None
        self.peakIndex  = None
        self.decayIndex = None

        if filepath:
            try:
                table_one = Table.read('%s' % filepath, format='ascii.qdp', table_id=0)
                table_two = Table.read('%s' % filepath, format='ascii.qdp', table_id=1)
                table_tre = Table.read('%s' % filepath, format='ascii.qdp', table_id=2)
                table_one['tableID'], table_two['tableID'], table_tre['tableID'] = 0, 1, 2

                final_table = vstack([table_one, table_two, table_tre])
                data = final_table.to_pandas()
                data = data.sort_values(by=['col1'])
                data = data.reset_index(drop=True)
                self.data       = data
                self.data_plot  = data
                self.data_fit   = data
                print("Successfully imported %s rows of data." % len(data.index))
            except:
                print(returnError('filepath'))
        else:
            print("Please provide a filepath. Use 'help data' for command usage.")

    def complete_data(self, text, line, begidx, endidx):
        before_arg = line.rfind(" ", 0, begidx)
        if before_arg == -1:
            return # arg not found

        fixed = line[before_arg+1:begidx]  # fixed portion of the arg
        arg = line[before_arg+1:endidx]
        pattern = arg + '*'

        completions = []
        for path in glob.glob(pattern):
            path = _append_slash_if_dir(path)
            completions.append(path.replace(fixed, "", 1))
        return completions

    def help_data(self):
        print(dedent("""\
        data <filename>
            Import your data, providing a valid filename or path to file. Current
            supported filetypes are: Swift XRT .qdp files. Currently the user may only
            load one set of data at a time."""))
        
    def do_flares(self, inp):
        if self.data is None:
            return print(returnError('nodata'))
    
        data = self.data
        sigma = self.detectionThreshold
        
        ### detect possible flare deviations
        possible_flare = []
        for index in data.index[data.col1 < 2000]:
            current       = tableValue(data,index,"flux")
            current_error = tableValue(data,index,"flux_perr")
            ahead_3       = tableValue(data,index+3,"flux")
            ahead_4       = tableValue(data,index+4,"flux")
            if ahead_3 > (current + sigma*current_error) and ahead_4 > (current + sigma*current_error):
                # adjust = minimaFinder(data,index)
                possible_flare.append(index)
        possible_flare = uniqueList(possible_flare)

        ### finding peak of flare
        index_peak = []
        for start in possible_flare:
            values = []
            for n in range(-4,50):
                values.append([tableValue(data,start+n,"flux"),n])
            index_peak.append(start+max(values)[1])
        index_peak = uniqueList(index_peak)

        ### adjust flare start, looking for local minima
        # don't allow for minima to be found after peak
        index_start = []
        for toAdjust in possible_flare:
            local_values = []
            reachedPeak = False
            for position in range(-20,20):
                new_time = tableValue(data,toAdjust+position,"time")
                if (toAdjust+position) >= 0:
                    if (toAdjust+position) not in index_peak and reachedPeak == False and new_time < 2000:
                        local_values.append([tableValue(data,toAdjust+position,"flux"),position])
                    else:
                        reachedPeak = True
            adjusted_index = min(local_values)[1]
            index_start.append(toAdjust + adjusted_index)
        index_start = uniqueList(index_start)

        # filter the list of peaks and minima
        # peak must be greater than minima by X amount

        ### look for when decay ends
        index_decay = []

        # calculate the a2 value, index from peak to N
        def gradientPeak(data,c_index,peak):
            deltaFlux = tableValue(data,c_index,"flux") - tableValue(data,peak,"flux")
            deltaTime = tableValue(data,c_index,"time") - tableValue(data,peak,"time")
            return deltaFlux/deltaTime
        # calculate the a1 value, index from N+1 to N
        def gradientNext(data,c_index):
            deltaFlux = tableValue(data,c_index+1,"flux") - tableValue(data,c_index,"flux")
            deltaTime = tableValue(data,c_index+1,"time") - tableValue(data,c_index,"time")
            return deltaFlux/deltaTime

        for peak in index_peak:
            endDecay = False
            N = peak
            while endDecay == False:
                current_time = tableValue(data,N,"time")
                N += 1
                next_time = tableValue(data,N,"time")

                if current_time >= 2000:
                    endDecay = True
                if (N+1) in index_start:
                    endDecay = True
                if next_time - current_time > 1000:
                    endDecay = True

                gradient_compare = []
                for i in range(10):
                    a1 = gradientPeak(data,N+i,peak)
                    a2 = gradientNext(data,N+i)
                    gradient_compare.append([a1,a2])
                condition = 0
                prev_a1 = gradient_compare[0][0]
                prev_a2 = gradient_compare[0][1]

                for check in gradient_compare:
                    if check[0] < check[1]:
                        if check[0] > prev_a1 and check[1] > prev_a2:
                            condition += 1
                        else:
                            if condition >= 0:
                                condition -= 0.5
                        prev_a1, prev_a2 = check[0], check[1]
                if condition >= 4:
                    N = N + 10
                    endDecay = True
            index_decay.append(N)
        index_decay = uniqueList(index_decay)

        ### output flare times
        self.startIndex, self.peakIndex, self.decayIndex = index_start,index_peak,index_decay

        ### print flare times
        for start, peak, decay in zip(self.startIndex, self.peakIndex, self.decayIndex):
            print("FLARE | start-time:",tableValue(self.data,start,"time"), \
                    "| peak-time:",tableValue(self.data,peak,"time"), \
                    "| end time:", tableValue(self.data,decay,"time") )

    def help_flares(self):
        print(dedent("""\
        flares
            Run an algorithm to search for possible flares in the lightcurve. It will
            identify the start, peak and end times of flares which can then be plotted
            or ignored."""))

    def do_plot(self,inp):
        if self.data is None:
            return print(returnError('nodata'))

        if inp in self.cmd_data:
            if self.logScale == True:
                plt.close()
                plotLightcurve(self.data_plot, 'log')
                plt.draw()
            else:
                plt.close()
                plotLightcurve(self.data_plot, 'normal')
                plt.draw()
        elif inp in self.cmd_flare:
            if self.decayIndex is None:
                return print("Error: please run 'flares' command first.")
            if self.logScale == True:
                plt.close()
                plotLightcurve(self.data_plot, 'log')
                plotFlares(self.data, self.startIndex, self.decayIndex)
                plt.draw()
            else:
                plt.close()
                plotLightcurve(self.data_plot, 'normal')
                plotFlares(self.data, self.startIndex, self.decayIndex)
                plt.draw()
        else:
            print("Error: please provide a valid argument to plot. See 'help plot' for usage.")

    def complete_plot(self, text, line, begidx, endidx):
        plot_commands = ['data', 'flares']
        if not text:
            completions = plot_commands[:]
        else:
            completions = [ f
                            for f in plot_commands
                            if f.startswith(text)
                            ]
        return completions

    def help_plot(self):
        print(dedent("""\
        plot <command>
            data   / d      --- plots the currently loaded lightcurve
            ldata  / ld     --- overrides log command to plot lightcurve on log scale
            flares / fl     --- draws the designated flares regions on the lightcurve
            ----------------------------------------------------------------------------
            Use 'log' command to toggle log scale.
            (Note: avoid drawing multiple plots in rapid succession)"""))

    def do_log(self, inp):
        self.logScale = not self.logScale
        print("Log scale:",self.logScale)

    def help_log(self):
        print(dedent("""\
        log 
            Toggle whether data is plotted with a log scale."""))

    def do_ignore(self,inp):
        if inp in self.cmd_flare or inp == '':

            if self.data is None:
                return print(returnError('nodata'))
            if self.startIndex is None:
                return print(returnError('runflares'))

            self.useFlares = not self.useFlares
            if self.useFlares == False:
                self.data_fit = removeData(self.data, self.startIndex, self.decayIndex)
                print("Flares will be ignored in fitting")
            else:
                self.data_fit = self.data
                print("Flares will be used in fitting")
        
        else:
            print(returnError('invalidarg'))

    def help_ignore(self):
        print(dedent("""\
        ignore <command>
            flares          --- Toggle whether to ignore excluded data such as flares
                                during modelling.
            ----------------------------------------------------------------------------
            Only affects fitting/analysis related functions. To simply not display the
            data during plots, see the 'show' commmand."""))

    def do_display(self,inp):
        if inp in self.cmd_flare:

            if self.data is None:
                return print(returnError('nodata'))
            if self.startIndex is None:
                return print(returnError('runflares'))

            self.showFlares = not self.showFlares
            if self.showFlares == False:
                self.data_plot = removeData(self.data, self.startIndex, self.decayIndex)
                print("Plot flares: False")
            else:
                self.data_plot = self.data
                print("Plot flares: True")    
        else:
            print(returnError('invalidarg'))
    
    def help_display(self):
        print(dedent("""\
        display <command>
            flares          --- Toggle whether to display excluded data such as flares
                                during plotting.
            ----------------------------------------------------------------------------
            Only affects plotting related commands. To exclude the data during modelling
            functions, see the 'ignore' command."""))

    def do_model(self,inp):
        if self.data is None:
            return print(returnError('nodata'))

        self.currentModel = None
        data = self.data_fit
        xdata = data['col1'].values
        ydata = data['col2'].values

        if inp == 'simplepowerlaw':
            model = Model(laffmodels.powerlaw_simple)

            alph = parameterInput(inp,'index')
            norm = parameterInput(inp,'norm')

            params = Parameters()
            params.add('alph', value=alph, min=-5, max=5, vary=True)
            params.add('norm', value=norm, min=1e-32, max=1e+32, vary=True)

            result = model.fit(ydata, params, x=xdata)
            print(result.fit_report())

            plt.plot(xdata, result.best_fit)
            plt.show()

        if inp == 'powerlaw':
            model = Model(laffmodels.powerlaw_general)
            
            # ask user for number of powerlaw breaks
            try:
                N = int(input('LAFF:powerlaw:breaks > '))
            except:
                print(returnError('invalidnumber'))

            params = Parameters()
            params.add('N', value=N, vary=False)

            # ask for index/break parameters - N breaks, N+1 powerlaws
            for i in range(6):
                if i < N:
                    params.add(f'b{i+1}', value=parameterInput(inp, f'break{i+1}'), min=0, max=1000000, vary=True)
                else:
                    params.add(f'b{i+1}', value=0, min=-1, max=1, vary=False)

            for i in range(5):
                if i < N+1:
                    params.add(f'a{i+1}', value=parameterInput(inp, f'index{i+1}'), min=0, max=1000000, vary=True)
                else:
                    params.add(f'a{i+1}', value=0, min=-1, max=1, vary=False)

            # ask user for norm
            norm = parameterInput(inp, 'norm')
            params.add('norm', value=norm, min=1e-13, max=1e+3, vary=True) 
            
            # perform fit and output results
            result = model.fit(ydata, params, x=xdata)
            print(result.fit_report())
            
            # plot fits
            plt.plot(xdata, result.best_fit)
            plt.show()

    def complete_model(self,text,line,begidx,endidx):
        model_commands = ['powerlaw', 'simplepowerlaw']
        if not text:
            completions = model_commands[:]
        else:
            completions = [ f
                            for f in model_commands
                            if f.startswith(text)
                            ]  
                            
        return completions

    def help_model(self):
        modelHelper = modelHelp()
        modelHelper.cmdloop()

    def do_threshold(self,inp):
        print("inp is :",inp)
        try:
            self.detectionThreshold = float(inp)
            print("changing")
        except:
            print(returnError('invalidnumber'))
    
    def help_threshold(self):
                    print(dedent("""\
                    threshold
                        Change the detection threshold in the 'flares' detection algorithm. Input number
                        represents the number of error bars required for subsequent bins to see if there
                        is a flare. See documentation on detection algorithm for further details."""))

# DEFAULT COMMANDS

    def do_shell(self, shellcommand):
        """! <shell command>\n\tRun a shell command, accepts 'shell' or '!'."""
        os.system(shellcommand)

    def do_quit(self, inp):
        """Exit the LAFF application."""
        print("Exiting the LAFF application.")
        return True

    def default(self, inp):
        if inp == 'q' or inp == 'exit':
            return self.do_quit(inp)
        else:
            print("'%s' is not a valid command. Type 'help' for a list of commands." % inp)

    def emptyline(self):
        pass

    do_EOF = do_quit

########################################################
# GENERAL FUNCTIONS
########################################################

# part of filepath autocompletion
def _append_slash_if_dir(p):
    if p and os.path.isdir(p) and p[-1] != os.sep:
        return p + os.sep
    else:
        return p

# remove duplicate values from list
def uniqueList(duplicate_list):
    unique_list = list(set(duplicate_list))
    unique_list.sort()
    return unique_list

# returns the parameter value from the datatable
def tableValue(input_data,index,column):
    if column == "time":
        return input_data.iloc[index].col1
    if column == "flux":
        return input_data.iloc[index].col2
    if column == "flux_perr":
        return input_data.iloc[index].col2_perr
    if column == "flux_nerr":
        return input_data.iloc[index].col2_nerr
    if column == "time_perr":
        return input_data.iloc[index].col1_perr
    if column == "time_nerr":
        return input_data.iloc[index].col1_nerr

# print error statements
def returnError(errortype):
    if errortype == 'nodata':
        return "Error: no valid dataset loaded. Please run data command."
    if errortype == 'filepath':
        return "Error: invalid filepath. Ensure path and file extension are correct."
    if errortype == 'runflares':
        return "Error: run 'flares' command first."
    if errortype == 'invalidnumber':
        return "Error: invalid number."
    if errortype == 'invalidarg':
        return "Error: invalid argument. See help <command> for usage."

# plot the main lightcurve, grouping different viewing modes
def plotLightcurve(data,scale):
    groups = data.groupby('tableID')
    for name, group in groups:
        if name == 0:
            plot_color = 'c'
        if name == 1:
            plot_color = 'b'
        if name == 2:
            plot_color = 'r'
        plt.errorbar(group.col1, group.col2, xerr=[-group.col1_nerr,group.col1_perr], yerr=[-group.col2_nerr,group.col2_perr], \
            marker='', linestyle='None', capsize=0, c=plot_color)
    plt.legend(["WT slew mode","WT mode","PC mode"])
    plt.ylabel("Flux (0.3-10 keV) (erg/cm$^{2}$/s)")
    plt.xlabel("Time since BAT trigger (s)")
    if scale == 'log':
        plt.loglog()
    if scale == 'normal':
        plt.semilogy()

# plot the flare regions, highlighted in red
def plotFlares(data,start_index,decay_index):
    for start, end in zip(start_index, decay_index):
        minval = tableValue(data,start,"time")
        maxval = tableValue(data,end,"time")
        plt.axvspan(minval, maxval, alpha=0.5, color='r')

# filter a set of data
def removeData(data,start_index,decay_index):
        # data_subset = data
        unfilteredData = data
        for start, decay in zip(start_index, decay_index):
            unfilteredData = unfilteredData[~unfilteredData['col1'].between(tableValue(data,start,"time"),tableValue(data,decay,"time"))]
        data_subset = unfilteredData
        return data_subset

def parameterInput(model, input_text):
    try:
        return float(input(f'LAFF:{model}:{input_text} > '))
    except:
        return print(returnError('invalidnumber'))

########################################################
# RUN MODEL HELP PROMPT
########################################################

class modelHelp(Cmd):
    prompt = 'LAFF:ModelHelp > '

    intro="""\
        Type the name of a model for more details.

        List of available models:
        
        powerlaw, simplepowerlaw, zzz, aa\
        """
    def default(self,inp):
        if inp == 'q' or inp == 'exit':
            return self.do_quit(inp)
        elif inp == 'powerlaw':
            print('here')
        elif inp == 'simplepowerlaw':
            print('simplepowerlaw')
        elif inp == 'zzz':
            print('zz')
        elif inp == 'aaa':
            print('aa')
        else:
            print('model not recognised. type help for list or something')
    ### 

    def do_quit(self, inp):
        """Exit the LAFF application."""
        print("Exiting the LAFF application.")
        return True

    def emptyline(self):
        pass

    do_EOF = do_quit

# test
laff().cmdloop()


# change input for function
# args: variable name and disply text
# take input and convert to float, catching errors