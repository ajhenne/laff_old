import numpy as np

# simple powerlaw with no breaks
def powerlaw_simple(x, alph, norm):
    return norm*x**(-alph)

# broken powerlaw of N breaks
def powerlaw_general(x,N,a1,a2,a3,a4,a5,b1,b2,b3,b4,norm):
    input_pars = locals()
    alph = []
    brks = []

    # assign alpha and break variables
    for par in input_pars:
        if par.startswith('a') and (input_pars[par] != 0):
            alph.append(input_pars[par])
        if par.startswith('b') and (input_pars[par] != 0):
            brks.append(input_pars[par])
    
    # check there is correct number of alpha and brk parameters
    if len(alph) != N+1 or len(brks) != N:
        print(N, len(alph), len(brks))
        return print("ERROR")

    # calculate the required conditions
    cond = []
    for count in range(len(brks)+1):
        if count == 0:
            cond.append(x<brks[count])
        elif count == len(brks):
            cond.append(x>brks[count-1])
        else:
            cond.append(np.logical_and(brks[count-1]<x,x<brks[count]))

    funclist = [lambda x: norm * (x**(-alph[0])), lambda x: norm * (x**(-alph[1])) * (brks[0]**(-alph[0]+alph[1])), lambda x: norm * (x**(-alph[2])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])), lambda x: norm * (x**(-alph[3])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])) * (brks[2]**(-alph[2]+alph[3])), lambda x: norm * (x**(-alph[4])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])) * (brks[2]**(-alph[2]+alph[3])) * (brks[3]**(-alph[3]+alph[4])), lambda x: norm * (x**(-alph[5])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])) * (brks[2]**(-alph[2]+alph[3])) * (brks[3]**(-alph[3]+alph[4])) * (brks[4]**(-alph[4]+alph[5]))]
    
    func = funclist[0:N+1]

    # if N=0, just func 0.
    # if N=1, func0 + func1
    # for N, func0 --> funcN
    
    return np.piecewise(x, cond, func)

    func1 = norm * (x**(-alph[0]))
    func2 = norm * (x**(-alph[1])) * (brks[0]**(-alph[0]+alph[1]))
    func3 = norm * (x**(-alph[2])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2]))
    func4 = norm * (x**(-alph[3])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])) * (brks[2]**(-alph[2]+alph[3]))
    func5 = norm * (x**(-alph[4])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])) * (brks[2]**(-alph[2]+alph[3])) * (brks[3]**(-alph[3]+alph[4]))
    func6 = norm * (x**(-alph[5])) * (brks[0]**(-alph[0]+alph[1])) * (brks[1]**(-alph[1]+alph[2])) * (brks[2]**(-alph[2]+alph[3])) * (brks[3]**(-alph[3]+alph[4])) * (brks[4]**(-alph[4]+alph[5]))
