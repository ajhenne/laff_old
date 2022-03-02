from lmfit import Model
import laffmodels

gmodel = Model(laffmodels.powerlaw_thisissurelyit)

params = gmodel.make_params(a1=2, a2=3, a3=4)

print(params)