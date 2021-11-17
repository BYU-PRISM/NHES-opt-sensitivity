import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
sns.set_theme()
from gekko import GEKKO
import shutil 

model_version = 2.6

# Maximum values for the uncertain parameters
max_params = {
    # System Parameters
    'smr_min_cap': 0.4,
    'tes_eff': 1.0, 
    'turb_eff': 0.4,
    'smr_ramp_up': 1000,
    'smr_ramp_down': -1000,
    'dcost_smr': 0.00001, # $/
    'dcost_tes': 0.00001, # $/tes_in
   
    # Economic Parameters
    'smr_cap_cost': (24303-666*0.35) * 1000, # $, currently a number for a whole nuscale plant
    'smr_fix_cost': (57.37-52*0.35) * 1000, # $/MWth-yr
    'smr_var_cost': 5,                  # $/MWh
    
    'tes_cap_cost': 131.47*1000,        # $/MWhth 
    'tes_fix_cost': 43,                 # $/MW-yr
    'tes_var_cost': 29,                 # $/MWhth

    'turb_cap_cost': 700*1000,          # $/MWe
    'turb_fix_cost': 60000,             # $/MWe-yr
    'turb_var_cost': 2.00,              # $/MWhe
}

# Minimum values for the uncertain parameters
min_params = {
    # System Parameters
    'smr_min_cap': 0,
    'tes_eff': 0.5, 
    'turb_eff': 0.3,
    'smr_ramp_up': 0,
    'smr_ramp_down': -0,
    'dcost_smr': 0.000001, # $/
    'dcost_tes': 0.000001, # $/tes_in
   
    # Economic Parameters
    'smr_cap_cost': (8143-666*0.35) * 1000,  # $, currently a number for a whole nuscale plant
    'smr_fix_cost': (38.25-52*0.35) * 1000,   # $/MWth-yr
    'smr_var_cost': 0,                  # $/MWh
    
    'tes_cap_cost': 31.34*1000,         # $/MWhth 
    'tes_fix_cost': 14,                 # $/MW-yr
    'tes_var_cost': 3,                  # $/MWhth

    'turb_cap_cost': 500*1000,          # $/MWe
    'turb_fix_cost': 40000,             # $/MWe-yr
    'turb_var_cost': 0.00,              # $/MWhe
}

# Nominal model parameters 
nom_params = {
    # System Parameters
    'smr_min_cap': 0.2,
    'tes_eff': 0.7, 
    'turb_eff': 0.35,
    'smr_ramp_up': 100,
    'smr_ramp_down': -100,
    'dcost_smr': 0.000002, # $/
    'dcost_tes': 0.000002, # $/tes_in
   

    # Economic Parameters
    'smr_cap_cost': (11429-666*0.35) * 1000, # $, currently a number for a whole nuscale plant
    'smr_fix_cost': (47.81-52*0.35) * 1000, # $/MWth-yr
    'smr_var_cost': 2.39-1.0*0.35,      # $/MWh
    
    'tes_cap_cost': 71.71 * 1000,       # $/MWhth 
    'tes_fix_cost': 20,                 # $/MW-yr
    'tes_var_cost': 14,                 # $/MWhth

    'turb_cap_cost': 666*1000,          # $/MWe
    'turb_fix_cost': 52000,             # $/MWe-yr
    'turb_var_cost': 1.0,               # $/MWhe
}

def model(params, nhrs=120, nshift=0, data_file='./grid_data/more_data_scaled.csv', 
            plot=False, remote=False, solver=3,
            load=None, wind=None, solar=None, 
            fixed_smr=None, fixed_tes=None, fixed_turb=None,
            solver_options=None, nodes=None, 
            sequential=False, remote_fallback=False, cleanup=True):
    '''
    Run the model of a simple NHES

    fixed_smr: Float. Use given value as fixed SMR capacity size. Defaults to optimizing the capacity
    fixed_tes: Float. Use given value as fixed TES capacity size. Defaults to optimizing the capacity
    sequential: Bool. Performs a sequential optimization scheme. See below.

    Sequential optimization: This scheme fixes the design variables and first optimizes just the dispatch. It then adds the design variables back in and reoptimizes both the dispatch and design. Note that this scheme assumes that neither design variable should be held fixed in the final optimization.

    returns a dict of the optimal operation results including TES capacity, SMR capacity, LCOE and compoenet dispatch

    '''

    if sequential:
        fixed_tes = fixed_tes if fixed_tes else 2000
        fixed_turb = fixed_turb if fixed_turb else 900 
        fixed_smr = fixed_smr if fixed_smr else 400

    # Load the data
    data = pd.read_csv(data_file)
    Wind = data['Wind'].values[nshift:nhrs+nshift] if wind is None else wind[nshift:nhrs+nshift]
    Solar= data['Solar'].values[nshift:nhrs+nshift] if solar is None else solar[nshift:nhrs+nshift]
    Load = data['Load'].values[nshift:nhrs+nshift] if load is None else load[nshift:nhrs+nshift]
   
    # ---------------- Estimated Lifetimes ----------------
    smr_lifetime = 60
    sol_lifetime = 35
    wind_lifetime = 30
    turb_lifetime = 50
    #Need to adjust LCOE based on minimum plant lifetime. This will adjust fixed costs by factor of sys_plant_lifetime/x_plant_lifetime
    sys_lifetime = 30
   
    # ---------------- System Parameters ----------------
    # smr_max_cap = params['smr_max_cap']
    wind_cap = max(Wind)
    sol_cap = max(Solar)
    tes_eff = params['tes_eff'] 
    turb_eff = params['turb_eff'] 
    smr_min_cap = params['smr_min_cap']
    smr_ramp_up = params['smr_ramp_up']
    smr_ramp_down = params['smr_ramp_down']
    dcost_smr = params['dcost_smr'] 
    dcost_tes = params['dcost_tes']
    
    # ---------------- Economic Parameters ----------------
    smr_cap_cost = params['smr_cap_cost']
    smr_fix_cost = params['smr_fix_cost']
    smr_var_cost = params['smr_var_cost']

    turb_cap_cost = params['turb_cap_cost']
    turb_fix_cost = params['turb_fix_cost']
    turb_var_cost = params['turb_var_cost']

    tes_cap_cost = params['tes_cap_cost']
    tes_fix_cost = params['tes_fix_cost']
    tes_var_cost = params['tes_var_cost']

    wind_cap_cost = 1877*1000 # params['wind_cap_cost']
    wind_fix_cost = 39.7*1000 # params['wind_fix_cost']
    wind_var_cost = 0.0001    # params['wind_var_cost']

    sol_cap_cost = 2534*1000  # params['sol_cap_cost']
    sol_fix_cost = 21.8*1000  # params['sol_fix_cost']
    sol_var_cost = 0.0001     # params['sol_var_cost']


    m = GEKKO(remote=remote)
    m.time = np.linspace(0, nhrs-1, nhrs)  # time in hours
    
    # Data from CAISO
    load = m.Param(value=Load, name='load')
    wind = m.Param(value=Wind, name='wind')
    solar = m.Param(value=Solar, name='solar')
    
    # Design Variables
    tes_batt_max = m.FV(value=fixed_tes, lb=2000, ub=20000, name='tes_batt_max') # MWh, Max TES storage
    if not fixed_tes:
        tes_batt_max.STATUS = 1

    smr_max_cap = m.FV(value=fixed_smr, lb=750, ub=1200, name='smr_max_cap')
    #smr_max_cap = m.FV(value=fixed_smr, lb=0, ub=1200)
    if not fixed_smr:
        smr_max_cap.STATUS = 1
        
    turb_max_cap = m.FV(value=fixed_turb, lb=350, ub=600, name='turb_max_cap')
    if not fixed_turb:
        turb_max_cap.STATUS = 1

    
    # A decision variable describing what percent of the generated heat stored.
    # It can go negative to simulate retreiving energy from the TES.
    tes_store = m.Var(lb=-1, ub=1, fixed_initial=False, name='tes_store')
   
    smr_gen = m.Var(value=1200, lb=000, ub=10000, fixed_initial=False, name='smr_gen')
    d_smr_gen = m.Var(fixed_initial=False, name='d_smr_gen')
    m.Equation(d_smr_gen == smr_gen.dt())
    m.Equation(d_smr_gen <= smr_ramp_up)
    m.Equation(d_smr_gen >= smr_ramp_down)
    
    tes_in = m.Var(lb=0, ub=1000, name='tes_in')
    tes_out = m.Var(lb=0, ub=1000, name='tes_out')

    d_TES = m.Var(name='d_TES') # Cannot have derivatives in the objective, so this must be defined separately
    
    m.Equation(d_TES == tes_in.dt())

    turb_in = m.Intermediate(smr_gen - tes_in + tes_out*(tes_eff/(tes_eff + (1 - tes_eff)/2)))
    turb_out = m.Intermediate(turb_eff*turb_in)
    
    tes_batt = m.Var(value=0, lb=0, ub=100000, name='tes_batt')
    
    m.Equation(tes_batt.dt() == tes_in*(tes_eff + (1 - tes_eff)/2) - tes_out)
    m.Equation(tes_batt <= tes_batt_max) #storage must stay below max stoarge used for cost calcs
    
    #m.periodic(tes_batt)
    
    # Make sure the turbine does not run backwards
    #m.Equation(turb_out >= 0)
    
    # Total electricity balance
    net_load = m.Intermediate(load - wind - solar)
    m.Equation(turb_out >= net_load)
    m.Equation(smr_gen <= smr_max_cap)
    m.Equation(smr_gen >= smr_min_cap*smr_max_cap)
    
    m.Equation(turb_max_cap>=turb_out)
    # Not entirely sure why, but this variable and constraint are required to make
    # the problem feasible
    sum_load = m.Var(value = 400*nhrs)
    m.Equation(sum_load == m.integral(load))
    
    avg_MW = m.Intermediate(sum_load/nhrs) #MW/hr load, used to calculate sys lifetime energy produced
    sum_solar = m.Intermediate(m.integral(solar))
    sum_wind = m.Intermediate(m.integral(wind))
    sum_smr = m.Intermediate(m.integral(smr_gen))
    sum_tes = m.Intermediate(m.integral(d_TES))
    sum_turb = m.Intermediate(m.integral(turb_out))
    hrs_per_yr = 8760 # number of hours in a year
    
    final = np.zeros(nhrs)
    final[-1] = 1
    f = m.Param(final)

    cap_costs = m.Intermediate(
        smr_cap_cost * smr_max_cap * sys_lifetime/smr_lifetime + 
        tes_cap_cost * tes_batt_max * sys_lifetime/smr_lifetime + 
        wind_cap_cost * wind_cap * sys_lifetime/wind_lifetime + 
        sol_cap_cost * sol_cap * sys_lifetime/sol_lifetime + 
        turb_cap_cost * turb_max_cap * sys_lifetime/turb_lifetime)

    
    fix_costs = m.Intermediate(
        smr_fix_cost +
        tes_fix_cost + 
        wind_fix_cost + 
        sol_fix_cost + 
        turb_fix_cost)

    
    var_costs = m.Intermediate(
        smr_var_cost * sum_smr + dcost_smr * m.integral(d_smr_gen**2) + 
        tes_var_cost * sum_tes + dcost_tes * (m.integral(tes_in) + m.integral(tes_out)) + 
        wind_var_cost * sum_wind + 
        sol_var_cost * sum_solar +
        turb_var_cost * sum_turb)
    
    LCOE = m.Intermediate(
        f*(cap_costs + fix_costs*sys_lifetime + var_costs / nhrs * hrs_per_yr * sys_lifetime)
         / (avg_MW*hrs_per_yr*sys_lifetime) )
    
    m.Obj(LCOE + .1/nhrs*(turb_out + wind + solar - load))

    m.options.SOLVER = solver 
    m.options.IMODE = 6
    m.options.MAX_ITER = 700
    if nodes:
        m.options.NODES = nodes
    if solver_options:
        m.solver_options = solver_options

    failed_result = {
        'LCOE': None,
        'tes_size': fixed_tes if fixed_tes else None,
        'smr_size': fixed_smr if fixed_smr else None,
        'turb_size': fixed_turb if fixed_turb else None,
        'solve_time': None,
        'cap_cost': None,
        'params': params,
        'solver': solver,
        'remote': remote,
        'smr_gen': None,
        'turb_in': None,
        'turb_out': None,
        'net_load': None,
        # Return the original np arrays rather than the GEKKO ones
        'wind': Wind,
        'solar': Solar,
        'load': Load,
        'tes_batt': None,
        'tes_in': None,
        'tes_out': None,
        'fixed_tes': fixed_tes,
        'fixed_smr': fixed_smr,
        'fixed_turb': fixed_turb,
        'sequential': sequential,
        'success': False,
        'model_version': model_version
    }

    if sequential:
        m.options.SOLVER = solver
        # Try and do the fixed design presolve first
        if plot:
            print('Running presolve with fixed design vars')
        try:
            m.solve(disp=plot)
            if cleanup: m.cleanup()
        except:
            if cleanup: m.cleanup()
            return failed_result
        # reset the fixed design variables, so they get returned properly
        fixed_tes = None
        fixed_smr = None
        fixed_turb = None
        # Add the design variables back in and update some constraints
        m.options.TIME_SHIFT = 0
        tes_batt_max.STATUS = 1
        smr_max_cap.STATUS = 1
        turb_max_cap = 1

        m.options.SOLVER = solver
        if plot:
            print('Completed presolve')
    try:
        m.solve(disp=plot)
        if cleanup: m.cleanup()
    except Exception as e:
        print(e)
        if cleanup: m.cleanup()
        if plot:
            print('Run failed')
        if remote_fallback:
            if plot:
                print('Retrying with remote IPOPT')
            m.options.SOLVER=3
            m._remote = True
            try:
                m.solve(disp=plot)
                if cleanup: m.cleanup()
            except:
                if cleanup: m.cleanup()
                return failed_result
        return failed_result

    if plot:    
        # SMR and TES capacities may be fixed in which case they are not gekko variables
        if fixed_tes is None:
            print(f'Optimized TES Cap.  {tes_batt_max.value[-1]} MWh')
        if fixed_smr is None:
            print(f'Optimized SMR Cap.  {smr_max_cap.value[-1]} MWth')
        if fixed_turb is None:
            print(f'Optimized Turb Cap. {turb_max_cap.value[-1]} MWe')

        print(f'System LCOE {LCOE.value[-1]} $/MW')
    
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=m.time, y=smr_gen.value, name='Nuclear (MWth)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=m.time, y=d_smr_gen.value, name='dSMR/dt'), row=1, col=1)
        fig.add_trace(go.Scatter(x=m.time, y=turb_in.value, name='Turbine in (MWth)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=m.time, y=turb_out.value, name='Turbine out (MWe)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=m.time, y=net_load.value, name='Net Load (MWe)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=m.time, y=tes_batt.value, name='TES Storage (MWhth)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=m.time, y=tes_in.value, name='TES usage (MWth)'), row=2, col=1)
        fig.show()
        
    return {
        'LCOE': LCOE.value[-1],
        'tes_size': fixed_tes if fixed_tes else tes_batt_max.value[-1],
        'smr_size': fixed_smr if fixed_smr else smr_max_cap.value[0],
        'turb_size': fixed_turb if fixed_turb else turb_max_cap.value[0],
        'solve_time': m.options.SOLVETIME,
        'cap_cost': cap_costs.value[-1],
        'params': params,
        'solver': m.options.SOLVER,
        'remote': remote,
        'smr_gen': smr_gen.value,
        'turb_in': turb_in.value,
        'turb_out': turb_out.value,
        'net_load': net_load.value,
        'wind': wind.value,
        'solar': solar.value,
        'load': load.value,
        'tes_batt': tes_batt.value,
        'tes_in': tes_in.value,
        'tes_out': tes_out.value,
        'fixed_tes': fixed_tes,
        'fixed_smr': fixed_smr,
        'fixed_turb': fixed_turb,
        'sequential': sequential,
        'success': True,
        'model_version': model_version
    }

