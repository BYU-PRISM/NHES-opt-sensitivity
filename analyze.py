from simple_nhes import model, max_params, min_params, nom_params
from multiprocessing import Pool, TimeoutError
import statsmodels.api as sm
from pyDOE import lhs
import pandas as pd
import numpy as np
import tqdm 
import psutil

def inner_time_horizon_run(data):
    # Use the nominal parameters to test the time horizon length
    return model(nom_params, **data)

def test_time_horizon_length():
    jobs = []
    # Later ones will need to be run manually
    ls = [60, 120, 200, 400, 600, 800] #, 1000, 1200, 1400] #, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]

    for l in ls:
        jobs.append({
            'nhrs': l,
            'remote': False,
            'plot': False,
            'solver': 1
        })
        jobs.append({
            'nhrs': l,
            'remote': False,
            'plot': False,
            'solver': 3
        })

    return parallel(inner_time_horizon_run, jobs)

def local_sens(params):
    '''Perform a local sensitivity anaylsis about the nominal parameter values'''
    
    kwargs = {
        'remote': False,
        'solver': 1,
        'plot': False,
        'nhrs': 495#120
    }
    
    dp = params.copy()
    dp['kwargs'] = kwargs
    all_passed = True

    try:
        nom = model(params, **kwargs)
        # Save the nominal values
        dp['smr_size'] = nom['smr_size']
        dp['tes_size'] = nom['tes_size']
        dp['turb_size'] = nom['turb_size']
        dp['LCOE'] = nom['LCOE']
        # Not currently storing the nominal dispatch

    except:
        # Skip the rest if the nominal case fails...
        dp['success'] = False
        return dp
    init_step_percent = 0.05
    
    # print(f'Estimating the local sensitivities using forward differencing:')

    # forward difference for each parameter individually
    # Uses a 5% increase of the nominal value
    for param, value in params.items():
        step_params = params
        step_percent = init_step_percent

        #print(f'\t{param}...', end='')

        # Need to handle the possibility that the new point is infeasible.
        # If the new point is infeasible then step back a little like in a line search
        success = False
        j = 0
        while not success:
            try:
                step_params[param] = value + step_percent*value
                step = model(step_params, **kwargs)
                success = True
            except:
                step_percent = step_percent / 2
                step = None
            # Make sure it doesn't try and go too far...
            j += 1
            if j > 5:
                all_passed = False
                break
        
        # Calculate the normalized sensitivity
        for key, val in nom.items():
            if key not in ['solve_time', 'cap_cost'] and type(val) == float and step and step[key]:
                #print(f'{value=}, {val=}')
                dp[f'd{key}_d{param}'] = value/val*(step[key] - val)/(step_percent*value) 

	    # Return parameter to original state
        step_params[param] = value
    dp['success'] = all_passed
    return dp


def single_param_sweep(param):
    param_vals = np.linspace(min_params[param], max_params[param])

    ps = []
    lcoes = []
    tes = []
    smr = []

    for p in param_vals:
        try:
            params = nom_params
            params[param] = p
            result = model(params, solver=1, remote=False, nhrs=120, plot=False)
            ps.append(p)
            tes.append(result[0])
            lcoes.append(result[1])
            smr.append(result[2])
        except:
            print(f'Failed at {param}: {p}')

    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(go.Scatter(x=ps, y=lcoes, name='LCOE ($/MWh)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ps, y=tes, name='TES Cap (MWhth)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ps, y=smr, name='SMR Cap (MWth)'), row=3, col=1)
    fig.show()

    return ps, lcoes, tes, smr

def scale_sample(unscaled_sample):
    '''Scale an LHS sample using the parameter bounds'''
    s = {}
    for i, p in enumerate(nom_params.keys()):
        scale = max_params[p] - min_params[p]
        s[p] = unscaled_sample[i]*scale + min_params[p]
    return s

def lhs_inner(sample):
    return model(sample, nhrs=495, solver=3, plot=False, remote=False)

def use_lhs_mc(n_samples=100, sens=False):
    '''Use Monte Carlo with Latin Hypercube Sampling to analyze the response'''

    samples_unscaled = lhs(len(nom_params.keys()), n_samples)
    samples_scaled = [scale_sample(s) for s in samples_unscaled]

    if sens:
        results = parallel(local_sens, samples_scaled)
    else:
        results = parallel(lhs_inner, samples_scaled)
    return results

def run_load_sample_inner(load):
    kwargs = {
        'nhrs': 495,
        'remote': False,
        'solver': 3,
        'load': load, 
        'nshift': 4
    }
    r = model(nom_params, **kwargs)
    if not r['success']:
        load += 0.001
        kwargs['load'] = load
        r = model(nom_params, **kwargs)
    r['kwargs'] = kwargs
    return r

def run_wind_sample_inner(wind):
    kwargs = {
        'nhrs': 495,
        'remote': False,
        'solver': 3,
        'wind': wind,
        'nshift': 2
    }
    r = model(nom_params, **kwargs)
    if not r['success']:
        wind += 0.001
        kwargs['wind'] = wind
        r = model(nom_params, **kwargs)
    r['kwargs'] = kwargs
    return r


def run_solar_sample_inner(solar):
    kwargs = {
        'nhrs': 495,
        'remote': False,
        'solver': 3,
        'solar': solar,
        'nshift': 3
    }
    r = model(nom_params, **kwargs)
    if not r['success']:
        solar += 0.001
        kwargs['solar'] = solar
        r = model(nom_params, **kwargs)
    r['kwargs'] = kwargs
    return r

def run_all_sample_inner(data):
    kwargs = {
        'nhrs': 495,
        'remote': False,
        'solver': 3,
        'load': data['load'],
        'wind': data['wind'],
        'solar': data['solar'],
        'nshift': 5
    }
    r = model(nom_params, **kwargs)
    # This helps navigate around numerical challenges that can occur
    if not r['success']: 
        load += 0.001 
        kwargs['load'] = load
        r = model(nom_params, **kwargs)
    r['kwargs'] = kwargs
    return r


def test_sample_variation(sample_type='load', n_samples=1000, smoothed=True):
    smooth_load = 4 
    smooth_wind = 2
    smooth_solar = 3 
    smooth_max = max([smooth_load, smooth_wind, smooth_solar])
    if sample_type not in ['all', 'load', 'wind', 'solar']:
        raise Exception('not a recognized sample type')

    if sample_type == 'load':
        loads = pd.read_csv('grid_data/Load1000.csv').drop(['Unnamed: 0'], axis=1)
        if smoothed:
            loads = loads.apply(lambda c: c.rolling(smooth_load).mean(), axis=0)
        loads = loads.transpose().to_numpy()
        return parallel(run_load_sample_inner, loads)

    if sample_type == 'wind':
        winds = pd.read_csv('grid_data/Wind1000.csv').drop(['Unnamed: 0'], axis=1)
        if smoothed:
            winds = winds.apply(lambda c: c.rolling(smooth_wind).mean(), axis=0)
        winds = winds.transpose().to_numpy()
        return parallel(run_wind_sample_inner, winds)
    
    if sample_type == 'solar':
        solars = pd.read_csv('grid_data/Solar1000.csv').drop(['Unnamed: 0'], axis=1)
        if smoothed:
            solars = solars.apply(lambda c: c.rolling(smooth_solar).mean(), axis=0)
        solars = solars.transpose().to_numpy()
        return parallel(run_solar_sample_inner, solars)

    if sample_type == 'all':
        loads = pd.read_csv('grid_data/Load1000.csv').drop(['Unnamed: 0'], axis=1)
        winds = pd.read_csv('grid_data/Wind1000.csv').drop(['Unnamed: 0'], axis=1)
        solars = pd.read_csv('grid_data/Solar1000.csv').drop(['Unnamed: 0'], axis=1)
        if smoothed:
            loads = loads.apply(lambda c: c.rolling(smooth_load).mean(), axis=0)
            winds = winds.apply(lambda c: c.rolling(smooth_wind).mean(), axis=0)
            solars = solars.apply(lambda c: c.rolling(smooth_solar).mean(), axis=0)
        loads = loads.transpose().to_numpy()
        winds = winds.transpose().to_numpy()
        solars = solars.transpose().to_numpy()
        datas = []
        for _ in range(n_samples):
            datas.append({
                'load': loads[int(np.floor(1000*np.random.random()))],
                'wind': winds[int(np.floor(1000*np.random.random()))],
                'solar': solars[int(np.floor(1000*np.random.random()))]
            })
        return parallel(run_all_sample_inner, datas)

def model_fixed(load):
    tes_cap = 2110.980188 
    smr_cap = 771.336108
    nhrs = 360
    load = load[0:nhrs]
    stuff = model(nom_params, nhrs=nhrs, load=load, remote=False, plot=False, solver=1, fixed_tes=tes_cap, fixed_smr=smr_cap) #, sequential=True)
    #if not stuff['success']:
    #    # For some reason some will not solve with sequential=True
    #    stuff = model(nom_params, nhrs=nhrs, load=load, remote=False, plot=False, fixed_tes=tes_cap, fixed_smr=smr_cap, solver=2, sequential=False)
    return stuff

def model_var(load):
    nhrs = 360
    load = load[0:nhrs]
    stuff = model(nom_params, nhrs=nhrs, load=load, remote=False, plot=False, solver=1, sequential=True)
    if not stuff['success']:
        # For some reason some will not solve with sequential=True
        stuff = model(nom_params, nhrs=nhrs, load=load, remote=True, solver=3, plot=False, sequential=False)
    return stuff

def parallel(func, data, nthreads=None):
    '''Run `func` for each item in `data` and return results as a dataframe
    func should take in a single element of data as it's only argument and
    return a dict with consistent keys.
    '''
    if nthreads is None:
        nthreads = psutil.cpu_count() - 2

    results = {}
    with Pool(processes=nthreads) as pool:

        results = []
        for result in tqdm.tqdm(pool.imap(func, data), total=len(data)):
            results.append(result)
        results = pd.DataFrame(results)
    return results

def full_lenvar_inner(run_data):
    return model(run_data['p'], nhrs=run_data['l'])

def full_lenvar(n_samples=100, **kwargs):
    # Make a list of jobs with varying parameters and time series
    run_datas = []
    lens = [60, 120, 200, 250, 300, 350, 400, 500, 600, 750, 845, 917, 990, 1062, 1134, 1206, 1254, 1345, 1405]
    points_unscaled = lhs(len(nom_params.keys()), n_samples)
    points = [scale_sample(s) for s in points_unscaled]
    for l in lens:
        for p in points:
            run_data = {
                'l': l,
                'p': p
            }
            run_datas.append(run_data)

    # Run the list concurrently
    data = parallel(full_lenvar_inner, run_datas, **kwargs)
    return data


if __name__ == "__main__":
    # results = test_time_horizon_length()

    # stuff1 = model(nom_params, solver=2, remote=False, nhrs=120, fixed_smr=1200, fixed_tes=250, plot=True)
    
    #res1 = single_param_sweep('smr_cap_cost')
    #res2 = single_param_sweep('tes_cap_cost')

    #data = local_sens(nom_params)

    #all_series = pd.read_csv('100loads-all.csv')
    #col_names = all_series.columns[1:] # get the first 12
    #loads = [all_series[col].values for col in col_names]
    #stuff = parallel(model_var, loads)

    #results = use_lhs_mc(n_samples=1000)
    # outs = pd.DataFrame(outs)
    # outs.columns = ['TES Capacity (MWhth)', 'LCOE ($/MWh)', 'SMR Capacity (MWth)', 'Solve Time']
    #ins = pd.DataFrame(ins)
    #fails = pd.DataFrame(fails)

    #data, feasible_series, infeasible_series, all_series = test_sample_variation()

    #load0 = pd.read_csv(f'./Load_samples/load_sample_0.csv')['Load'].values[0:360]
    #load15 = pd.read_csv(f'./Load_samples/load_sample_15.csv')['Load'].values[0:360]

    #stuff0 = model(nom_params, load=load0, nhrs=360, remote=False, plot=True, solver=1)
    #stuff15 = model(nom_params, load=load15, nhrs=360, remote=True, plot=True, solver=1)

    #infdata = pd.read_csv('./Load_samples/load_sample_1.csv').Load.values[0:120]

    # Tests to run
    #   - Hold time series fixed and run local sensitivity analysis to each of the parameters
    #   - Watch LCOE and TES change with expanding time horizon to find the critical length
    #   - Watch LCOE, feasibility when holding TES size fixed over a range of stochastic time horizons using nominal parameter values
    #   - Run first order sensitivity analysis over the full grid for the most influential parameters
    pass
