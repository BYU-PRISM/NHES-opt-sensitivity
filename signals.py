import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import glob as glob

def analyze_signal(df):
    '''Analyze a dataframe of synthetic signals in the form of 100loads.csv.
    Returns a new dataframe showing the stats and smoothness characteristics 
    for the signals'''
    df.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')
    sigs_list = df.transpose().to_numpy()
    sigs = pd.DataFrame([{'signal': s} for s in sigs_list])
    # Calculate the average of the signal
    sigs['avg'] = sigs.apply(lambda r: np.average(r['signal']), axis=1)
    # Calcualte the std dev of the signal
    sigs['stddev'] = sigs.apply(lambda r: np.std(r['signal']), axis=1)
    # Calculate the lag-1 autocorrelation coefficient
    sigs['lag1'] = sigs.apply(lambda r: sm.tsa.acf(r['signal'], nlags=4, fft=False)[1], axis=1)
    # Calculate the lag-2 autocorrelation coefficient
    sigs['lag2'] = sigs.apply(lambda r: sm.tsa.acf(r['signal'], nlags=4, fft=False)[2], axis=1)
    
    return sigs

def smooth_signals(df, n=10, w=1):
    df.drop(['Unnamed: 0'], inplace=True, errors='ignore', axis=1)
    smoothed = df.apply(lambda c: c.rolling(n).mean().dropna(), axis=0)
    return smoothed

def plot_signal(synthetic, real, smoothed=None):
    syn_data = analyze_signal(synthetic)
    real_data = analyze_signal(real)
    if smoothed is not None:
        smoothed_data = analyze_signal(smoothed)

    # Plot the histograms on top of each other
    plt.subplot(3, 2, 1)
    plt.title('Average')
    plt.hist(syn_data.avg, label='synthetic', histtype='step', density=True)
    plt.hist(real_data.avg, label='real', histtype='step', density=True)
    if smoothed is not None:
        plt.hist(smoothed_data.avg, label='smoothed', histtype='step', density=True)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.title('Std. Deviation')
    plt.hist(syn_data.stddev, label='synthetic', histtype='step', density=True)
    plt.hist(real_data.stddev, label='real', histtype='step', density=True)
    if smoothed is not None:
        plt.hist(smoothed_data.stddev, label='smoothed', histtype='step', density=True)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.title('Lag-1')
    plt.hist(syn_data.lag1, label='synthetic', histtype='step', density=True)
    plt.hist(real_data.lag1, label='real', histtype='step', density=True)
    if smoothed is not None:
        plt.hist(smoothed_data.lag1, label='smoothed', histtype='step', density=True)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.title('Lag2')
    plt.hist(syn_data.lag2, label='synthetic', histtype='step', density=True)
    plt.hist(real_data.lag2, label='real', histtype='step', density=True)
    if smoothed is not None:
        plt.hist(smoothed_data.lag2, label='smoothed', histtype='step', density=True)
    plt.legend()


    # Plot a few representative signals on top of each other
    plt.subplot(3, 1, 3)
    plt.plot(synthetic[synthetic.columns[0]], label='synthetic')
    plt.plot(real[real.columns[0]], label='real')
    if smoothed is not None:
        plt.plot(smoothed[smoothed.columns[0]], label='smoothed') 
    plt.legend()

    plt.show()

def gather_signals():
    '''Gather all the individual signal files into a single CSV useable for tsvar analysis'''

    for stype in ['Wind', 'Load', 'Solar']:
    
        files = glob.glob(f'grid_data/{stype}_samples/load_sample_*.csv')
        dfs = pd.DataFrame()

        for i, f in enumerate(files):
            fdata = pd.read_csv(f)
            dfs[f'sample_{i}'] = fdata[stype]

        dfs.to_csv(f'grid_data/{stype}1000.csv')

def get_net_load(r):
    if type(r.load) is not np.ndarray:
        return np.array(r.load) - np.array(r.wind) - np.array(r.solar)
    return r.load - r.wind - r.solar

#def sig_diff(df='tsvar_all_500.pkl')
if __name__ == '__main__':
    df = 'tsvar_all_500.pkl'
    d = pd.read_pickle(df)

    # Calculate the net load
    d['net_load'] = d.apply(get_net_load, axis=1) 

    d['net_min'] = d.apply(lambda r: min(r.net_load), axis=1)
    d['net_max'] = d.apply(lambda r: max(r.net_load), axis=1)
    # Calculate the average of the signal
    d['avg'] = d.apply(lambda r: np.average(r['net_load']), axis=1)
    # Calcualte the std dev of the signal
    d['stddev'] = d.apply(lambda r: np.std(r['net_load']), axis=1)
    # Calculate the lag-1 autocorrelation coefficient
    d['lag1'] = d.apply(lambda r: sm.tsa.acf(r['net_load'], nlags=4, fft=False)[1], axis=1)
    # Calculate the lag-2 autocorrelation coefficient
    d['lag2'] = d.apply(lambda r: sm.tsa.acf(r['net_load'], nlags=4, fft=False)[2], axis=1)

    # Plot the net loads



    # load_data = pd.read_csv('grid_data/Load1000.csv')
    # wind_data = pd.read_csv('grid_data/Wind1000.csv')
    # solar_data = pd.read_csv('grid_data/Solar1000.csv')

    # load_real_data = pd.read_csv('grid_data/signal_analysis_load.csv')
    # wind_real_data = pd.read_csv('grid_data/signal_analysis_wind.csv')
    # solar_real_data = pd.read_csv('grid_data/signal_analysis_solar.csv')
