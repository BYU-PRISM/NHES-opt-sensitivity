# Tools for plotting results from the analysis of the Simple NHES
from plotly.subplots import make_subplots
from scipy.stats.kde import gaussian_kde
import plotly.graph_objects as go
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Map the programming names to the math/presentation names
label_map = {
    'smr_size': r'$N_{SMR}$', 
    'tes_size': r'$N_{TES}$',
    'turb_size': r'$N_{turb}',
    'LCOE': 'LCOE',
    'smr_min_cap': r'$N_{SMR, min}$', 
    'tes_eff': r'$\eta_{TES}$', 
    'turb_eff': r'$\eta_{turb}$', 
    'smr_ramp_up': r'$r_{up}$', 
    'smr_ramp_down': r'$r_{down}$', 
    'dcost_smr': r'$D_{cost, SMR}$',
    'dcost_tes': r'$D_{cost, TES}$',
    'smr_cap_cost': r'$C_{cap, SMR}$',
    'smr_fix_cost': r'$C_{fix, SMR}$',
    'smr_var_cost': r'$C_{var, SMR}$',
    'tes_cap_cost': r'$C_{cap, TES}$',
    'tes_fix_cost': r'$C_{fix, TES}$',
    'tes_var_cost': r'$C_{var, TES}$',
    'turb_cap_cost': r'$C_{cap, turb}$',
    'turb_fix_cost': r'$C_{fix, turb}$',
    'turb_var_cost': r'$C_{var, turb}$',
}


def plot_result(r):
    '''Plot the result of a single run'''
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(y=r['smr_gen'], name='Nuclear (MWth)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=r['turb_in'], name='Turbine in (MWth)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=r['turb_out'], name='Turbine out (MWe)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=r['net_load'], name='Net Load (MWe)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=r['tes_batt'], name='TES Storage (MWhth)'), row=2, col=1)
    fig.add_trace(go.Scatter(y=r['tes_in'], name='TES usage (MWth)'), row=2, col=1)
    fig.add_trace(go.Scatter(y=r['tes_out'], name='TES out (MWth)'), row=2, col=1)
    fig.show()

def plot_sensitivities(df, limited=0, show_all=True, **kwargs):
    '''Plot the sensitivities resulting from an LHS/MC/FD run
    limited is an optional arg that limits the sensitivities plotted
        limited=1 -> Only sensitivities that are not essentially all zero including outliers
        limited=2 -> Only a specific set of sensitivities expected to actually have significance
    Any kwargs are passed to the boxplot method
    '''
    io_cols = ['smr_min_cap', 'tes_eff', 'turb_eff', 'smr_ramp_up', 'smr_ramp_down',
        'dcost_smr', 'dcost_tes', 'smr_cap_cost', 'smr_fix_cost',
        'smr_var_cost', 'tes_cap_cost', 'tes_fix_cost', 'tes_var_cost',
        'turb_cap_cost', 'turb_fix_cost', 'turb_var_cost',
        'kwargs', 'smr_size', 'tes_size', 'turb_size', 'LCOE']
    dmodel_cols = [c for c in df.columns if c.startswith('dmodel_version')]
    sens = df.drop(['Unnamed: 0', 'success', *io_cols, *dmodel_cols], axis=1, errors='ignore')
    fig = plt.Figure(figsize=(10, 20))

    if limited == 1:
        # Plots only the sensitivities that are not essentially zero
        # Defined as sensitivities where both the maximum and minimum values are within cutoff of 0
        cutoff = 0.01
        zeros = []
        for s in sens:
            s_stats = sens[s].describe()
            if s_stats[3] > -cutoff and s_stats[7] < cutoff:
                zeros.append(s)
        sens.drop(zeros, axis=1, inplace=True)
        ax = sens.boxplot(**kwargs)
        plt.setp(ax.get_xticklabels(), ha='right', rotation=45)
        plt.show()
        return

    if limited == 2:
        sigs = ['dtes_size_dtes_eff', 'dLCOE_dturb_eff', 'dtes_size_dturb_eff', 'dsmr_size_dturb_eff', 'dLCOE_dsmr_cap_cost', 'dLCOE_dsmr_var_cost']
        sig_labels = [r'$\frac{d\ N_{TES}}{d\ \eta_{TES}}$', 
                r'$\frac{d\ LCOE}{d\ \eta_{turb}}$',
                r'$\frac{d\ N_{TES}}{d\ \eta_{turb}}$',
                r'$\frac{d\ N_{SMR}}{d\ \eta_{turb}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, SMR}}$',
                r'$\frac{d\ LCOE}{d\ C_{var, SMR}}$']
        s = sens[sigs]
        ax = s.boxplot(**kwargs)
        ax.set_xticklabels(sig_labels)
        #plt.setp(ax.get_xticklabels(), ha='right', rotation=0)
        plt.show()
        return

    if limited == 3:
        sigs1 = ['dtes_size_dtes_eff', 'dLCOE_dturb_eff', 'dtes_size_dturb_eff', 'dsmr_size_dturb_eff', 'dLCOE_dsmr_cap_cost']
        sigs2 = ['dLCOE_dsmr_var_cost', 'dLCOE_dturb_var_cost', 'dLCOE_dtes_eff', 'dsmr_size_dtes_eff', 'dLCOE_dtes_cap_cost', 'dLCOE_dturb_cap_cost']
        sig_labels1 = [r'$\frac{d\ N_{TES}}{d\ \eta_{TES}}$', 
                r'$\frac{d\ LCOE}{d\ \eta_{turb}}$',
                r'$\frac{d\ N_{TES}}{d\ \eta_{turb}}$',
                r'$\frac{d\ N_{SMR}}{d\ \eta_{turb}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, SMR}}$']
        sig_labels2 = [r'$\frac{d\ LCOE}{d\ C_{var, SMR}}$', 
                r'$\frac{d\ LCOE}{d\ C_{var, turb}}$',
                r'$\frac{d\ LCOE}{d\ \eta_{TES}}$',
                r'$\frac{d\ N_{SMR}}{d\ \eta_{TES}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, TES}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, turb}}$']

        s1 = sens[sigs1]
        s2 = sens[sigs2]

        nplots = 3 if show_all else 2
        plt.subplot(1, nplots, 1)
        ax1 = s1.boxplot(**kwargs)
        ax1.set_xticklabels(sig_labels1)
        plt.setp(ax1.get_xticklabels(), ha='right', rotation=45)
        plt.ylabel('Normalized Sensitivity')

        plt.subplot(1, nplots, 2)
        ax2 = s2.boxplot(**kwargs)
        ax2.set_xticklabels(sig_labels2)
        plt.setp(ax2.get_xticklabels(), ha='right', rotation=45)

        if show_all:
            sigs3 = [c for c in list(sens.columns) if c not in sigs1 + sigs2]
            s3 = sens[sigs3]
            plt.subplot(1, nplots, 3)
            ax3 = s3.boxplot(**kwargs)
            plt.setp(ax3.get_xticklabels(), ha='right', rotation=45)
        
        plt.show()
        return

    if limited == 4:
        sigs = ['dtes_size_dtes_eff', 'dtes_size_dturb_eff', 'dsmr_size_dturb_eff', 'dLCOE_dturb_eff', 'dLCOE_dsmr_cap_cost', 'dsmr_size_dtes_eff', 'dLCOE_dsmr_var_cost', 'dLCOE_dtes_eff', 'dLCOE_dtes_cap_cost', 'dLCOE_dturb_cap_cost', 'dLCOE_dturb_var_cost']
        sig_labels = [r'$\frac{d\ N_{TES}}{d\ \eta_{TES}}$', 
                r'$\frac{d\ N_{TES}}{d\ \eta_{turb}}$',
                r'$\frac{d\ N_{SMR}}{d\ \eta_{turb}}$',
                r'$\frac{d\ LCOE}{d\ \eta_{turb}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, SMR}}$',
                r'$\frac{d\ N_{SMR}}{d\ \eta_{TES}}$',
                r'$\frac{d\ LCOE}{d\ C_{var, SMR}}$', 
                r'$\frac{d\ LCOE}{d\ \eta_{TES}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, TES}}$',
                r'$\frac{d\ LCOE}{d\ C_{cap, turb}}$',
                r'$\frac{d\ LCOE}{d\ C_{var, turb}}$']

        s = sens[sigs]

        for sig in sigs:
            max_dev = max(abs(sens[sig]), )
            print(f'{sig}: {max_dev}')

        ax = s.boxplot(**kwargs)
        ax.set_xticklabels(sig_labels)
        plt.setp(ax.get_xticklabels(), ha='right', rotation=45)
        plt.ylabel('Normalized Sensitivity')

        plt.show()
        return


    s1_cols = ['dtes_size_dtes_eff', 'dtes_size_dturb_eff', 'dsmr_size_dturb_eff', 'dLCOE_dturb_var_cost']
    s2_cols = ['dtes_size_dsmr_min_cap', 'dtes_size_dsmr_ramp_up', 'dtes_size_ddcost_smr', 'dtes_size_ddcost_tes', 'dtes_size_dsmr_var_cost', 'dtes_size_dtes_var_cost']
    s3_cols = ['dLCOE_dturb_eff', 'dLCOE_dsmr_cap_cost', 'dLCOE_dsmr_var_cost']
    s1 = df[s1_cols]
    s2 = df[s2_cols]
    s3 = df[s3_cols]
    sens.drop([*s1_cols, *s2_cols, *s3_cols], axis=1, inplace=True)
    plt.subplot(2, 2, 1)
    ax1 = s1.boxplot(**kwargs)
    plt.setp(ax1.get_xticklabels(), ha='right', rotation=10)
    plt.subplot(2, 2, 2)
    ax3 = s3.boxplot(**kwargs)
    plt.setp(ax3.get_xticklabels(), ha='right', rotation=10)
    plt.subplot(2, 1, 2)
    ax4 = sens.boxplot(**kwargs)
    plt.setp(ax4.get_xticklabels(), ha='right', rotation=45)
    plt.subplots_adjust(top=0.99, right=0.99, left=0.03, bottom=0.2)
    plt.show()

def plot_correlations(df):
    '''Plot the correlations between inputs and outputs in an LHS/MC run'''
    io_cols = ['smr_min_cap', 'tes_eff', 'turb_eff', 'smr_ramp_up', 'smr_ramp_down',
        'dcost_smr', 'dcost_tes', 'smr_cap_cost', 'smr_fix_cost',
        'smr_var_cost', 'tes_cap_cost', 'tes_fix_cost', 'tes_var_cost',
        'turb_cap_cost', 'turb_fix_cost', 'turb_var_cost', #'kwargs', 
        'smr_size', 'tes_size', 'turb_size', 'LCOE']
    if 'params' in df.columns:
        # This indicates that the params need to unpacked into the top level
        for col in io_cols:
            if col in ['kwargs', 'smr_size', 'tes_size', 'turb_size', 'LCOE']:
                continue
            # FIXME: This line will fail if the dataframe is not read from a file
            df[col] = df.apply(lambda r: eval(r.params)[col], axis=1)

    io_data = df[io_cols]
    #cdata = io_data.corr().drop(['smr_size', 'tes_size', 'turb_size', 'LCOE'], axis=1).tail(n=4)
    cdata = io_data.corr().drop(['turb_size'], axis=1).tail(n=4)
    cdata.drop(['turb_size'], inplace=True, axis=0)

    ylabels = [label_map[l] for l in cdata.index]
    #ylabels=[r'$N_{SMR}$', r'$N_{TES}$', 'LCOE']
    #xlabels=[r'$N_{SMR, min}$', r'$\eta_{TES}$', r'$\eta_{turb}$', r'$r_{up}$', r'$r_{down}$', 
    xlabels = [label_map[l] for l in cdata.columns]
 
    heatmap = sns.heatmap(cdata.round(2), xticklabels=xlabels, yticklabels=ylabels, cmap='viridis', annot=True, annot_kws={"size": 8})
    plt.setp(heatmap.get_yticklabels(), ha='right', rotation=0)
    plt.setp(heatmap.get_xticklabels(), ha='right', rotation=30)
    plt.subplots_adjust(top=0.95, right=1, left=0.07, bottom=0.229)
    plt.show()

def plot_distributions(df):
    '''Plot the output distributions for a LHS/MC run'''
    plt.subplot(1, 3, 1)
    df.LCOE.hist(density=True, bins=20)
    plt.xlabel('LCOE')
    plt.subplot(1, 3, 2)
    df.tes_size.hist(density=True, bins=20)
    plt.xlabel(r'$N_{TES}$ (MWhₜₕ)')
    ax2 = plt.gca()
    xticks = np.arange(5000, 15000, 8000)
    ax2.set_xticks(xticks)
    plt.subplot(1, 3, 3)
    # df.turb_size.hist(density=True, bins=30)
    # plt.xlabel(r'$N_{turb}$ (MWₑ)')
    # plt.subplot(1, 4, 4)
    df.smr_size.hist(density=True, bins=20)
    plt.xlabel(r'$N_{SMR}$ (MWₜₕ)')
    plt.show()

def plot_tsvar(d_load, d_wind, d_solar):
    '''Plot the output distributions resulting from an analysis of time series variance''' 
    plt.subplot(2, 2, 1)
    plt.hist(d_load.LCOE, histtype='step', label='Load')
    plt.hist(d_wind.LCOE, histtype='step', label='Wind')
    plt.hist(d_solar.LCOE, histtype='step', label='Solar')
    plt.legend()
    plt.xlabel('LCOE ($/MWh)')

    plt.subplot(2, 2, 2)
    plt.hist(d_load.tes_size, histtype='step', label='Load')
    plt.hist(d_wind.tes_size, histtype='step', label='Wind')
    plt.hist(d_solar.tes_size, histtype='step', label='Solar')
    plt.legend()
    plt.xlabel('TES Size (MWhth)')

    plt.subplot(2, 2, 3)
    plt.hist(d_load.smr_size, histtype='step', label='Load')
    plt.hist(d_wind.smr_size, histtype='step', label='Wind')
    plt.hist(d_solar.smr_size, histtype='step', label='Solar')
    plt.legend()
    plt.xlabel('SMR Size (MWth)')

    plt.subplot(2, 2, 4)
    plt.hist(d_load.turb_size, histtype='step', label='Load')
    plt.hist(d_wind.turb_size, histtype='step', label='Wind')
    plt.hist(d_solar.turb_size, histtype='step', label='Solar')
    plt.legend()
    plt.xlabel('Turb Size (MWe)')

    plt.show()

def plot_tsvar_final():
    dload = pd.read_pickle('tsvar_load495_1000.pkl')
    dwind = pd.read_pickle('tsvar_wind495_1000.pkl')
    dsolar = pd.read_pickle('tsvar_solar495_1000.pkl')

    # Generate the KDEs
    kde_load_lcoe = gaussian_kde(dload['LCOE'].dropna())
    kde_load_tes = gaussian_kde(dload['tes_size'].dropna())
    kde_load_smr = gaussian_kde(dload['smr_size'].dropna())
    kde_load_turb = gaussian_kde(dload['turb_size'].dropna())

    kde_wind_lcoe = gaussian_kde(dwind['LCOE'].dropna())
    kde_wind_tes = gaussian_kde(dwind['tes_size'].dropna())
    kde_wind_smr = gaussian_kde(dwind['smr_size'].dropna())
    kde_wind_turb = gaussian_kde(dwind['turb_size'].dropna())

    kde_solar_lcoe = gaussian_kde(dsolar['LCOE'].dropna())
    kde_solar_tes = gaussian_kde(dsolar['tes_size'].dropna())
    kde_solar_smr = gaussian_kde(dsolar['smr_size'].dropna())
    kde_solar_turb = gaussian_kde(dsolar['turb_size'].dropna())

    # Generate the xranges
    min_lcoe = min(dload['LCOE'].min(), dwind['LCOE'].min(), dsolar['LCOE'].min())
    max_lcoe = max(dload['LCOE'].max(), dwind['LCOE'].max(), dsolar['LCOE'].max())
    min_tes = min(dload['tes_size'].min(), dwind['tes_size'].min(), dsolar['tes_size'].min())
    max_tes = max(dload['tes_size'].max(), dwind['tes_size'].max(), dsolar['tes_size'].max())
    min_smr = min(dload['smr_size'].min(), dwind['smr_size'].min(), dsolar['smr_size'].min())
    max_smr = max(dload['smr_size'].max(), dwind['smr_size'].max(), dsolar['smr_size'].max())
    min_turb = min(dload['turb_size'].min(), dwind['turb_size'].min(), dsolar['turb_size'].min())
    max_turb = max(dload['turb_size'].max(), dwind['turb_size'].max(), dsolar['turb_size'].max())

    xlcoe = np.linspace(min_lcoe-0.5, max_lcoe, 100)
    xtes = np.linspace(min_tes-700, max_tes, 100)
    xsmr = np.linspace(min_smr, max_smr, 100)
    xturb = np.linspace(min_turb-5, max_turb, 100)

    plt.subplot(2, 2, 1)
    plt.plot(xlcoe, kde_load_lcoe(xlcoe), label='Load')
    plt.plot(xlcoe, kde_wind_lcoe(xlcoe), label='Wind')
    plt.plot(xlcoe, kde_solar_lcoe(xlcoe), label='Solar')
    plt.legend()
    plt.xlabel('LCOE ($/MWh)')

    # plt.subplot(2, 2, 2)
    # plt.hist(dload.tes_size, histtype='step', label='Load')
    # plt.hist(dwind.tes_size, histtype='step', label='Wind')
    # plt.hist(dsolar.tes_size, histtype='step', label='Solar')
    # plt.legend()
    # plt.xlabel('TES Size (MWhth)')

    plt.subplot(2, 2, 2)
    plt.plot(xtes, kde_load_tes(xtes), label='Load')
    plt.plot(xtes, kde_wind_tes(xtes), label='Wind')
    plt.plot(xtes, kde_solar_tes(xtes), label='Solar')
    plt.legend()
    plt.xlabel('TES Size (MWhth)')

    plt.subplot(2, 2, 3)
    plt.plot(xsmr, kde_load_smr(xsmr), label='Load')
    plt.plot(xsmr, kde_wind_smr(xsmr), label='Wind')
    plt.plot(xsmr, kde_solar_smr(xsmr), label='Solar')
    plt.legend()
    plt.xlabel('SMR Size (MWth)')

    plt.subplot(2, 2, 4)
    plt.plot(xturb, kde_load_turb(xturb), label='Load')
    plt.plot(xturb, kde_wind_turb(xturb), label='Wind')
    plt.plot(xturb, kde_solar_turb(xturb), label='Solar')
    plt.legend()
    plt.xlabel('Turb Size (MWe)')

    plt.show()

def plot_tsvar_all_final(d_file='tsvar_all5000.pkl' ):
    d = pd.read_pickle(d_file)

    kde_lcoe= gaussian_kde(d.LCOE.dropna())
    kde_tes = gaussian_kde(d.tes_size.dropna())
    kde_smr = gaussian_kde(d.smr_size.dropna())
    kde_turb = gaussian_kde(d.turb_size.dropna())

    xlcoe = np.linspace(d.LCOE.min(), d.LCOE.max(), 100)
    xtes = np.linspace(d.tes_size.min(), d.tes_size.max(), 100)
    xsmr = np.linspace(d.smr_size.min(), d.smr_size.max(), 100)
    xturb = np.linspace(d.turb_size.min(), d.turb_size.max(), 100)

    plt.subplot(1, 4, 1)
    plt.plot(xlcoe, kde_lcoe(xlcoe))
    plt.xlabel('LCOE (USD/MWₕₑ)')
    plt.ylabel('Probability Density Function')

    plt.subplot(1, 4, 2)
    plt.plot(xtes, kde_tes(xtes))
    plt.xlabel('TES Size (MWₕₜₕ)')
    plt.xticks([5000, 12000])
    
    plt.subplot(1, 4, 3)
    plt.plot(xsmr, kde_smr(xsmr))
    plt.xlabel('SMR Size (MWₜₕ)')

    plt.subplot(1, 4, 4)
    plt.plot(xturb, kde_turb(xturb))
    plt.xlabel('Turb Size (MWₑ)')
    plt.show()




def final_plot_lhs_dist():
    '''Plot the output distributions for a LHS/MC run for final presentation'''

    df_120hrs = pd.read_csv('lhs20000_v26.csv')
    df_360hrs = pd.read_csv('lhs10000_360hrs.csv')
    
    p0 = plt.subplot(1, 3, 1)
    plt.ylabel('Approx. Probability Density Function')
    df_120hrs.LCOE.hist(histtype='step', density=True, bins='scott')
    df_360hrs.LCOE.hist(histtype='step', density=True, bins='scott')
    plt.xlabel('LCOE ($/MWh)')

    p1 = plt.subplot(1, 3, 2)
    df_120hrs.tes_size.hist(histtype='step', density=True, bins='scott')
    df_360hrs.tes_size.hist(histtype='step', density=True, bins='scott')
    plt.xlabel('TES Size (MWhth)')

    # Excluded because it doesn't really vary
    #p2 = plt.subplot(1, 4, 3)
    #df_120hrs.turb_size.hist(histtype='step', density=True)
    #df_360hrs.turb_size.hist(histtype='step', density=True)
    #plt.xlabel('Turbine size (MWe)')

    p3 = plt.subplot(1, 3, 3)
    df_120hrs.smr_size.hist(histtype='step', density=True, bins='scott', label='120 hr dispatch')
    df_360hrs.smr_size.hist(histtype='step', density=True, bins='scott', label='360 hr dispatch')
    plt.xlabel('SMR size (MWth)')
    
    plt.legend()
    plt.show()

def plot_lenvar(data_file='lenvar_full_200.csv'):
    nhrs = [60, 120, 200, 250, 300, 350, 400, 600, 750, 845, 917, 990, 1062, 1134, 1206, 1254, 1345, 1393, 1405, 1410]
    tes = [1645.44284, 1645.44284, 1810.29844, 6819.18, 13152, 13683, 13683.948595, 13683.948595, 13683.948596, 13683.948596, 13683.948596, 16968.249831, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931]
    smr = [873.302466, 873.302466, 873.302466, 873.302, 873.302, 873.302, 873.302466, 873.302466, 873.30246598, 873.30246598, 873.30246598, 873.30246597, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598]
    turb = [393.708709, 401.059414, 401.059414, 433.551, 433.551, 433.551, 433.551536, 437.167193, 518.03708793, 518.03708793, 518.03708793, 544.17739485, 544.17739485, 544.17739485, 544.17739485, 544.17739485, 544.17739485, 547.05645945, 547.05645945, 547.05645945]
    lcoe = [64.496811, 63.015237, 63.517664, 64.7385, 66.669, 67.446, 67.761032, 67.85356, 66.967171092, 66.441484409, 66.138844198, 66.499294987, 66.782094387, 66.589946927, 66.533659873, 66.50513466, 66.403750474, 66.326018583, 66.273331911, 66.143181481]

    d = pd.read_csv(data_file)

    vlcoe = []
    vtes = []
    vsmr = []
    vturb = []
    vn = []
    ns = [60, 120, 200, 250, 300, 350, 400, 500, 600, 750, 845, 917, 990, 1062, 1134, 1206, 1254, 1345, 1393, 1405]

    for n in ns:
        dn = d[d.length == n]
        print(len(dn))
        dl = [l for l in dn.LCOE if l > 0]
        dt = [t for t in dn.tes_size if t > 0]
        du = [t for t in dn.turb_size if t > 0]
        ds = [s for s in dn.smr_size if s > 0]

        if len(dl) > 0:
            vlcoe.append(dl)
            vturb.append(du)
            vtes.append(dt)
            vsmr.append(ds)
            vn.append(n)

    print({n: len(l) for n, l in zip(vn, vlcoe)})

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    ax1.plot(nhrs, tes)
    ax1.violinplot(vtes, positions=vn, widths=50)
    ax1.set_ylabel(r'$N_{TES}$ (MWhth)')
    
    ax2.plot(nhrs, smr)
    ax2.violinplot(vsmr, positions=vn, widths=50)
    ax2.set_ylabel(r'$N_{SMR}$ (MWth)')
    
    ax3.plot(nhrs, turb)
    #ax3.violinplot(vturb, positions=vn, widths=50)
    ax3.set_ylabel(r'$N_{turb}$ (MWe)')
    
    ax4.plot(nhrs, lcoe)
    ax4.violinplot(vlcoe, positions=vn, widths=50)
    ax4.set_xlabel('Dispatch Length (hrs)')
    ax4.set_ylabel('LCOE (USD)')

    plt.show()


def plot_bins():
    ''' A quick test of the different methods of auto-binning for histograms'''
    d = pd.read_csv('lenvar_full_100.csv')
    d60 = d[d.length == 60] # Note: These methods seemed to get overwhelmed by too much data, so I had to trim it...
    thing = d60.LCOE

    supported_methods = ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']

    for i, method in enumerate(supported_methods):
        plt.subplot(3, 3, i+1)
        plt.hist(thing, bins=method)
        plt.title(method)

    plt.show()

def plot_distribution_final():
    d500 = pd.read_csv('lhs20000_long.csv')
    d360 = pd.read_csv('lhs20000_v26.csv')

    d500_lcoes = d500['LCOE'].dropna()
    kde_500_lcoe = gaussian_kde(d500_lcoes)

    d500_tes = d500['tes_size'].dropna()
    kde_500_tes = gaussian_kde(d500_tes)

    d500_smr = d500['smr_size'].dropna()
    kde_500_smr = gaussian_kde(d500_smr)

    d360_lcoes = d360['LCOE'].dropna()
    kde_360_lcoe = gaussian_kde(d360_lcoes)

    d360_tes = d360['tes_size'].dropna()
    kde_360_tes = gaussian_kde(d360_tes)

    d360_smr = d360['smr_size'].dropna()
    kde_360_smr = gaussian_kde(d360_smr)
   
    # Find the ranges to plot over
    lcoe_range = np.linspace(min(min(d360_lcoes), min(d500_lcoes)), max(max(d360_lcoes), max(d500_lcoes)), 100)
    tes_range = np.linspace(min(min(d360_tes), min(d500_tes)), max(max(d360_tes), max(d500_tes)), 100)
    smr_range = np.linspace(min(min(d360_smr), min(d500_smr)), max(max(d360_smr), max(d500_smr)), 100)
    c1, c2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    plt.subplot(1, 3, 1)
    plt.plot(lcoe_range, kde_360_lcoe(lcoe_range))
    plt.plot(lcoe_range, kde_500_lcoe(lcoe_range))
    plt.plot(np.median(d360_lcoes), 0, '^', color=c1)
    plt.plot(np.median(d500_lcoes), 0, '^', color=c2)
    plt.ylabel('Probability Density Funtion')
    plt.xlabel('LCOE (USD/MWₕₑ)')
    
    print('LCOE:', np.median(d360_lcoes) - np.median(d500_lcoes))
    print('TES: ', np.median(d360_tes) - np.median(d500_tes))
    print('SMR: ', np.median(d360_smr) - np.median(d500_smr))

    plt.subplot(1, 3, 2)
    plt.plot(tes_range, kde_360_tes(tes_range))
    plt.plot(tes_range, kde_500_tes(tes_range))
    plt.plot(np.median(d360_tes), 0, '^', color=c1)
    plt.plot(np.median(d500_tes), 0, '^', color=c2)
    ax2 = plt.gca()
    xticks = np.arange(2000, 15000, 10000)
    ax2.set_xticks(xticks)
    plt.xlabel(r'$N_{TES}$ (MWhₜₕ)')
    plt.legend(['360 h', '500 h'], loc='upper right')

    plt.subplot(1, 3, 3)
    plt.plot(smr_range, kde_360_smr(smr_range))
    plt.plot(smr_range, kde_500_smr(smr_range))
    plt.plot(np.median(d360_smr), 0, '^', color=c1)
    plt.plot(np.median(d500_smr), 0, '^', color=c2)
    plt.xlabel(r'$N_{SMR}$ (MWₜₕ)')

    plt.show()

def plot_compare():
    tsvar_data = pd.read_pickle('tsvar_all495_10000.pkl')
    param_data = pd.read_pickle('lhs_20000_495.pkl')
    nhrs = [60, 120, 200, 250, 300, 350, 400, 600, 750, 845, 917, 990, 1062, 1134, 1206, 1254, 1345, 1393, 1405, 1410]
    lenvar_tes = [1645.44284, 1645.44284, 1810.29844, 6819.18, 13152, 13683, 13683.948595, 13683.948595, 13683.948596, 13683.948596, 13683.948596, 16968.249831, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931, 18407.877931]
    lenvar_smr = [873.302466, 873.302466, 873.302466, 873.302, 873.302, 873.302, 873.302466, 873.302466, 873.30246598, 873.30246598, 873.30246598, 873.30246597, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598, 873.30246598]
    lenvar_turb = [393.708709, 401.059414, 401.059414, 433.551, 433.551, 433.551, 433.551536, 437.167193, 518.03708793, 518.03708793, 518.03708793, 544.17739485, 544.17739485, 544.17739485, 544.17739485, 544.17739485, 544.17739485, 547.05645945, 547.05645945, 547.05645945]
    lenvar_lcoe = [64.496811, 63.015237, 63.517664, 64.7385, 66.669, 67.446, 67.761032, 67.85356, 66.967171092, 66.441484409, 66.138844198, 66.499294987, 66.782094387, 66.589946927, 66.533659873, 66.50513466, 66.403750474, 66.326018583, 66.273331911, 66.143181481]
    
    lcoe = [param_data.LCOE.dropna(), lenvar_lcoe, tsvar_data.LCOE.dropna()]
    smr = [param_data.smr_size.dropna(), lenvar_smr, tsvar_data.smr_size.dropna()]
    tes = [param_data.tes_size.dropna(), lenvar_tes, tsvar_data.tes_size.dropna()]
    turb = [param_data.turb_size.dropna(), lenvar_turb, tsvar_data.turb_size.dropna()]

    xticks = ['Parameter', 'Dispatch Length', 'Time Series']

    ax1 = plt.subplot(1, 4, 1)
    plt.violinplot(lcoe)
    plt.xticks([1, 2, 3], xticks)
    plt.setp(ax1.get_xticklabels(), ha='right', rotation=20)
    plt.ylabel('LCOE')

    ax2 = plt.subplot(1, 4, 2)
    plt.violinplot(smr)
    plt.xticks([1, 2, 3], xticks)
    plt.setp(ax2.get_xticklabels(), ha='right', rotation=20)
    plt.ylabel(r'$N_{SMR}$ (MWₜₕ)')
    
    ax3 = plt.subplot(1, 4, 3)
    plt.violinplot(tes)
    plt.xticks([1, 2, 3], xticks)
    plt.setp(ax3.get_xticklabels(), ha='right', rotation=20)
    plt.ylabel(r'$N_{TES}$ (MWhₜₕ)')
    
    ax4 = plt.subplot(1, 4, 4)
    plt.violinplot(turb)
    plt.xticks([1, 2, 3], xticks)
    plt.setp(ax4.get_xticklabels(), ha='right', rotation=20)
    plt.ylabel(r'$N_{turb}$ (MWe)')

    plt.show()
