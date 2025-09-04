import json
import numpy as np
import matplotlib.pyplot as plt

################################################################################
lumi = 3.2 * 1000

metadata = {'ggh': {'xsec': 28.3, 
                    'filter': 0.000124, 
                    'kfactor': 1.717,
                    'num_events': 1598000,
                    'sum_weights': 45231011.19517517},
            'zz':  {'xsec': 1.2974,
                    'filter': 1.0,
                    'kfactor': 1.0,
                    'num_events': 250000,
                    'sum_weights': 8900161062.01826},}


################################################################################

def scale_factor(process):
    proc = metadata[process]
    scale = (lumi * proc['xsec'] * (proc['kfactor']) * proc['filter']) / proc['sum_weights']

    return scale

with open('data/masses.json', "r") as f:
    data = json.load(f)

data_2015 = np.array(data['2015']['masses'])
counts, bin_edges = np.histogram(data_2015, bins=36)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
errors = np.sqrt(counts)

plt.errorbar(bin_centers, counts, yerr=errors,
    fmt="o", color="black", label=r"Data ($3.2fb^{-1}$)")

data_ggh = np.array(data['ggh']['masses'])
weight_ggh = np.array(data['ggh']['weights']) * scale_factor('ggh')
data_zz = np.array(data['zz']['masses'])
weight_zz = np.array(data['zz']['weights']) * scale_factor('ggh')

plt.hist([data_zz, data_ggh], 
         bins=36, 
         stacked=True, 
         label=[r'$ZZ^{*}$', 'Higgs'], 
         weights=[weight_zz, weight_ggh], 
         color=['red', 'skyblue',])

plt.title("Histogram from JSON")
plt.xlabel("Mass [MeV]")
plt.ylabel("Entries")
plt.legend() 

plt.savefig("histogram.png") 
