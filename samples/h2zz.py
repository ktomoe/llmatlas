import uproot
import vector
import json
import glob
import math
import numpy as np
import awkward as ak


################################################################
metadata = {
  '2015': {'mode': 'data', 'file_path': './raid/data/2015/*.root*'},
  #'2016': {'mode': 'data', 'file_path': './raid/data/2016/*.root*'},
  'ggh' : {'mode': 'mc',   'file_path': './raid/mc/ggh/*.root*'},
  'zz' : {'mode': 'mc',   'file_path': './raid/mc/zz/*.root*'},
}
event_container = 'EventInfoAuxDyn'
vertex_container = 'PrimaryVerticesAuxDyn'
muon_container = 'AnalysisMuonsAuxDyn'
combined_muon_container = 'CombinedMuonTrackParticlesAuxDyn'
electron_container = 'AnalysisElectronsAuxDyn'

branches = {
    'event_weight': f'{event_container}.mcEventWeights',
    #'event_pileup': f'{event_container}.PileupWeight_NOSYS',
    'event_beamz': f'{event_container}.beamPosZ',
    'vertex_type': f'{vertex_container}.vertexType',
    'vertex_z': f'{vertex_container}.z',
    'mu_pt': f'{muon_container}.pt',
    'mu_eta': f'{muon_container}.eta',
    'mu_phi': f'{muon_container}.phi',
    'mu_charge': f'{muon_container}.charge',
    'mu_ptcone': f'{muon_container}.ptcone20_Nonprompt_All_MaxWeightTTVA_pt1000',
    'mu_etcone': f'{muon_container}.topoetcone20',
    'mu_type': f'{muon_container}.muonType',
    'mu_link': f'{muon_container}.combinedTrackParticleLink.m_persIndex',
    'cb_mu_z0': f'{combined_muon_container}.z0',
    'cb_mu_theta': f'{combined_muon_container}.theta',
    'cb_mu_vz': f'{combined_muon_container}.vz',
    'el_pt': f'{electron_container}.pt',
    'el_eta': f'{electron_container}.eta',
    'el_phi': f'{electron_container}.phi',
    'el_charge': f'{electron_container}.charge',
    'el_ptcone': f'{electron_container}.ptcone20_Nonprompt_All_MaxWeightTTVALooseCone_pt1000',
    'el_etcone': f'{electron_container}.topoetcone20',
    'el_loose': f'{electron_container}.DFCommonElectronsLHLoose',
    }

################################################################

def good_vertices(event, num_vertices):
    """ Returns selected vertices"""
    rtn_vertices = []

    for ii in range(num_vertices):
        vertex_type = event[branches['vertex_type']][ii]
        vertex_z = event[branches['vertex_z']][ii]

        if vertex_type != 1:
            continue 

        rtn_vertices.append(vertex_z)

    return rtn_vertices


def good_muons(event, num_muons):
    """ Returns selected muons with vector formats"""
    rtn_muons = []

    for ii in range(num_muons):
        mu_pt = event[branches['mu_pt']][ii]
        mu_eta = event[branches['mu_eta']][ii]
        mu_phi = event[branches['mu_phi']][ii]
        mu_charge =  event[branches['mu_charge']][ii]
        mu_ptcone =  event[branches['mu_ptcone']][ii]
        mu_etcone =  event[branches['mu_etcone']][ii]
        mu_type =  event[branches['mu_type']][ii]
        mu_link =  event[branches['mu_link']][ii]

        if mu_type != 0:
            continue

        if mu_pt <= 5000.:
            continue

        if abs(mu_eta) >= 2.7:
            continue

        if mu_ptcone / mu_pt > 0.15:
            continue

        if mu_etcone / mu_pt > 0.20:
            continue

        pv_z0 = event[branches['vertex_z']][0]
        z0 = event[branches['cb_mu_z0']][mu_link]
        theta = event[branches['cb_mu_theta']][mu_link]
        beamz = event[branches['event_beamz']]

        z0sin = (pv_z0 - (z0+beamz)) * math.sin(theta)

        if abs(z0sin) > 0.5:
            continue
        
        muon = vector.obj(pt=mu_pt, eta=mu_eta, phi=mu_phi, mass=105.6)
        rtn_muons.append({'vector': muon, 'pid': 13 * mu_charge, 'index': ii})

    return rtn_muons


def good_electrons(event, num_electrons):
    """ Returns selected electrons with vector formats"""
    rtn_electrons = []

    for ii in range(num_electrons):
        el_pt = event[branches['el_pt']][ii]
        el_eta = event[branches['el_eta']][ii]
        el_phi = event[branches['el_phi']][ii]
        el_charge =  event[branches['el_charge']][ii]
        el_ptcone =  event[branches['el_ptcone']][ii]
        el_etcone =  event[branches['el_etcone']][ii]
        el_loose =  event[branches['el_loose']][ii]

        if not el_loose:
            continue

        if el_pt <= 7000.:
            continue

        if abs(el_eta) >= 2.47:
            continue

        if el_ptcone / el_pt > 0.15:
            continue

        if el_etcone / el_pt > 0.20:
            continue

        electron = vector.obj(pt=el_pt, eta=el_eta, phi=el_phi, mass=0.511)
        rtn_electrons.append({'vector': electron, 'pid': 11 * el_charge, 'index': ii})

    return rtn_electrons


def good_dileptons(leptons):
    """ Retuns SFOS di-leptons orderd by Z mass"""

    tmp_dileptons = []
    masses = []
    for ii, lepton0 in enumerate(leptons):
        for jj, lepton1 in enumerate(leptons[ii+1:]):
            if abs(lepton0['pid']) !=  abs(lepton1['pid']):
                continue

            if lepton0['pid'] * lepton1['pid'] > 0:
                continue

            dilepton = {'vector':  lepton0['vector'] + lepton1['vector'],
                        'childs':  [lepton0, lepton1],
                        'pid':     abs(lepton0['pid']),
                        'indices': [lepton0['index'], lepton1['index']]}
            tmp_dileptons.append(dilepton)

            mass_diff = abs(91187.6 - dilepton['vector'].mass)
            masses.append(mass_diff)

    masses = np.array(masses)
    has_done = np.zeros(len(tmp_dileptons), dtype=bool)

    rtn_dileptons = []
    while not all(has_done):
        min_index = np.where(has_done == False, masses, np.inf).argmin()

        min_dilepton = tmp_dileptons[min_index]
        rtn_dileptons.append(min_dilepton)

        for ii, dilepton in enumerate(tmp_dileptons):
            if dilepton['pid'] != min_dilepton['pid']:
                continue

            if bool(set(min_dilepton['indices']) & set(dilepton['indices'])):
                has_done[ii] = True

    return rtn_dileptons


def good_zbosons(dileptons):
    rtn_zbosons = []

    child0 = dileptons[0]['childs'][0]
    child1 = dileptons[0]['childs'][1]
    child2 = dileptons[1]['childs'][0]
    child3 = dileptons[1]['childs'][1]

    pts = [child0['vector'].pt,
           child1['vector'].pt,
           child2['vector'].pt,
           child3['vector'].pt]
    
    pts = np.array(pts)

    if np.sum(pts > 20000.) < 1:
        return rtn_zbosons 

    if np.sum(pts > 15000.) < 2:
        return rtn_zbosons 

    if np.sum(pts > 10000.) < 3:
        return rtn_zbosons 

    zboson0 = dileptons[0]['vector']
    zboson1 = dileptons[1]['vector']

    if 50000. < zboson0.mass < 106000.:
        rtn_zbosons.append(dileptons[0])

    if 12000. < zboson1.mass < 115000.:
        rtn_zbosons.append(dileptons[1])

    return rtn_zbosons    


def good_higgs(zbosons):
    rtn_higgs = []

    higgs = zbosons[0]['vector'] + zbosons[1]['vector']

    if 80000. < higgs.mass < 170000.:
        rtn_higgs.append( {'vector': higgs})

    return rtn_higgs


def event_loop(file_path, mode):
    arrays = uproot.concatenate(file_path, filter_name=branches.values())

    num_events = len(arrays)
    event_weights = arrays[branches['event_weight']].to_numpy()[:, 0]
    #pileup_weights = arrays[branches['event_pileup']].to_numpy()

    num_vertices = ak.num(arrays[branches['vertex_z']]).to_numpy()
    num_muons = ak.num(arrays[branches['mu_pt']]).to_numpy()
    num_electrons = ak.num(arrays[branches['el_pt']]).to_numpy()

    arrays = arrays.to_list()
    masses = []
    weights = []
    for ii, event in enumerate(arrays):

        if (ii % 10000 == 0) and (ii != 0):
            print (f'inf> {ii}/{num_events} events processed')

        if (num_electrons[ii] + num_muons[ii]) < 4:
            continue

        vertices =  good_vertices(event, num_vertices[ii])

        if len(vertices) < 1:
           continue

        muons = good_muons(event, num_muons[ii])
        electrons = good_electrons(event, num_electrons[ii])

        if len(muons + electrons) < 4:
           continue

        dileptons = good_dileptons(muons+electrons)

        if len(dileptons) < 2:
           continue

        zbosons = good_zbosons(dileptons)

        if len(zbosons) < 2:
           continue

        higgs = good_higgs(zbosons)

        if len(higgs) < 1:
           continue

        masses.append(float(higgs[0]['vector'].mass))
        weight = float(event_weights[ii])
        weights.append(weight)

    return {'masses': masses, 'weights': weights}


if __name__ == "__main__":

    results = {}   
    for key, value in metadata.items(): 
        file_names = glob.glob(value['file_path'])

        file_path = []
        for ifile in file_names:
            file_name = f'{ifile}:CollectionTree'
            print (f'inf> {file_name}')
            file_path.append(file_name)

        print (f'inf> {len(file_path)} files will be processed')

        results[key] = event_loop(file_path, value['mode'])
        print (f'inf> {len(results[key]["weights"])} events selected for {key}')

    with open(f'data/masses.json', 'w') as f:
        json.dump(results, f)
