import numpy as np
import scipy.sparse

import sptrack as spt



particles = np.array([[1.2, 3.4],
                      [2.3, 2.0],
                      [4.5, 3.1]])
                      

#%%
def test_subdivide_image():
    regions = spt.subdivide_image(256, 256, 8)
    assert len(regions) == 32 * 32
    
    regions = spt.subdivide_image(11, 10, 2)
    assert len(regions) == 6 * 5
    

#%%
def test_particle_networks():
    #%% First test
    pdg = scipy.sparse.lil_matrix((6, 6))
    pdg[0,2] = 1.0
    pdg[0,3] = 1.2
    pdg[1,3] = 0.9
    pdg[1,4] = 1.1
    pdg[1,5] = 2.0
    n_0 = 2
    network_labels = spt.particle_networks(pdg, n_0)
    assert (network_labels == np.zeros(6)).all()
    
    connections = spt.connect_particles(pdg, network_labels, n_0)
    assert connections == ((0, 0), (1, 1))
    
    #%% Second test
    pdg = scipy.sparse.lil_matrix((8, 8))
    pdg[0,3] = 1.4
    pdg[0,4] = 0.4
    pdg[1,5] = 1.1
    pdg[1,6] = 0.9
    pdg[1,7] = 2.0
    pdg[2,4] = 0.6
    n_0 = 3
    assert (spt.particle_networks(pdg, n_0) 
               == np.array([0, 1, 0, 0, 0, 1, 1, 1])).all()
    
    
    