import numpy as np
import scipy.sparse
import pandas as pd
import sklearn.utils.linear_assignment_
import numba


#%%
class Subregion(object):
    """
    Container for information of particles in a subregion of an image.
    """

    def __init__(self, width=None, i_range=[], j_range=[], neighbors=[],
                 particles=[], particle_ids=[]):
        """
        Container for information of particles in a subregion of an image.

        Attributes
        ----------
        width : int
            Width of the subregion
        i_range : 2-tuple of ints
            the starting and ending index, inclusive, of the rows
            included in the subregion
        j_range : 2-tuple of ints
            the starting and ending index, inclusive, of the columns
            included in the subregion
        neighbors : list of 2-tuples of ints
            List of the i-j indices of neighboring subregions.
            Diagonally connected subregions are considered connected.
        particles : ndarray, shape (n_particles, 2)
            particles[k] is the (i, j) position  k'th particle in
            the subregion.
        particle_ids : array_list
            particle_ids[k] is the identifer for the k'th particle in
            the subregion
        """
        self.width = width
        self.i_range = i_range
        self.j_range = j_range
        self.neighbors = neighbors
        self.particles = particles
        self.particle_ids = particle_ids


#%%
class Track(object):
    """
    Class containing information about the track of a particle.
    """

    def __init__(self, i=[], j=[], frame=[]):
        """
        Initialize track values
        """
        self.frame = frame
        self.i = i
        self.j = j


#%%
def subdivide_image(n_r, n_c, subregion_width):
    """
    Divide an n_r by n_c image into square subregions.

    Parameters
    ----------
    n_r : int
        Number of rows of pixels in the image to be subdivided.
    n_c : int
        The number of columns of pixels in the image to be subdivided.
    subregion_width : int
        The width of the square subregions in pixels.
    particles : ndarray, shape (n_particles, 2)
        particles[k] is the (i, j) position  k'th particle in
        the subregion

    Returns
    -------
    output : list of list of Subregion instances
        output[i][j] contains a Subregion instance populated
        with width, i_range, j_range, and neighbors attributes.

    Notes
    -----
    .. No information about particles is included, only the spatial
       information of the subregions.
    """

    # Get number of rows and columns of subregions
    n_row = n_r // subregion_width + (n_r % subregion_width > 0)
    n_col = n_c // subregion_width + (n_c % subregion_width > 0)

    # Determine neighbors regions, taking care of hitting boundaries
    neighbors = _ListOfLists(
                    [[[] for j in range(n_col)] for i in range(n_row)])
    for i in range(n_row):
        for j in range(n_col):
            if i > 0:
                if j > 0:
                    neighbors[i,j].append((i - 1, j - 1))

                neighbors[i,j].append((i - 1, j))

                if j < n_col - 1:
                    neighbors[i,j].append((i - 1, j + 1))

            if j > 0:
                neighbors[i,j].append((i, j - 1))

            if j < n_col - 1:
                neighbors[i,j].append((i, j + 1))

            if i < n_row - 1:
                if j > 0:
                    neighbors[i,j].append((i + 1, j - 1))

                neighbors[i,j].append((i + 1, j))

                if j < n_col - 1:
                    neighbors[i,j].append((i + 1, j + 1))

    # Determine ranges of pixels in bins
    i_range = []
    for i in range(n_row):
        i_range.append((i * subregion_width,
                        min((i+1) * subregion_width, n_c) - 1))

    j_range = []
    for j in range(n_col):
        j_range.append((j * subregion_width,
                        min((j+1) * subregion_width, n_r) - 1))

    # Initialize subregion list
    subregions = SubregionList(
                    [[None for j in range(n_col)] for i in range(n_row)])

    # Create subregions
    for i in range(n_row):
        for j in range(n_col):
            subregions[i,j] = Subregion(
                width=subregion_width, i_range=i_range[i],
                j_range=j_range[j], neighbors=tuple(neighbors[i,j]))

    return subregions


#%%
def populate_subregions(particles, subregions):
    """
    Populate the particles and particle_ids fields of a Subregion
    instance.

    Parameters
    ----------
    particles : ndarray, shape (n_particles, 2)
        particles[k] is the (i, j) position  k'th particle in
        the entire image
    subregions : List of of lists of Subregion instances
        List of lists of pre-instantiated Subregion instance missing only
        particle information, as would be returned by the function
        subdivide_image.

    Returns
    -------
    output : List of lists of Subregion instances
        List of lits of Subregion instance populated with the
        particle positions.

    Notes
    -----
    .. The particle_ids and particles lists in the Subregions
       instances should be pre-initialized to empty lists.
    """
    
    assert type(subregions) == SubregionList, \
                                'subregions must be a SubregionList instance.'

    # Determine width of subregions (same for all in list)
    width = subregions[0,0].width

    # Initialize all particle counts
    for ind in subregions.indices:
        subregions[ind].particles = []
        subregions[ind].particle_ids = []
   
    # Initialize list of indices of subregions that have particles
    subregion_inds = set()

    for k, particle in enumerate(particles):
        # Determine subregion indices
        i, j = (particle // width).astype(int)
        subregion_inds.add((i,j))

        # Place particle in the correct subregion
        subregions[i,j].particle_ids.append(k)
        subregions[i,j].particles.append(particle)
        
    # Convert all occupied particle arrays to NumPy arrays and IDs to tuples
    for ind in subregion_inds:
        subregions[ind].particles = np.array(subregions[ind].particles)
        subregions[ind].particle_ids = tuple(subregions[ind].particle_ids)
        
    # Return output subregions and list of those that have particles in them
    return subregions, tuple(subregion_inds)
    

#%%
def close_particles(coord, subregions, d=None):
    """
    Find all particles close to a specified point in space.

    Parameters
    ----------
    coord : array_like, shape (2,)
        The point in space where we want to find close particles.
    subregions : list of lists of Subregion instances
        List of lists of Subregion instances with particle positions
        and IDs, as would be returned by the function
        populate_subregions.
    d : float, default is subergion width.
        Distance in units of pixels below which a particle is
        considered close.

    Returns
    -------
    close_ids : tuple of ints
        Tuple containing the particle IDs of all neighbors.
    pp_dist : nd_array
        Array containing the particle-particle distance between
        coord and the neighbors

    Notes
    -----
    .. Searchs the subregion containing coord and neighboring
       subregions to find particles within a distance d.
    """

    # Set up sqaured distance between particles
    w = subregions[0,0].width
    if d is None:
        d2 = float(w)**2
    elif d > w:
        raise ValueError('d must be <= subregion width')
    else:
        d2 = float(d)**2
        
    # Get the subregion indices containing coord
    i, j = (coord // w).astype(int)

    # List of particle IDs that are close
    close_ids = []
    pp_dist2 = []

    # Find particles in same subregion
    for k, particle in enumerate(subregions[i,j].particles):
        pp2 = _pp_dist2(coord, particle)
        if pp2 < d2:
            close_ids.append(subregions[i,j].particle_ids[k])
            pp_dist2.append(pp2)

    # Find all particles in neighboring subregions
    for n in subregions[i,j].neighbors:
        for k, particle in enumerate(subregions[n].particles):
            pp2 = _pp_dist2(coord, particle)
            if pp2 < d2:
                close_ids.append(subregions[n].particle_ids[k])
                pp_dist2.append(pp2)

    # Return a tuple close particle ids and an array with the square distances
    return tuple(close_ids), np.array(pp_dist2)


#%%
def particle_distances(particles_0, particles_1, subregions_1, d=None):
    """
    Determine local networks of possible particle "connections" between
    two sets of particles.
    
    Paramters
    ---------
    particles_0 : ndarray, shape(n_0, 2)
        Array of particle positions for first frame.
    particles_1 : ndarray, shape(n_1, 2)
        Array of particle positions for second frame.
    subregions_1 : SubregionList
        SubregionList instance populated with particles for
        the second frame.

    Returns
    -------
    output : sparse CSR matrix
        A CSR matrix representation of a compressed sparse graph.
        Entry i, n_0 + j is nonzero and equal to the square of 
        the distance between particles_0[i] and particles_1[j].
        
    Notes
    -----
    .. The outputted graph is strongly pruned.  Only connections that
       are within a distance d are allowed.
    .. If the interparticle distance is identically zero, a small number
       (1e-12) is put in instead.  This avoids confusion using sparse
       matrices.
    """

    # Determine number of particles in frame 0 and frame 1
    n_0 = particles_0.shape[0]
    n_1 = particles_1.shape[0]
    
    # Set up lil sparse matrix for entering graph
    graph = scipy.sparse.lil_matrix((n_0 + n_1, n_0 + n_1), dtype=np.float)

    # Connect all frame 0 particles to possible frame 1 particles
    for k, particle in enumerate(particles_0):
        close_ids, pp_dist2 = close_particles(particle, subregions_1, d=d)
        for i, pid in enumerate(close_ids):
            graph[k, pid+n_0] = max(1e-12, pp_dist2[i])
            
    return graph.tocsr()


#%%
def particle_networks(particle_distance_graph, n_0):
    """
    Determine local networks of possible particle "connections" between
    two sets of particles.
    
    Parameters
    ----------
    particle_distance_graph : array_like or sparse matrix
        An compressed sparse undirected graph represented as a matrix.
    
    Returns
    -------
    network_labels_0 : ndarray of ints, shape (n_0,)
        Labels for network that each particle in the first frame
        belongs to.
    network_labels_1 : ndarray of ints, shape (n_1,)
        Labels for network that each particle in the second frame
        belongs to.
    """                                                        
    # Compute connected networks
    _, network_labels = scipy.sparse.csgraph.connected_components(
                                      particle_distance_graph, directed=False)

    # Split out the network labels for the first and second frames
    return network_labels
    

#%%
def connect_particles(particle_distance_graph, network_labels, n_0):
    """
    Connect particles based on minimizing sum of squared distances.
    
    Parameters
    ----------
    
    Returns
    -------
    """

    # Get the unique networks containing particles in the first frame
    networks = np.unique(network_labels)
    
    # Slice out useful parts of particle_distance_graph and labels
    pdg = particle_distance_graph[:n_0, n_0:]
    net_rows = network_labels[:n_0]
    net_cols = network_labels[n_0:]
    
    # Initiate list of particle connections
    connections = []
   
    # Loop through each connected network and make linear assignment
    for net in networks:
        # Special case of a one-edge network
        if (net_rows==net).sum() == 1 and (net_cols==net).sum() == 1:
            connections.append((np.nonzero(net_rows==net)[0][0],
                                np.nonzero(net_cols==net)[0][0]))
        else:
            # Extract cost matrix
            cost_mat = pdg[np.ix_(net_rows==net, net_cols==net)].toarray()
            
            # Set zero elements to infinity (cannot be connections)
            cost_mat[cost_mat==0] = np.inf
    
            # Compute connections in local network using Hungarian algorithm
            connecs = \
                 sklearn.utils.linear_assignment_.linear_assignment(cost_mat)
    
            # Recast indices of particles in respective frames
            f0_parts = np.nonzero(net_rows==net)[0]
            f1_parts = np.nonzero(net_cols==net)[0]
            
            for conn in connecs:
                connections.append((f0_parts[conn[0]], f1_parts[conn[1]]))
    
    return tuple(connections)
    

#%%
class _ListOfLists(object):
    """
    Generic class of list of lists to allow easy indexing,
    i.e., [i,j] instead of [i][j].
    """
    def __init__(self, list_of_lists):
        self.list_of_lists = list_of_lists
        self.i_len = len(list_of_lists)
        self.j_len = tuple([len(list_of_lists[i]) for i in range(self.i_len)])
        self.indices = tuple([(i, j) for i in range(self.i_len) 
                                        for j in range(self.j_len[i])])

    def __getitem__(self, key):
        return self.list_of_lists[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.list_of_lists[key[0]][key[1]] = value
        
    def __len__(self):
        return len(self.indices)


#%%
class SubregionList(_ListOfLists):
    """
    Generic class of list of lists to allow easy indexing,
    i.e., [i,j] instead of [i][j].
    """
    pass


#%%
@numba.jit(nopython=True)
def _pp_dist2(point_1, point_2):
    """
    Square of point-point distance
    """
    return (point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2
