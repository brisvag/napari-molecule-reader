import numpy as np
from scipy.spatial.ckdtree import cKDTree


def guess_bonds(atoms, fudge=1.1):
    """
    guess bonds based on distances and vdw radii
    """
    if len(atoms) == 0:
        return np.empty((0, 2))

    coords = atoms[['x', 'y', 'z']].to_numpy()
    elem = atoms['element'].to_numpy()

    # tolerance of 0.4 from chimera
    # https://www.cgl.ucsf.edu/chimerax/docs/user/atomtypes.html
    radii = atoms['cov_radius'].to_numpy() + 0.4

    max_dist = np.max(radii)
    if max_dist == 0:
        return np.empty((0, 2))

    # get all neighbours
    tree = cKDTree(coords)
    matrix = tree.sparse_distance_matrix(tree, max_dist * fudge)
    pairs, dists = zip(*matrix.items())

    # discard duplicates (symmetric matrix)
    pairs = np.array(pairs)
    non_duplicate = pairs[:, 0] < pairs[:, 1]
    pairs = pairs[non_duplicate]

    # get vdw radii for each pair and calculate their distance
    radii_pairs = radii[pairs]
    vdw_dists = radii_pairs.sum(axis=1) * fudge

    # bonds are where the actual distance is less than the sum of the vdw radii
    dists = np.array(dists)[non_duplicate]
    bonds = dists <= vdw_dists

    # inter-residue H-bonds
    elem_pairs = elem[pairs]
    hydro = np.any(np.isin(elem_pairs, ('H', 'D')), axis=1)
    resid = atoms['resid'].to_numpy()
    same_resid = np.equal.reduce(resid[pairs], axis=1)
    hydro_different_resid = np.logical_and(hydro, ~same_resid)

    # same chain
    chain = atoms['chain'].to_numpy()
    same_chain = np.equal.reduce(chain[pairs], axis=1)

    # same model
    model = atoms['model'].to_numpy()
    same_model = np.equal.reduce(model[pairs], axis=1)

    # alternate loc indicator
    # alt_loc = atoms['alt_location'].to_numpy()
    # TODO

    # find where both are hydrogens
    both_hydro = np.all(np.isin(elem_pairs, ('H', 'D')), axis=1)

    # keep only if not inter-residue H-bonds/both hydrogens, and if different chains/models
    keep = np.logical_and.reduce([
        ~both_hydro,
        ~hydro_different_resid,
        same_chain,
        same_model,
    ])

    return pairs[np.logical_and(bonds, keep)]
