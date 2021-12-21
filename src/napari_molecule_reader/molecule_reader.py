from pathlib import Path

import numpy as np
import pandas as pd
import atomium

from .bonds import guess_bonds
from .colors import COLORS


property_columns = ['element', 'resid', 'resname', 'chain', 'cov_radius']


def read_molecule(path):
    name = f'{Path(path).stem}'
    raw_data = atomium.open(path)
    # parse data and get everything in one dataframe
    data = []
    for i, model in enumerate(raw_data.models):
        atoms = []
        for atom in model.atoms():
            atoms.append([
                *atom.location,
                atom.element,
                atom.het.id if atom.het else '',
                atom.het.name if atom.het else '',
                atom.chain.id if atom.chain else '',
                atom.covalent_radius,
            ])
        atoms = pd.DataFrame(atoms, columns=['x', 'y', 'z', *property_columns])
        atoms['model'] = i
        data.append(atoms)
    data = pd.concat(data)

    # this could be done with property mappings in the future
    sizes = atoms['cov_radius'] * 2
    colors = np.empty((len(data), 3), float)
    elem = atoms['element'].to_numpy()
    for el in np.unique(elem):
        colors[elem == el] = COLORS.get(el, [255, 0, 102])  # pink
    colors /= 255

    # times 100 to avoid artifacts of high vispy zoom
    sizes *= 100

    properties = atoms[property_columns]

    # get different assemblies
    assemblies = {}
    if raw_data.assemblies:
        for ass in raw_data.assemblies:
            assemblies[ass['id']] = {'affine': []}
            for tr in ass['transformations']:
                # TODO: assemblies may use different combinations of chains, and not all of them
                affine = np.eye(4)
                affine[:3, :3] = tr['matrix']
                affine[:3, -1] = tr['vector']
                assemblies[ass['id']]['affine'].append(affine)
    else:
        assemblies[0] = {'affine': [np.eye(4)]}

    # get coords, models go in 4th dimension
    coords = atoms[['model', 'x', 'y', 'z']].to_numpy()
    coords[:, 1:] *= 100
    # guess bonds based on distances
    bonds = guess_bonds(data)

    # apply affines and concatenate properties
    # coordinates in homogeneous space
    coords_h = np.ones((len(coords), 4))
    coords_h[:, :3] = coords[:, 1:]
    for ass, data in assemblies.items():
        ass_coords = []
        ass_bonds = []
        for i, affine in enumerate(data['affine']):
            # apply affines to coordinates and add to the list
            transformed = coords.copy()
            transformed[:, 1:] = (coords_h @ affine.T)[:, :-1]
            ass_coords.append(transformed)
            ass_bonds.append(bonds + i * len(coords))
        ass_coords = np.concatenate(ass_coords)
        ass_bonds = np.concatenate(ass_bonds)
        ass_bond_coords = ass_coords[ass_bonds]
        # for vectors, napari wants [origin, vector], not [coord1, coord2]
        ass_bond_coords[:, 1, 1:] -= ass_bond_coords[:, 0, 1:]
        # 'model' coord must be zero, otherwise vectors will be 4D and span models
        ass_bond_coords[:, 1, 0] = 0

        assemblies[ass]['coords'] = ass_coords
        assemblies[ass]['bonds'] = ass_bond_coords

    def tile(stuff, assembly):
        amount = len(assemblies[assembly]['affine'])
        if isinstance(stuff, np.ndarray):
            return np.tile(stuff, (amount, 1))
        if isinstance(stuff, (pd.DataFrame, pd.Series)):
            return pd.concat([stuff for _ in range(amount)])

    layers = []
    for ass, data in assemblies.items():
        # create a vec layers for the bonds and a points layer for the atoms
        atom_kwargs = dict(
            name=f'{name} - {ass} - atoms',
            properties=tile(properties, ass),
            size=tile(sizes, ass),
            face_color=tile(colors, ass),
            edge_width=0,
            shading='spherical',
        )

        bond_kwargs = dict(
            name=f'{name} - {ass} - bonds',
            edge_width=5,
        )

        layers.extend([
            (data['bonds'], bond_kwargs, 'vectors'),
            (data['coords'], atom_kwargs, 'points'),
        ])

    # fix for empty vector layers until napari#2295 is merged
    return layers


def read_molecules(paths):
    paths = [paths] if isinstance(paths, str) else paths

    return [tup for path in paths for tup in read_molecule(path)]
