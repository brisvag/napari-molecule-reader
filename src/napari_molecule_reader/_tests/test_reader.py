import numpy as np
from napari_molecule_reader.molecule_reader import read_molecules


def test_reader(tmp_path):
    my_test_file = str(tmp_path / "myfile.pdb")
    fake_pdb = """
ATOM     29  P    DA B   2     -52.420 -28.259   4.599  1.00327.69           P
    """
    with open(my_test_file, 'w+') as f:
        f.write(fake_pdb)

    vec_data, point_data = read_molecules(my_test_file)
    assert vec_data[0].shape == (0, 2, 4)
    assert point_data[0].shape == (1, 4)
    properties = {
        'element': {0: 'P'},
        'resid': {0: 'B.2'},
        'resname': {0: 'DA'},
        'chain': {0: ''},
        'cov_radius': {0: 1.07},
    }
    assert np.all(point_data[1]['properties'].to_dict() == properties)
