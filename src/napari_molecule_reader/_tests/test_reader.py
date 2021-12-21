from napari_molecule_reader import napari_get_reader


fake_pdb = """
ATOM     29  P    DA B   2     -52.420 -28.259   4.599  1.00327.69           P
"""


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = str(tmp_path / "myfile.pdb")
    with open(my_test_file, 'w+') as f:
        f.write(fake_pdb)

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
