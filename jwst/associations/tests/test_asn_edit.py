import pytest
from astropy.utils.data import get_pkg_data_filename

from jwst.associations import asn_edit

JSON_FILE = get_pkg_data_filename("data/asn_level2.json", package="jwst.associations.tests")


def test_add_asn():
    """Test adding a product to an association"""
    asn = asn_edit.reader(JSON_FILE)
    asn = asn_edit.add(asn, ["test_lrs5_rate.fits"], "science")

    nproducts = len(asn["products"])
    assert nproducts == 6
    found = asn_edit._lookup(asn, "test_lrs5_rate.fits")
    assert len(found) == nproducts


@pytest.mark.parametrize("ignore", [False, True])
def test_remove_asn(ignore):
    """Test removing a product from an association"""
    asn = asn_edit.reader(JSON_FILE)
    assert len(asn["products"]) == 6
    asn = asn_edit.remove(asn, ["test_lrs1_rate.fits"], ignore)

    assert len(asn["products"]) == 5
    found = asn_edit._lookup(asn, "test_lrs1_rate.fits")
    assert len(found) == 0
