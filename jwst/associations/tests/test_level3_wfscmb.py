from astropy.utils.data import get_pkg_data_filename

from jwst.associations.main import Main
from jwst.associations.tests.helpers import combine_pools


def jitter_name_base(fname):
    """
    Remove the post fix from a pool applicate file name.
    """
    sf = fname.split("_")
    new_name = "_".join([el for el in sf[:-1]])
    return new_name


def get_jitter_not_jitter(pool_path):
    """
    Get a list of pool candidates marked as jitter.
    """
    with open(pool_path, "r") as fd:
        lines = fd.readlines()
    header = lines[0]
    pool = lines[1:]

    delim = "|"
    sheader = header.split(delim)
    didx = sheader.index("dms_note")
    fidx = sheader.index("filename")

    val = "WFSC_LOS_JITTER".lower()
    jitter = []
    for candidate in pool:
        scan = candidate.split(delim)
        dms_note = scan[didx]
        if val in dms_note:
            jitter_can = scan[fidx]
            jitter_base = jitter_name_base(jitter_can)
            jitter.append(jitter_base)

    return jitter


def get_expnames(asn):
    """
    Get the list of all exp_names in an association file.
    """
    expnames = []
    for product in asn["products"]:
        members = product["members"]
        for member in members:
            expnames.append(member["expname"])
    return expnames


def jitter_present(jitter, associations):
    """
    For level 3 images make sure pool candidates identified as
    jitter are not present in association files.
    """
    for asn in associations:
        # Only care about level 3 images
        if "image3" not in asn.asn_name:
            continue
        # For level 3 images make sure pool candidates identified as
        # jitter are not present in association files.
        exp_names = get_expnames(asn)
        for jit in jitter:
            for ename in exp_names:
                if jit in ename:
                    return True

    return False


def test_level3_wfscmb_jitter_suppression(tmp_path):
    """
    Make sure no candidate from the pool with DMS_NOTE equal to
    WFSC_LOS_JITTER is in any association.
    """
    pool_path = str(tmp_path / "pool.csv")
    pool = combine_pools(
        get_pkg_data_filename("data/pool_033_wfs_jitter.csv", package="jwst.associations.tests")
    )
    pool.write(pool_path, format="ascii", delimiter="|")
    cmd_args = [pool_path, f"--path={tmp_path}"]
    generated = Main.cli(cmd_args)
    jitter = get_jitter_not_jitter(pool_path)
    assert not jitter_present(jitter, generated.associations)
