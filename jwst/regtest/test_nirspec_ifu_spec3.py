"""Regression tests for NIRSpec IFU"""

import pytest

from jwst.regtest import regtestdata as rt

# Mark all tests in this module
pytestmark = [pytest.mark.bigdata, pytest.mark.slow]


@pytest.fixture(scope="module")
def run_spec3_multi(rtdata_module, resource_tracker):
    """Run Spec3Pipeline"""
    rtdata = rtdata_module

    step_params = {
        "input_path": "nirspec/ifu/jw01249-o005_20230622t074431_spec3_00001_asn.json",
        "step": "calwebb_spec3",
        "args": {
            "--steps.master_background.save_results=true",
            "--steps.outlier_detection.save_results=true",
            "--steps.resample_spec.save_results=true",
            "--steps.cube_build.save_results=true",
            "--steps.extract_1d.save_results=true",
            "--steps.combine_1d.save_results=true",
        },
    }
    with resource_tracker.track():
        rtdata = rt.run_step_from_dict(rtdata, **step_params)
    return rtdata


def test_log_tracked_resources_spec3(log_tracked_resources, run_spec3_multi):
    log_tracked_resources()


@pytest.mark.parametrize(
    "output",
    [
        "jw01249005001_03101_00001_nrs1_o005_crf.fits",
        "jw01249005001_03101_00001_nrs2_o005_crf.fits",
        "jw01249005001_03101_00002_nrs1_o005_crf.fits",
        "jw01249005001_03101_00002_nrs2_o005_crf.fits",
        "jw01249005001_03101_00003_nrs1_o005_crf.fits",
        "jw01249005001_03101_00003_nrs2_o005_crf.fits",
        "jw01249005001_03101_00004_nrs1_o005_crf.fits",
        "jw01249005001_03101_00004_nrs2_o005_crf.fits",
        "jw01249-o005_t001_nirspec_g395h-f290lp_s3d.fits",
        "jw01249-o005_t001_nirspec_g395h-f290lp_x1d.fits",
    ],
)
def test_spec3_multi(run_spec3_multi, fitsdiff_default_kwargs, output):
    """Regression test matching output files"""
    rt.is_like_truth(
        run_spec3_multi,
        fitsdiff_default_kwargs,
        output,
        truth_path="truth/test_nirspec_ifu",
        is_suffix=False,
    )
