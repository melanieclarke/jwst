#!/usr/bin/env python
import logging

from stdatamodels.jwst import datamodels

from jwst.charge_migration import charge_migration_step
from jwst.clean_flicker_noise import clean_flicker_noise_step
from jwst.dark_current import dark_current_step
from jwst.dq_init import dq_init_step
from jwst.emicorr import emicorr_step
from jwst.firstframe import firstframe_step
from jwst.gain_scale import gain_scale_step
from jwst.group_scale import group_scale_step
from jwst.ipc import ipc_step
from jwst.jump import jump_step
from jwst.lastframe import lastframe_step
from jwst.linearity import linearity_step
from jwst.persistence import persistence_step
from jwst.ramp_fitting import ramp_fit_step
from jwst.refpix import refpix_step
from jwst.reset import reset_step
from jwst.rscd import rscd_step
from jwst.saturation import saturation_step
from jwst.stpipe import Pipeline
from jwst.stpipe.utilities import record_step_status
from jwst.superbias import superbias_step

__all__ = ["Detector1Pipeline"]

# Define logging
log = logging.getLogger(__name__)


class Detector1Pipeline(Pipeline):
    """
    Apply all calibration steps to raw JWST ramps to produce a 2-D slope product.

    Included steps are:
    group_scale, dq_init, emicorr, saturation, ipc, superbias, refpix, rscd,
    firstframe, lastframe, linearity, dark_current, reset, persistence,
    charge_migration, jump detection, clean_flicker_noise, ramp_fit,
    and gain_scale.
    """

    class_alias = "calwebb_detector1"

    spec = """
        save_calibrated_ramp = boolean(default=False)
    """  # noqa: E501

    # Define aliases to steps
    step_defs = {
        "group_scale": group_scale_step.GroupScaleStep,
        "dq_init": dq_init_step.DQInitStep,
        "emicorr": emicorr_step.EmiCorrStep,
        "saturation": saturation_step.SaturationStep,
        "ipc": ipc_step.IPCStep,
        "superbias": superbias_step.SuperBiasStep,
        "refpix": refpix_step.RefPixStep,
        "rscd": rscd_step.RscdStep,
        "firstframe": firstframe_step.FirstFrameStep,
        "lastframe": lastframe_step.LastFrameStep,
        "linearity": linearity_step.LinearityStep,
        "dark_current": dark_current_step.DarkCurrentStep,
        "reset": reset_step.ResetStep,
        "persistence": persistence_step.PersistenceStep,
        "charge_migration": charge_migration_step.ChargeMigrationStep,
        "jump": jump_step.JumpStep,
        "clean_flicker_noise": clean_flicker_noise_step.CleanFlickerNoiseStep,
        "ramp_fit": ramp_fit_step.RampFitStep,
        "gain_scale": gain_scale_step.GainScaleStep,
    }

    # start the actual processing
    def process(self, input_data):
        """
        Run the Detector1 pipeline on the input data.

        Parameters
        ----------
        input_data : str or `~jwst.datamodels.RampModel`
            The input data to process.

        Returns
        -------
        `~jwst.datamodels.JwstDataModel`
            The calibrated data model.
        """
        log.info("Starting calwebb_detector1 ...")

        # open the input data as a RampModel
        input_data = datamodels.RampModel(input_data)

        # propagate output_dir to steps that might need it
        self.dark_current.output_dir = self.output_dir
        self.ramp_fit.output_dir = self.output_dir

        instrument = input_data.meta.instrument.name
        if instrument == "MIRI":
            # process MIRI exposures;
            # the steps are in a different order than NIR
            log.debug("Processing a MIRI exposure")
            cal_steps = [
                "group_scale",
                "dq_init",
                "emicorr",
                "saturation",
                "ipc",
                "firstframe",
                "lastframe",
                "reset",
                "linearity",
                "rscd",
                "dark_current",
                "refpix",
            ]
        else:
            # process Near-IR exposures
            log.debug("Processing a Near-IR exposure")
            cal_steps = [
                "group_scale",
                "dq_init",
                "saturation",
                "ipc",
                "superbias",
                "refpix",
                "linearity",
            ]

            # skip persistence for NIRSpec
            if instrument != "NIRSPEC":
                cal_steps.append("persistence")

            # run dark_current for all NIR
            cal_steps.append("dark_current")

        # Add a few steps for all instruments
        cal_steps.extend(["charge_migration", "jump", "clean_flicker_noise"])

        # Run all the steps so far.
        # To save memory, don't attempt to run steps marked "skip".
        for cal_step in cal_steps:
            step = getattr(self, cal_step)
            if step.skip:
                # Just record the status as "SKIPPED"
                if cal_step == "dark_current":
                    # Exception: dark_current is recorded as "dark_sub"
                    record_step_status(input_data, "dark_sub", False)
                else:
                    record_step_status(input_data, cal_step, False)
            else:
                # Run the step
                input_data = step.run(input_data)

        # save the corrected ramp data, if requested
        if self.save_calibrated_ramp:
            self.save_model(input_data, "ramp")

        # Apply the ramp_fit step.
        # If skipped, set ints_model to None, since it is not created.
        if self.ramp_fit.skip:
            record_step_status(input_data, "ramp_fit", False)
            ints_model = None
        else:
            input_data, ints_model = self.ramp_fit.run(input_data)

        # apply the gain_scale step to the exposure-level product
        if input_data is not None:
            if self.gain_scale.skip:
                record_step_status(input_data, "gain_scale", False)
            else:
                self.gain_scale.suffix = "gain_scale"
                input_data = self.gain_scale.run(input_data)
        else:
            log.info("NoneType returned from ramp_fit.  Gain Scale step skipped.")

        # apply the gain scale step to the multi-integration product,
        # if it exists, and then save it
        if ints_model is not None:
            if self.gain_scale.skip:
                record_step_status(ints_model, "gain_scale", False)
            else:
                self.gain_scale.suffix = "gain_scaleints"
                ints_model = self.gain_scale.run(ints_model)
            self.save_model(ints_model, "rateints")

        # setup output_file for saving
        self.setup_output(input_data)

        log.info("... ending calwebb_detector1")

        return input_data

    def setup_output(self, input_data):
        """
        Set up the output file suffix based on which steps were run successfully.

        Parameters
        ----------
        input_data : `~jwst.datamodels.JwstDataModel`
            The output data product from the Detector1 pipeline
        """
        if input_data is None:
            return
        # Determine the proper file name suffix to use later
        if input_data.meta.cal_step.ramp_fit == "COMPLETE":
            self.suffix = "rate"
        else:
            self.suffix = "ramp"
