from __future__ import annotations

import argparse
import pickle as pkl
import time
from pathlib import Path

import lh5
import matplotlib as mpl
import numpy as np
import pygama.math.distributions as pmd  # noqa: F401
from dbetto.catalog import Props
from lgdo import Array, Table
from pygama.pargen.data_cleaning import generate_cuts
from pygama.pargen.dplms_ge_dict import dplms_ge_dict
from pygama.pargen.dsp_optimize import run_one_dsp

from legenddataflowscripts.utils import (
    build_log,
    check_input_files,
    convert_dict_np_to_float,
    expand_filelist,
    prepare_output_paths,
    require_config_keys,
    require_peaks_present,
    take_table_rows,
)

mpl.use("agg")  # no display needed; avoids X11 when DISPLAY is set


def _build_argparser(description: str) -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument(
        "--fft-raw-filelist", help="fft_raw_filelist", type=str
    )
    argparser.add_argument(
        "--peak-file", help="tcm_filelist", type=str, required=True
    )
    argparser.add_argument("--inplots", help="in_plot_path", type=str)
    argparser.add_argument(
        "--database", help="database", type=str, required=True
    )

    argparser.add_argument("--log", help="log_file", type=str)
    argparser.add_argument(
        "--log-config",
        help="Log config file",
        type=str,
        required=False,
        default={},
    )

    argparser.add_argument(
        "--processing-chain",
        help="Processing chain config",
        type=str,
        nargs="*",
        required=True,
    )
    argparser.add_argument(
        "--config-file", help="Config file", type=str, nargs="*", required=True
    )

    argparser.add_argument("--channel", help="channel", type=str, required=True)
    argparser.add_argument(
        "--raw-table-name", help="raw table name", type=str, required=True
    )

    argparser.add_argument(
        "--dsp-pars", help="dsp_pars", type=str, required=True
    )
    argparser.add_argument(
        "--lh5-path", help="lh5_path", type=str, required=True
    )
    argparser.add_argument("--plot-path", help="plot_path", type=str)

    return argparser


def _run_dplms_step(db_key: str, ene_par: str) -> None:
    """Shared DPLMS optimisation step.

    ``db_key`` names the DSP parameter block (and LH5 table) written by this
    run: ``dplms`` for the dplmsEmax optimisation, ``dplms_eftp`` for the
    dplmsEftp optimisation.  ``ene_par`` is the energy quantity the filter
    grid is optimised against (its FOM FWHM) and the source column of the
    ``<ene_par>_ctc`` correction that is written.

    The step writes:

    - the filter coefficients to the LH5 group
      ``{channel}/{db_key}/coefficients`` (so the DSP chain can load them
      with ``loadlh5(...)``);
    - the parameter block under ``db_key`` (length, coefficients pointer,
      dp_coeffs and, when a ``pickoffs`` grid was run, the best pickoff)
      plus the ``{ene_par}_ctc`` entry into the parameter database written
      to *dsp-pars*.

    Existing database keys (e.g. the ``dplms`` block from a preceding
    dplmsEmax step, or other ctc corrections) are preserved: ctc entries are
    merged rather than replaced.
    """
    argparser = _build_argparser(f"DPLMS optimisation ({db_key}/{ene_par})")
    args = argparser.parse_args()

    log = build_log(args.log_config, args.log)

    check_input_files(args.peak_file, "--peak-file")
    check_input_files(args.inplots, "--inplots")
    prepare_output_paths(args.lh5_path, args.dsp_pars, args.plot_path)

    dsp_config = Props.read_from(args.processing_chain)

    t0 = time.time()

    dplms_dict = Props.read_from(args.config_file)
    db_dict = Props.read_from(args.database)
    require_config_keys(
        dplms_dict, ["run_dplms"], f"dplms config ({args.config_file})"
    )

    if dplms_dict["run_dplms"] is True:
        fft_files = expand_filelist(
            [args.fft_raw_filelist], "--fft-raw-filelist"
        )

        t0 = time.time()
        log.info("\nLoad fft data")
        energies = lh5.read_as(
            f"{args.raw_table_name}/daqenergy", fft_files, library="np"
        )
        eidxs = np.where(energies <= 10)[0]
        raw_fft = lh5.read(
            args.raw_table_name,
            fft_files,
            n_rows=dplms_dict["n_baselines"],
            idx=eidxs,
        )
        t1 = time.time()
        msg = f"Time to load fft data {(t1 - t0):.2f} s, total events {len(raw_fft)}"
        log.info(msg)

        log.info("\nRunning event selection")
        peaks_kev = np.array(dplms_dict["peaks_kev"])

        peaks_rounded = [int(peak) for peak in peaks_kev]
        peaks = lh5.read_as(
            f"{args.raw_table_name}/peak", args.peak_file, library="np"
        )
        require_peaks_present(
            np.unique(peaks),
            peaks_rounded,
            f"peak file {args.peak_file} requested by dplms config ({args.config_file})",
        )
        ids = np.isin(peaks, peaks_rounded)

        raw_cal = lh5.read(args.raw_table_name, args.peak_file, idx=ids)
        msg = f"Time to run event selection {(time.time() - t1):.2f} s, total events {len(raw_cal)}"
        log.info(msg)
        del peaks, ids

        if isinstance(dsp_config, str | list):
            dsp_config = Props.read_from(dsp_config)

        dsp_fft = run_one_dsp(raw_fft, dsp_config, db_dict=db_dict)

        cut_dict = generate_cuts(dsp_fft, cut_dict=dplms_dict["bls_cut_pars"])
        log.debug("Cuts are %s", cut_dict)
        idxs = np.full(len(dsp_fft), True, dtype=bool)
        for outname, info in cut_dict.items():
            outcol = dsp_fft.eval(
                info["expression"], info.get("parameters", None)
            )
            dsp_fft.add_column(outname, outcol)
        for cut in cut_dict:
            idxs = dsp_fft[cut].nda & idxs

        # idxs is a boolean mask over the rows already loaded in raw_fft, so
        # subset in memory instead of re-reading the FFT files from disk
        # (row-identical: the first read was idx=eidxs capped at n_baselines,
        # so the re-applied n_rows cap was a no-op)
        raw_fft = take_table_rows(raw_fft, idxs)
        del dsp_fft, energies, eidxs, cut_dict, idxs

        log.debug("Applied Cuts")

        out_dict, plot_dict = dplms_ge_dict(
            raw_fft,
            raw_cal,
            dsp_config,
            db_dict,
            dplms_dict,
            fom_func=eval(dplms_dict.get("fom_func", "pmd.gauss_on_step")),
            ene_par=ene_par,
            db_key=db_key,
            display=1 if args.plot_path else 0,
        )
        if args.plot_path:
            if args.inplots:
                with Path(args.inplots).open("rb") as r:
                    inplot_dict = pkl.load(r)
                inplot_dict.update({db_key: plot_dict})
            else:
                inplot_dict = {db_key: plot_dict}

        coeffs = out_dict[db_key].pop("coefficients")
        dplms_pars = Table(col_dict={"coefficients": Array(coeffs)})
        out_dict[db_key]["coefficients"] = (
            f"loadlh5('{args.lh5_path}', '{args.channel}/{db_key}/coefficients')"
        )
        msg = f"DPLMS ({db_key}) creation finished in {(time.time() - t0) / 60} minutes"
        log.info(msg)
    else:
        out_dict = {}
        dplms_pars = Table(col_dict={"coefficients": Array([])})
        if args.plot_path and args.inplots:
            with Path(args.inplots).open("rb") as r:
                inplot_dict = pkl.load(r)
        else:
            inplot_dict = {}

    # merge ctc corrections instead of replacing: a preceding step (e.g. the
    # dplmsEmax run) may already have written its own ctc entries to the db
    if isinstance(out_dict.get("ctc_params"), dict) and isinstance(
        db_dict.get("ctc_params"), dict
    ):
        db_dict["ctc_params"].update(out_dict.pop("ctc_params"))
    db_dict.update(out_dict)

    # one LH5 file per channel, one table (group) per optimisation run:
    # the dplms_eftp step appends {channel}/dplms_eftp/coefficients to the
    # same file that already holds {channel}/dplms/coefficients
    lh5.write(
        Table(col_dict={db_key: dplms_pars}),
        name=args.channel,
        lh5_file=args.lh5_path,
        wo_mode="overwrite",
    )
    Props.write_to(args.dsp_pars, convert_dict_np_to_float(db_dict))

    if args.plot_path:
        with Path(args.plot_path).open("wb") as f:
            pkl.dump(inplot_dict, f, protocol=pkl.HIGHEST_PROTOCOL)


def par_geds_dsp_dplms() -> None:
    """Compute DPLMS optimal filter coefficients for HPGe detectors.

    CLI entry point registered as ``par-geds-dsp-dplms``.  The *Discrete
    Prolate Laplacian Maximum Signal* (DPLMS) filter is constructed from two
    data samples:

    1. **FFT baselines** - low-energy (``daqenergy <= 10`` ADC) events from
       dedicated FFT run files used to characterise the noise power spectrum.
    2. **Calibration peak events** - gamma-line events read from the
       *peak-file* produced by :func:`par_geds_dsp_evtsel`.

    The filter grid is optimised against the ``dplmsEmax`` quantity.  The
    coefficients are computed by
    :func:`pygama.pargen.dplms_ge_dict.dplms_ge_dict` and written to the LH5
    group ``{channel}/dplms/coefficients`` in *lh5-path* (so that the DSP
    chain can load them with ``loadlh5(...)``).  The DSP parameter database
    is updated with the filter configuration under the ``dplms`` key and the
    ``dplmsEmax_ctc`` correction, and written to *dsp-pars*.

    Notes
    -----
    **Command-line arguments**

    ``--fft-raw-filelist`` : str
        Path to a text file listing the FFT raw LH5 input files.
    ``--peak-file`` : str
        LH5 file containing preselected calibration peak events.
    ``--inplots`` : str, optional
        Existing pickle plot file to update with DPLMS plots.
    ``--database`` : str
        Path to the existing DSP parameter database (JSON/YAML).
    ``--log`` : str, optional
        Path to the log file.
    ``--log-config`` : str, optional
        Logging configuration file.
    ``--processing-chain`` : list of str
        Processing chain configuration file(s).
    ``--config-file`` : list of str
        DPLMS configuration file(s).  Must contain ``run_dplms`` (bool),
        ``n_baselines`` (int), and ``peaks_kev`` (list).  The optional key
        ``use_log_pdf`` (bool, default false) is forwarded to
        :func:`pygama.pargen.dplms_ge_dict.dplms_ge_dict` and builds the
        grid-search FOM fits from the model's log-density (iminuit
        ``log=True``) — substantially faster, results differ at
        machine-precision level; requires a pygama version with
        ``use_log_pdf`` support.
    ``--channel`` : str
        Channel identifier; used as the HDF5 group name in *lh5-path*.
    ``--raw-table-name`` : str
        LH5 table path within the raw and peak files.
    ``--dsp-pars`` : str
        Output path for the updated DSP parameter database (JSON/YAML).
    ``--lh5-path`` : str
        Output LH5 file path for the DPLMS filter coefficients.
    ``--plot-path`` : str, optional
        Output path for diagnostic plots (pickle).
    """
    _run_dplms_step(db_key="dplms", ene_par="dplmsEmax")


def par_geds_dsp_dplms_eftp() -> None:
    """Compute a DPLMS filter optimised against ``dplmsEftp``.

    CLI entry point registered as ``par-geds-dsp-dplms-eftp``.  Runs the
    same grid optimisation as :func:`par_geds_dsp_dplms` but minimises the
    FWHM of the ``dplmsEftp`` quantity (fixed-time pickoff readout) instead
    of ``dplmsEmax``.

    Results are written under the separate ``dplms_eftp`` namespace so that
    the regular ``dplms`` block is never overwritten:

    - LH5 group ``{channel}/dplms_eftp/coefficients`` in the *same* LH5 file
      as the dplmsEmax coefficients (pass the dplms step's *lh5-path* as
      ``--lh5-path``);
    - parameter block ``dplms_eftp`` (length, coefficients pointer,
      dp_coeffs and, when the config contains a ``pickoffs`` grid, the best
      fixed-time pickoff) plus the ``dplmsEftp_ctc`` correction in *dsp-pars*.

    The command-line arguments are identical to
    :func:`par_geds_dsp_dplms`.
    """
    _run_dplms_step(db_key="dplms_eftp", ene_par="dplmsEftp")
