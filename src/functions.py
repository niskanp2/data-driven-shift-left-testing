import torch
import itertools
import numpy as np
import pandas as pd
import json


phase_data = {}
with open("src/phases.txt", "r") as f:
    for line in f.readlines():
        strings = line.strip().split(" ")
        phase_data[strings[0]] = [node for node in strings[1:]]


def unpack_path(path):
    unpacked_path = []

    for step in path:
        step_list = []
        components = len(step.split("_"))
        if components == 1:
            phase = step
            teams = phase_data[phase]
            for team in teams:
                step_list.append(f"{phase}_{team}")
        elif components == 2:
            if step == "Verification_IV":
                phase = "Verification"
                teams = [team for team in phase_data["Verification"] if "IV" in team]
                for team in teams:
                    step_list.append(f"{phase}_{team}")
            else:
                step_list.append(step)
        elif components == 3:
            if "/" in step:
                phase = "Verification_IV"
                teams = step.split("_")[-1].split("/")
                for team in teams:
                    step_list.append(f"{phase}_{team}")
            else:
                step_list.append(step)

        unpacked_path.append(step_list)

    return unpacked_path


def get_path_permutations(path_template):
    """Returns all permutations of a path template."""
    if isinstance(path_template[0], str):
        path_template = unpack_path(path_template)
    return list(itertools.product(*path_template))


def get_path_probability(path, G):
    prev_node = "SPAWN"
    total_logprob = 0
    for node_i in range(1, len(path)):
        current_node = path[node_i]
        prob = G[prev_node][current_node]["prob"]
        logprob = np.log(prob)
        total_logprob += logprob
        prev_node = current_node
    return np.exp(total_logprob)


def expand_template(path_template):
    concrete_paths = []
    for combo in itertools.product(*path_template):
        full_path = ("SPAWN",) + combo + ("DETECTION",)
        concrete_paths.append(full_path)
    return concrete_paths


def get_total_variation_distance(sample_pdf, true_pdf):
    sample_pdf = np.array(sample_pdf)
    true_pdf = np.array(true_pdf)
    assert len(true_pdf) == len(sample_pdf), "The lengths of the pdfs must be the same."
    assert 0.99 < np.sum(sample_pdf) < 1.01, (
        "First input must be a probability density function."
    )
    assert 0.99 < np.sum(true_pdf) < 1.01, (
        "Second input must be a probability density function."
    )
    return 0.5 * np.sum(np.abs(true_pdf - sample_pdf))


def get_chi_squared_distance(sample_pdf, true_pdf):
    sample_pdf = np.array(sample_pdf)
    true_pdf = np.array(true_pdf)
    assert len(true_pdf) == len(sample_pdf), "The lengths of the pdfs must be the same."
    assert 0.99 < np.sum(sample_pdf) < 1.01, (
        "First input must be a probability density function."
    )
    assert 0.99 < np.sum(true_pdf) < 1.01, (
        "Second input must be a probability density function."
    )
    return np.sum(np.square(sample_pdf - true_pdf) / true_pdf)


def get_kullback_leibler_divergence(sample_pdf, true_pdf):
    sample_pdf = np.array(sample_pdf)
    true_pdf = np.array(true_pdf)
    assert len(true_pdf) == len(sample_pdf), "The lengths of the pdfs must be the same."
    assert 0.99 < np.sum(sample_pdf) < 1.01, (
        "First input must be a probability density function."
    )
    assert 0.99 < np.sum(true_pdf) < 1.01, (
        "Second input must be a probability density function."
    )
    mask = sample_pdf > 0
    return np.sum(sample_pdf[mask] * np.log(sample_pdf[mask] / true_pdf[mask]))


#############


def severity_matches(report_severity, severity):
    if severity is None:
        return True
    severity_list = [severity] if isinstance(severity, str) else severity
    return report_severity in severity_list


def get_report_timespent_from_task(issue_key, report_data, task_data, verbose=False):
    if verbose:
        print(f"Defect Report {issue_key}:")
    report_row = report_data[report_data["key"] == issue_key]

    linked_tasks = report_row["linked_tasks"].iloc[0]
    if verbose:
        print(f"{len(linked_tasks)} task(s) found.")
    if len(linked_tasks) == 0:
        return None

    task_timespent_list = []
    for task_key in linked_tasks:
        if verbose:
            print(f"Task {task_key}:")
        task_row = task_data[task_data["key"] == task_key]
        logged_timespent = task_row["timespent"].iloc[0]
        story_points = task_row["storypoints"].iloc[0]
        shared_reports = task_row["shared_reports"].iloc[0]

        if verbose:
            print(
                (
                    f"Logged timespent: {logged_timespent / 60 / 60:.2f}h\n"
                    f"Story points:     {story_points * 7.5:.2f}h"
                )
            )

        if isinstance(logged_timespent, int):
            logged_timespent /= shared_reports
            task_timespent_list.append(logged_timespent / (60 * 60))
        elif isinstance(story_points, float):
            story_points /= shared_reports
            task_timespent_list.append(story_points * 7.5)

    if len(task_timespent_list) == 0:
        return None

    timespent = sum(task_timespent_list)
    if verbose:
        print(f"Total: {timespent:.2f}h")
    return timespent


def get_report_timespent_from_report_changelog(
    issue_key, report_data, verbose=False, log_adjusted=False, threshold=15, scale=5
):
    if verbose:
        print(f"Defect Report {issue_key}:")
    report_changelog = report_data.loc[
        report_data["key"] == issue_key, "changelog"
    ].iloc[0]

    if verbose:
        print("Report timespent:")
    timespent = None

    if verbose:
        print(f"Total: {timespent:.2f}h")
    return timespent


def get_report_timespent_from_fix_changelog(
    issue_key,
    report_data,
    fix_data,
    log_adjusted=False,
    threshold=15,
    scale=5,
    verbose=False,
):
    if verbose:
        print(f"Defect Report {issue_key}:")

    linked_fixes = report_data.loc[
        report_data["key"] == issue_key, "linked_fixes"
    ].iloc[0]

    if verbose:
        print(f"{len(linked_fixes)} defect fix(es) found.")
    if len(linked_fixes) == 0:
        return None

    fix_timespent_list = []
    for fix_key in linked_fixes:
        if verbose:
            print(f"Defect Fix {fix_key}:")
        changelog = fix_data.loc[fix_data["key"] == fix_key, "changelog"].iloc[0]
        fix_timespent = None
        fix_timespent_list.append(fix_timespent)

    timespent = min(fix_timespent_list)
    if verbose:
        print(f"Total: {timespent:.2f}h")
    return timespent


def estimate_report_timespent(
    issue_key,
    report_data,
    fix_data,
    verbose=False,
    estimate_from=["Fix"],
    log_adjusted=False,
    threshold=15,
    scale=1,
):
    assert set(estimate_from) <= set(["Fix", "Report"]), (
        "Types must be 'Fix' and/or 'Report'"
    )
    timespent_estimate = 0
    if "Fix" in estimate_from:
        fix_estimate = get_report_timespent_from_fix_changelog(
            issue_key,
            report_data=report_data,
            fix_data=fix_data,
            verbose=verbose,
            log_adjusted=log_adjusted,
            threshold=threshold,
            scale=scale,
        )
        if fix_estimate is not None:
            timespent_estimate += fix_estimate
    if "Report" in estimate_from:
        report_estimate = get_report_timespent_from_report_changelog(
            issue_key,
            report_data=report_data,
            verbose=verbose,
            log_adjusted=log_adjusted,
            threshold=threshold,
            scale=scale,
        )
        if report_estimate is not None:
            timespent_estimate += report_estimate

    if timespent_estimate == 0:
        return None
    else:
        return timespent_estimate


def log_adjust_hours(hours, threshold=15, scale=5):
    if hours > threshold:
        return threshold + scale * np.log((hours - threshold) / scale + 1)
    else:
        return hours


def get_path(issue_key: str, report_data):
    start = report_data.loc[
        report_data["key"] == issue_key, "should_have_been_found"
    ].iloc[0]
    end = report_data.loc[report_data["key"] == issue_key, "detection_phase"].iloc[0]
    cause = report_data.loc[report_data["key"] == issue_key, "defect_root_cause"].iloc[
        0
    ]

    with open("src/path_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if start in mapping.keys() and end in mapping[start].keys():
        # Check if there's a specific mapping for defect root cause
        if (
            isinstance(mapping[start][end], dict)
            and cause in mapping[start][end].keys()
        ):
            return mapping[start][end][cause]
        # If not, use only start and end
        elif isinstance(mapping[start][end], list):
            return mapping[start][end]
        else:
            return None
