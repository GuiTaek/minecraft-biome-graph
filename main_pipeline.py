import os
import json
import re
import shlex
import subprocess
from subprocess import PIPE
import shutil

from analyse import main_analyse

MY_FIRST_COMMIT_HASH = "0442958"
MCMETA_FIRST_COMMIT_HASH = "156e3801"
MCMETA_IGNORE_REGEX = re.compile("🚀 Update data-json for (.*)")
GET_COMMIT_MESSAGES_COMMAND = "git log --oneline --pretty=format:\"%s\" {}..HEAD"
GET_COMMIT_WITH_HASH_COMMAND = "git log --oneline --pretty=format:\"%h\" {}..HEAD"
CHECKOUT_COMMIT_HASH = "git checkout {}"
COMMIT = "git commit --allow-empty -m \"{}\""
ADD_ALL = "git add ."
OVERWORLD_PATH = "mcmeta/data/minecraft/dimension/overworld.json"
RUN_SCRIPT = "python analyse.py"
FILES_TO_MOVE = [
    "probabilities.json",
    "overworld_graph_result_0.20_quantile.graphml",
    "overworld_graph_result_0.35_quantile.graphml",
    "overworld_graph_result_0.50_quantile.graphml",
    "overworld_graph_result_full.graphml"
]

def fail():
    with open("FAILED", mode="w") as f:
        pass

def get_git_with_command(path, command):
    cwd = os.getcwd()
    os.chdir(path)
    # shlex.split is save, because no user input is given (in fact, the whole pipeline has no user input)
    process = subprocess.Popen(shlex.split(command), stdout=PIPE, stderr=PIPE, encoding="utf-8")
    stdout, stderr = process.communicate()
    os.chdir(cwd)
    if stderr:
        print(stderr)
        raise RuntimeError("unkown stderr output")
    return stdout.strip().split("\n")

def get_commit_messages(first_hash, path):
    command = GET_COMMIT_MESSAGES_COMMAND.format(first_hash)
    return get_git_with_command(path, command)

def get_commit_hashes(first_hash, path):
    command = GET_COMMIT_WITH_HASH_COMMAND.format(first_hash)
    return get_git_with_command(path, command)

def get_own_commits():
    finished_versions = get_commit_messages(MY_FIRST_COMMIT_HASH, "results")
    return finished_versions

def get_other_commits():
    commit_messages = get_commit_messages(MCMETA_FIRST_COMMIT_HASH, "mcmeta")

    ready_versions = [
            match_res.group(1)
            for commit_message in commit_messages
            for match_res in [MCMETA_IGNORE_REGEX.match(commit_message)]
            if match_res
    ]
    commit_hashes = get_commit_hashes(MCMETA_FIRST_COMMIT_HASH, "mcmeta")
    return list(reversed(list(zip(commit_hashes, ready_versions))))

def get_versions_to_process(own_commits, other_commits):
    own_commits = set(own_commits)
    res = []
    for commit_hash, version in other_commits:
        if version in own_commits:
            continue
        res.append((commit_hash, version))
    return res

def copy_overworld(commit_hash, copy_to_path):
    os.chdir("../mcmeta")

    # shell is no problem because no user input
    subprocess.Popen(CHECKOUT_COMMIT_HASH.format(commit_hash), shell=True).wait()
    os.chdir("..")
    shutil.copyfile(OVERWORLD_PATH, copy_to_path)
    os.chdir("scripts")

def process_versions(versions_to_process):
    for commit_hash, version in versions_to_process:
        print(f"procesing {version}")
        copy_overworld(commit_hash, "scripts/overworld.json")
        main_analyse()
        for filename in FILES_TO_MOVE:
            shutil.copyfile(f"{filename}", f"../results/{filename}")
        os.chdir("../results")
        # shell is no problem because no user input
        subprocess.Popen(ADD_ALL, shell=True).wait()
        subprocess.Popen(COMMIT.format(version), shell=True).wait()
        os.chdir("../scripts")
        

def main():
    os.chdir("..")
    own_commits = get_own_commits()
    other_commits = get_other_commits()
    versions_to_process = get_versions_to_process(own_commits, other_commits)
    os.chdir("scripts")
    process_versions(versions_to_process)


if __name__ == "__main__":
    main()