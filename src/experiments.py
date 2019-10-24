"""
Given an experiment directory path <exp_dir> executes the experiments with parameters indicated in the in <exp_dir>/in
and stores the results in the <exp_dir>/out;
each input file in <exp_dir>/in should be in json format and contain all the input parameters for the script and the
number of times the experiment should run
"""
import argparse
import os
import json
import subprocess

from utils import args2cmd


def current_commit_revision():
	"""
	useful in case checkout to the working version is needed,
	use 'git checkout <sha1>' to checkout to the needed version
	:return:
	"""
	git_revision_sha1_short = subprocess.check_output("git rev-parse --short HEAD".split())
	return git_revision_sha1_short.decode("utf-8")


def run_experiment(in_file):
	"""
	runs experiment according to in_file parameters
	:param in_file: json file containing fields 'script' with the script name,
					'script args' with the dictionary of script arguments and
					'n_repetitions' with the number of times the experiment
					should run
	:return:
	"""
	data = dict()
	with open(in_file, "r") as f:
		data = json.load(f)

	cmd = args2cmd(args=data["script_args"], exec_name=data["script"])
	for _ in range(data["n_repetitions"]):
		subprocess.call(cmd.split())


if __name__ == "__main__":
	# reading arguments

	parser = argparse.ArgumentParser(description='Experiments run')
	parser.add_argument('--exp_dir', help="experiment directory")

	args = parser.parse_args()

	in_directory = args.exp_dir
	if not ("/in" in in_directory):
		in_directory += "/in"
	out_directory = in_directory.replace("/in", "/out")

	if not os.path.exists(out_directory):
		os.makedirs(out_directory)

	# write the current commit revision
	sha1 = current_commit_revision()
	f = open(out_directory + "/commit_revisions", 'a+')
	f.seek(0)
	lines = f.readlines()
	# check if the last commit revision is the one we are still using
	if len(lines) == 0 or (sha1 not in lines[-1]):
		f.write(sha1)
	f.close()

	for sub_dir, _, files in os.walk(in_directory):
		out_sub_dir = out_directory + sub_dir.split(in_directory)[1]
		if not os.path.exists(out_sub_dir):
			os.makedirs(out_sub_dir)
		for file in files:
			if file.lower().endswith(".json"):
				run_experiment(sub_dir + "/" + file)

