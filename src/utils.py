import collections


def args2cmd(args, exec_name, hpc=False):
	"""
	outputs command string with arguments in args
	:param args: arguments dictionary
	:param exec_name: string with the name of python script
	:return: string with command
	"""
	if hpc:
		out = "python3 " + exec_name
	else:
		out = "python " + exec_name
	for k, v in args.items():
		out += " "
		out += "--{}={}".format(k, v)
	return out


def dict2csv(args, csv_name):
	"""
	writes dictionary in csv format to the csv_name file
	:param args: in_dictionary
	:param csv_name: out file name
	:return:
	"""
	with open(csv_name, "w") as f:
		f.write(",".join(args.keys()) + "\n")
		f.write(",".join(str(x) for x in args.values()) + "\n")


class ReadOnlyWrapper(collections.Mapping):
	"""
	To make dicts read only (stackoverflow).

	"""

	def __init__(self, data):
		self._data = data

	def __getitem__(self, key):
		return self._data[key]

	def __len__(self):
		return len(self._data)

	def __iter__(self):
		return iter(self._data)

	def __str__(self):
		return str(self._data)

	def get_copy(self):
		return self._data.copy()


def make_dict_read_only(dict):
	"""
	Make a dictionary into a new read only dictionary.
	:param dict:
	:return:
	"""
	return ReadOnlyWrapper(dict)
