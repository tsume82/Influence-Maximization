
def args2cmd(args, exec_name):
	"""
	outputs command string with arguments in args
	:param args: arguments dictionary
	:param exec_name: string with the name of python script
	:return: string with command
	"""
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
