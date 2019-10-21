
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