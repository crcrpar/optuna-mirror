# You can set these variables from the command line.
PYTHON = python

protobuf:
	$(PYTHON) -m grpc_tools.protoc -I./ --python_out=../ --grpc_python_out=../ `find . -type d -name 'lib' -prune  -o -name '*.proto' -print`
	sed -i '1s/^/# flake8: NOQA\n/' `find ../ -name '*pb2*.py'`
	autopep8 .. -r --in-place
