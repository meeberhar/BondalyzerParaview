import sys
print("Python version:", sys.version, flush=True)
print("sys.executable:", sys.executable, flush=True)
print("sys.path:", sys.path, flush=True)
import paraview
print("paraview loaded", flush=True)
import paraview.simple as pvs
print("ParaView version:", paraview.__version__, flush=True)
