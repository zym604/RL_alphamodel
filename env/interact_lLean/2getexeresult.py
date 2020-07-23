import subprocess
path = r'E:\Lean-master\Launcher\bin\Debug\QuantConnect.Lean.Launcher.exe'
p1 = subprocess.Popen(path,cwd=r'E:\Lean-master\Launcher\bin\Debug',stdin=subprocess.PIPE,stdout=subprocess.PIPE, shell=True)
state = p1.communicate()[0]
print(state)