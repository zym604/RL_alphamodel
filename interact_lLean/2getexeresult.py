from subprocess import check_output as qx

#cmd = r'E:\Lean-master\Launcher\bin\Debug\QuantConnect.Lean.Launcher.exe'
#output = qx(cmd)
#print(output)
#用不起来，还是让lean每次自己生成一个result.txt小文件，然后读取吧
a = qx(['ls', '-l'])
print(a)