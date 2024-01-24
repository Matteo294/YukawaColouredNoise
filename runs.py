import os
import toml
import sys
import fileinput
import subprocess

cluster = sys.argv[1]



configurations = []


'''yukawas = [0.0 + 0.2 * n for n in range(10)]
cutoffs = [1.0, 1/2, 1/4]
for s in cutoffs:
    for g in yukawas:
        configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 0.4*s*s, "kappa": 0.18, "lambda": 0.02, "cutFraction": s}, \
                            "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 200, "MeasureDriftCount": 60}, \
                            "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                            "random": {"seed": 1432}, \
                            "fermions": {"yukawa_coupling": g, "fermion_mass": 1.0, "driftMode": 1, "WilsonParam": 1.0}, \
                            "lattice": {"Nt": int(16/s), "Nx": int(16/s)} })'''


#sqmasses = []

NTs = range(2, 25, 4)
s = 0.5
for t in NTs:
        configurations.append({ "physics": {"useMass": "true", "mass": -0.5*s*s, "g": 1.582*s*s, "kappa": 0.275, "lambda": 0.02, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.005, "MaxLangevinTime": 50000.0, "ExportTime": 1.0, "burnCount": 100, "MeasureDriftCount": 100}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 2904}, \
                        "fermions": {"yukawa_coupling": 0.07, "fermion_mass": 0.01, "driftMode": 1, "WilsonParam": 0.0}, \
                        "lattice": {"Nt": int(t), "Nx": int(16/s)} })


n_old_confs = max([int(d.replace("conf", "")) for d in os.listdir("./") if "conf" in d], default=0)
print("tot old configurations:", n_old_confs)



for count, conf in enumerate(configurations):

    count += n_old_confs
    
    
    print()
    print("=======================================================")
    print("Configuration", count + 1)
    
    dirname = "conf" + str(count + 1)
    
    # Create folde for this configuration
    process = subprocess.Popen("mkdir " + dirname, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    # Copy files into new folder
    process = subprocess.Popen("cp -r Yukawa_theory/*" + " " + dirname + "/", shell=True, stdout=subprocess.PIPE)
    process.wait()
    
     # Enter directory for this configuration
    os.chdir(dirname)
    
    # Load and modify toml params
    data = toml.load("./input.toml") 
    for section in conf.keys():
        if section in data.keys():
            print()
            for param in conf[section].keys():
                if param in data[section].keys():
                    print(section, param, conf[section][param])
                    data[section][param] = conf[section][param]
                    f = open("./input.toml",'w')
                    toml.dump(data, f)
                    f.close()
        if param == "Nt" or param == "Nx":
             print(section, param, conf[section][param])
    
     # Edit params.h file
    if conf["lattice"]["Nt"] != 0 and conf["lattice"]["Nx"] != 0:
        for line in fileinput.input("src/params.h", inplace=1):
            if "dimArray constexpr Sizes =" in line:
                line = line.replace("16, 16", str(conf["lattice"]["Nt"]) + ", " + str(conf["lattice"]["Nx"]))
            sys.stdout.write(line)
        process = subprocess.Popen("make clean && make -j", shell=True, stdout=subprocess.PIPE)
        process.wait()
    
    if cluster == "itp":
        filename = "runitp.sh"
        for line in fileinput.input(filename, inplace=1):
            if "cd" in line:
                line = line.replace("cd QuarkMesonModel/Yukawa_theory", "cd QuarkMesonModel/" + dirname)
            sys.stdout.write(line)
        process = subprocess.Popen("qsub runitp.sh", shell=True, stdout=subprocess.PIPE)
        process.wait()
    elif cluster == "bw":
        pass
        process = subprocess.Popen("sbatch runbw.sh", shell=True, stdout=subprocess.PIPE)
        process.wait()
        
    os.chdir("../")
    print()
    print("Done!")
    print("=======================================================")
    print()
    print()
