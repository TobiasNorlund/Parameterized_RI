import sys
import numpy as np

# Which embeddings, models and datasets are available?
available_emb = ("BOW", "RI", "SGD_RI", "ATT_RI", "PMI_RI", "W2V", "PMI")
available_mdl = ("MLP", "CNN")
available_ds  = ("PL05", "SST")

if len(sys.argv) != 4:
    print "ERROR: Invalid number of arguments, format is: EMBEDDING1,EMBEDDING2,... MODEL1,MODEL2 DATASET(S)"
    sys.exit()

embeddings = sys.argv[1]
if embeddings == "-":
    embeddings = available_emb
else:
    embeddings = embeddings.split(",")
    
models = sys.argv[2]
if models == "-":
    models = available_mdl
else:
    models = models.split(",")
     
dss = sys.argv[3]
if dss == "-":
    dss = available_ds
else:
    dss = dss.split(",")   

f_res = open("batch_results.txt", "w")
results = []

for emb in embeddings:
  for mdl in models:
    for ds in dss:
      res = {}
      sys.argv = ["run.py", emb, mdl, ds]
      execfile("run.py", res)
      
      results.append((emb + " " + mdl + " " + ds, np.mean(res["accuracies"])*100, np.std(res["accuracies"])*100))

      # Write result to file
      f_res.write("{}: {:.2f}% +- {:.2f}%\n".format(emb + " " + mdl + " " + ds, np.mean(res["accuracies"])*100, np.std(res["accuracies"])*100))
      f_res.flush()
      
f_res.close()

# Print results:
print ""
print " RESULTS:"
print ""

for experiment, acc, std in results:

    print "{}: {:.2f}% +- {:.2f}%".format(experiment, acc, std)
      