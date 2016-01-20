import sys
import datetime
import numpy as np

# Which embeddings, models and datasets are available?
available_emb = ("BOW", "RI", "SGD_RI", "ATT_RI", "PMI_RI", "W2V", "PMI")
available_mdl = ("MLP", "CNN")
available_ds  = ("PL05", "SST")

replications = 1

if len(sys.argv) != 4 and not(len(sys.argv) == 5 and str.isdigit(sys.argv[4])):
    print "ERROR: Invalid number of arguments, format is: EMBEDDING1,EMBEDDING2,... MODEL1,MODEL2 DATASET(S) [replications=1]"
    sys.exit()

# How many replications should we run?
if len(sys.argv) == 5:
    replications = int(sys.argv[4])

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

f_res = open("batch_results_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + ".txt", "w")
results = []

for emb in embeddings:
  for mdl in models:
    for ds in dss:
      
      res = {}
      sys.argv = ["run.py", emb, mdl, ds]
      aggr_acc = []
      try:
          for rep in range(replications):  
            execfile("run.py", res)
            aggr_acc += res["accuracies"]
          
          results.append((emb + " " + mdl + " " + ds, aggr_acc))
  
          # Write result to file
          f_res.write("{}: {:.2f}% +- {:.2f}% \t{}\n".format(emb + " " + mdl + " " + ds, 100*np.mean(aggr_acc), 100*np.std(aggr_acc), aggr_acc))
          f_res.flush()
      except:
          print "Error running:" + emb + " " + mdl + " " + ds
      
f_res.close()

# Print results:
print ""
print " RESULTS:"
print ""

for experiment, aggr_acc in results:
    print "{}: {:.2f}% +- {:.2f}%".format(experiment, 100*np.mean(aggr_acc), 100*np.std(aggr_acc))
      