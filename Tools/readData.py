import pandas as pd
import os
try:
    import uproot3 as uproot
except ImportError:
    import uproot
from dask import delayed
import dask.dataframe as dd
import gc

def daskframe_from_rootfiles(processes, treepath,branches,flatten='False',debug=False):
    def get_df(Class,file, xsecwt, selection, treepath=None,branches=['ele*'],multfactor=1):
        tree = uproot.open(file)[treepath]
        if debug:
            ddd=tree.pandas.df(branches=branches,flatten=flatten,entrystop=1000).query(selection)
        else:
            ddd=tree.pandas.df(branches=branches,flatten=flatten).query(selection)
        #ddd["Category"]=Category
        ddd["Class"]=Class
        if type(xsecwt) == type(('xsec',2)):
            ddd["xsecwt"]=ddd[xsecwt[0]]*xsecwt[1]
        elif type(xsecwt) == type("hello"):
            ddd["xsecwt"]=ddd[xsecwt]
        elif type(xsecwt) == type(0.1):
            ddd["xsecwt"]=xsecwt
        elif type(xsecwt) == type(1):
            ddd["xsecwt"]=xsecwt
        else:
            print("CAUTION: xsecwt should be a branch name or a number or a tuple... Assigning the weight as 1")        
        print(file)
        return ddd

    dfs=[]
    for process in processes:
        
        #print(type(process['path']))
        #print(isinstance(process['path'], tuple))
        #print(len(process['path']))
        if isinstance(process['path'], list):
            if isinstance(process['xsecwt'], list):
                for onefile,onexsecwt in zip(process['path'],process['xsecwt']):
                    dfs.append(delayed(get_df)(process['Class'],onefile,xsecwt,process['selection'],treepath, branches))
            else:            
                for onefile in process['path']:
                    dfs.append(delayed(get_df)(process['Class'],onefile,process['xsecwt'],process['selection'],treepath, branches))
        elif isinstance(process['path'], tuple) and len(process['path'])==2:
            listoffiles=[process['path'][0]+'/'+f for f in os.listdir(process['path'][0]) if f.endswith(process['path'][1])]
            print(listoffiles)
            for onefile in listoffiles:
                dfs.append(delayed(get_df)(process['Class'],onefile,process['xsecwt'],process['selection'],treepath, branches))
        elif isinstance(process['path'], str):
            dfs.append(delayed(get_df)(process['Class'],process['path'],process['xsecwt'],process['selection'],treepath, branches))
        else:
            print("There is some problem with process path specification. Only string, list or tuple allowed")
    print("Creating dask graph!")
    print("Testing single file first")
    daskframe = dd.from_delayed(dfs)
    print("Finally, getting data from")
    dddf_final=daskframe.compute()
    dddf_final.reset_index(inplace = True, drop = True)
    return dddf_final

