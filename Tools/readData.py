import pandas as pd
try:
    import uproot3 as uproot
except ImportError:
    import uproot
from dask import delayed
import dask.dataframe as dd
import gc

def daskframe_from_rootfiles(processes, treepath,branches,flatten='False',debug=False):
    def get_df(Class,file, xsecwt, selection, treepath=None,branches=['ele*']):
        tree = uproot.open(file)[treepath]
        if debug:
            ddd=tree.pandas.df(branches=branches,flatten=flatten,entrystop=1000).query(selection)
        else:
            ddd=tree.pandas.df(branches=branches,flatten=flatten).query(selection)
        #ddd["Category"]=Category
        ddd["Class"]=Class
        if type(xsecwt) == type("hello"):
            ddd["xsecwt"]=ddd[xsecwt]
        elif type(xsecwt) == type(0.1):
            ddd["xsecwt"]=xsecwt
        elif type(xsecwt) == type(1):
            ddd["xsecwt"]=xsecwt
        else:
            print("CAUTION: xsecwt should be a branch name or a number... Assigning the weight as 1")        
        print(file)
        return ddd

    dfs=[]
    for process in processes:
        dfs.append(delayed(get_df)(process['Class'],process['path'],process['xsecwt'],process['selection'],treepath, branches))
    print("Creating dask graph!")
    print("Testing single file first")
    daskframe = dd.from_delayed(dfs)
    print("Finally, getting data from")
    dddf_final=daskframe.compute()
    dddf_final.reset_index(inplace = True, drop = True)
    return dddf_final

