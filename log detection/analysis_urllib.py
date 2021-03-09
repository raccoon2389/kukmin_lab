import numpy as np
import pandas as pd
import urllib.parse as parse
import re
F_PATH = ['req2log2','req2logANOMAL2','req2logTEST2']
for P in F_PATH:
    req = pd.read_csv(f'log detection/dataset/{P}.csv',index_col=0)
    dataset = np.zeros((req.shape[0],5)) # Len_req, Len_arg, Num_arg, Len_path, Num_sp_char_path

    print(req.shape)
    l = []
    for idx,row in req.iterrows():
        path = row["Path:"]

        parsed =parse.urlsplit(path)


        if not parsed.query:
            arg = row["Arg:"]
            if str(arg) == 'nan':
                arg=''
        else:
            arg = parsed.query


        dataset[idx,:] = [
                len(row["Method:":"HTTP_ver:"]),
                len(arg),
                arg.count("="),
                len(parsed.path),
                len(re.findall("[^a-zA-Z0-9_]",parsed.path))

                ]
        # print(a.path)
        # try:
        #     dataset[idx,-1] = len(re.findall("[^a-zA-Z0-9_]",parsed.path))
        # except ValueError:
        #     dataset[idx,-1] = 0
        # l.append(a)
        # print(a)
    print(dataset)
    np.save(f'log detection/dataset/{P}.npy',dataset)
