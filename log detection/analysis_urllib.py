from math import isnan
import numpy as np
import pandas as pd
import urllib.parse as parse
import re
F_PATH = ['req2log2','req2logANOMAL2']

all = re.compile("[^a-zA-Z0-9_]")
dot = re.compile('[.]')
dig = re.compile("[0-9]")
char = re.compile("[a-zA-Z]")
letter = re.compile("[a-zA-Z0-9_]")



for P in F_PATH:
    req = pd.read_csv(f'log detection/dataset/{P}.csv',index_col=0)
    dataset = np.zeros((req.shape[0],10))    # Len_req, Len_arg, Num_arg, Len_path, Num_sp_char_path
                                            # num_digit_arg, num_letter_arg, num_letter_char_arg,Maximum_bytes

    # print(req.shape)
    l = []
    for idx,row in req.iterrows():
        con = row["Content-Length:"]
        con_len = 0
        if not isnan(con):
            con_len = con
        # quit()
        path = row["Path:"]
        parsed =parse.urlsplit(path)
        path = path.split("/")[3:-1]
        path = "/".join(path)
        # print(path)
        # quit()
        
        if not parsed.query:
            arg = row["Arg:"]
            if str(arg) == 'nan':
                arg=''
        else:
            arg = parsed.query
        # req_len = "".join(row["Method:":"HTTP_ver:"])
        req_len = "".join( row["Method:":].astype("str"))

        # print(req_len)
        sp_len= len(all.findall(parsed.path))
        sp_len_dot = len(dot.findall(parsed.path))
        sp_len = sp_len-sp_len_dot
        # print(parsed.path)
        # print(sp_len)
        # quit()
        

        dataset[idx,:] = [
                len(req_len),
                len(arg),
                arg.count("="),
                len(path),
                len(re.findall("[^a-zA-Z0-9_]",arg)),
                sp_len,
                len(dig.findall(arg)),
                len(letter.findall(arg)),
                len(char.findall(path)),
                con_len


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
