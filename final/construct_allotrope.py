from allotropes.graphene import graphene
from allotropes.agnr import agnr
from allotropes.zgnr import zgnr
from allotropes.gnr import gnr
from allotropes.acnt import acnt
from allotropes.zcnt import zcnt
from allotropes.cnt import cnt

def construct_allotrope(name, config, include_H, backend):

    if name == "Graphene":
        return graphene(config, backend)
    
    elif name.find("AGNR") != -1:
        dash_idx = name.find("-")
        if dash_idx == -1:
            return None
        else:
            N_dimers = int(name[0:dash_idx])
            return agnr(name, N_dimers, config, include_H, backend)
        
    elif name.find("ZGNR") != -1:
        dash_idx = name.find("-")
        if dash_idx == -1:
            return None
        else:
            N_dimers = int(name[0:dash_idx])
            return zgnr(name, N_dimers, config, include_H, backend)
        
    elif name.find("GNR") != -1:
        dash_idx = name.find("-")
        comma_idx = name.find(",")
        lpar_idx = name.find("(")
        rpar_idx = name.find(")")
        if dash_idx == -1 or comma_idx == -1 or rpar_idx == -1 or lpar_idx == -1:
            return None
        else:
            N_dimers_x = int(name[lpar_idx+1:comma_idx])
            N_dimers_y = int(name[comma_idx+1:rpar_idx])
            return gnr(name, N_dimers_x, N_dimers_y, config, include_H, backend)
        
    elif name.find("CNT") != -1:
        dash_idx = name.find("-")
        comma_idx = name.find(",")
        lpar_idx = name.find("(")
        rpar_idx = name.find(")")
        if dash_idx == -1 or comma_idx == -1 or rpar_idx == -1 or lpar_idx == -1:
            return None
        else:
            if dash_idx == rpar_idx+1: # Single-wall
                N_1 = int(name[lpar_idx+1:comma_idx])
                N_2 = int(name[comma_idx+1:rpar_idx])
                if N_2 == N_1:
                    return acnt(name, N_1, config, include_H, backend)
                elif N_2 == 0:
                    return zcnt(name, N_1, config, include_H, backend)
                else:
                    return cnt(name, N_1, N_2, config, include_H, backend)

    else:
        return None
    