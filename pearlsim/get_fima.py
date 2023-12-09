
def get_fima(conc_dict, initial_atoms ):
    final_actinides = 0.0
    for key in conc_dict.keys():
        if int(key[0:2]) >= 89:
            final_actinides += conc_dict[key]
    return (initial_atoms - final_actinides)/initial_atoms
    