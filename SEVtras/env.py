Adata = None
Inter_adata = None
Thershold = None
Same = None
Max_M = None

def initializer(inter_adataI, sameI, thersholdI, max_M):
    global Inter_adata
    global Thershold
    global Same
    global Max_M
    Inter_adata = inter_adataI
    Thershold = thersholdI
    Same = sameI
    Max_M = 15000 if max_M < 500 else max_M


def initializer_simple(inter_adataI):
    global Inter_adata
    Inter_adata = inter_adataI

def initializer_adata(adataI, sameI, thersholdI, max_M):
    global Adata
    global Thershold
    global Same
    global Max_M
    Adata = adataI
    Thershold = thersholdI
    Same = sameI
    Max_M = 15000 if max_M < 500 else max_M

