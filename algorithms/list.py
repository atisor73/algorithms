def flatten(lst):
    try: 
        iter(lst)
        return [elem for sublst in lst for elem in flatten(sublst)]
    except: 
        return [lst]