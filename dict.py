dictus = [{'k1':'v1', 'k2':'v2', 'k3':'v3'}, {'k4':'v1', 'k2':'v2', 'k3':'v3'}]

for d in dictus:

    d['k7'] = d.pop('k4')

print(dictus)


