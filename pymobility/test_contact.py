from pymobility.models.contact import dynamic_gnp

m = dynamic_gnp(200, p=0.01)

for contacts in m:
    for source, target in contacts:
        print(source, target)
