import random


# flips random bit in number
def bitFlipper(number):
    a = bin(number)[2:]
    rand = random.randint(0,len(a)-1)
    print(rand)
    b = a[rand]
    if b == '1':
        b = '0'
    else:
        b = '1'
    d = a[:rand] + str(b) + a[rand+1:]
    return int(d, 2)
