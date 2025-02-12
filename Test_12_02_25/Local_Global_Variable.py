x = 8
print(f"The global variable x - value is {x}")
def my_fn():
    y = 12
    print (f"The local variable y - value is {y}")
    print(f"The global variable x - value is {x} ")
    
my_fn()