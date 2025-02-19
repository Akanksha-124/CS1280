n = int(input("Enter a number:"))
for i in range(n-1,0,-1):
    print(" " * (n-i) +"*" * (2 * i -1))