import random


print("Let's assume the number!")
print("Number Range : 1 ~ 20")
print("Total trial : 3 times")

target = random.randrange(1,20)
status = 0


for i in range(3):
    anw = int(input(" Guess what : "))
    if anw > target:
        print("Down!")
    elif anw < target:
        print("Up!")
    elif anw == target:
        status = 1
        break

if status == 1:
    print("Correct!")
else:
    print("Fail... :(")