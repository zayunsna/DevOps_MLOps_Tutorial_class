
word = ["취존","솔까말","케바케"]
sol = ["취향존중","솔직히 까놓고 말해서","케이스 바이 케이스"]
count = 0

print("Wellcome!")

for i in range(len(sol)):
    print(word[i]+" 의 뜻은 ? ")
    aws = input(" 입력 : ")
    if aws == sol[i]:
        count += 1
        print("정답")
    else:
        print("오답")
    

print(f'3개 퀴즈 중 {count}개 정답')
