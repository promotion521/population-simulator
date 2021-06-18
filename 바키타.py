import random
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 개체(객체) 생성을 위한 클래스. 수컷 1, 암컷 2
class popul(object):
    def __init__(self):
        global age_num
        self.age = age_num  # 시작 나이
        self.sex = random.randint(1, 2)


class new_popul(object):
    def __init__(self, parent1, parent2):
        self.age = 0
        self.sex = random.randint(1, 2)

# 혼획, 피식 등에 의한 사망 함수
def random_death(population):
    new_population = list(set(population) - set(
        random.sample(population, abs(int(0.05 * len(population) * float(np.random.randn(1)) + random_death_number)))))
    return new_population


# 전체 사망 함수
def death(population):
    population = random_death(population)  # 일단 주석처리, 큰 의미가 없음.
    return(population)

# 번식
def generate(population):
    popul_density = len(population)/inhabiting_area
    breeding_ratio = 0.1 * popul_density ** (1/4)
    for n in population:
        n.age += 1
    old = {n for n in population if n.age >= 20}  # 죽을 개체 set
    population = list(set(population) - old)
    adult_popul = [n for n in population if n.age >= 5]  # 번식 가능 나이(성체)
    baby_popul = [n for n in population if n.age <= 4]  # 번식 불가능 나이(새끼)
    generating_popul = int(breeding_ratio * len(adult_popul))  # 번식 참여 개체 수(암수 각각) 설정, 상수 0~0.5 사이의 값, 번식 참여율은 상수*200%
    if generating_popul > len([n for n in adult_popul if n.sex == 1]):
        male_list = [n for n in adult_popul if n.sex == 1]
    else:
        male_list = random.sample([n for n in adult_popul if n.sex == 1], generating_popul)  # 번식참여비율만큼 번식 참여
    if generating_popul > len([n for n in adult_popul if n.sex == 2]):
        female_list = [n for n in adult_popul if n.sex == 2]
    else:
        female_list = random.sample([n for n in adult_popul if n.sex == 2], generating_popul)
    for n in range(1, generating_popul):
        if male_list == [] or female_list == []:
            break
        male = random.choice(male_list)
        female = random.choice(female_list)
        population.extend(offspring(male, female))  # 임의의 암컷과 수컷이 번식
    return population


def offspring(p1, p2):
    lst = []
    offspring_num = 1  # 새끼수
    for n in range(int(offspring_num)):
        lst.append(new_popul(p1, p2))  # 전체 개체수 리스트에 새로 태어난 새끼 추가
    return lst


# 메인 프로그램
strt_popul_num = list(input('시작 개체수: n세의 개체수').split(','))
generation_number = int(input('세대수'))
inhabiting_area = int(input('서식 면적'))
report_term = int(input('결과 출력 간격'))

# 일부 변수, 상수는 입력받지 않고 프로그램 내에서 수정
population = []
popul_data = []
male_data = []
female_data = []
term_data = list(range(0, generation_number + 1, report_term))
gene_death_lst = []
sum_gene_death_lst = []
gene_death_data = []
gene_pool_data = []

# 시작 후 개체군 생성
for n in range(0, 19 + 1):
    age_num = n
    for i in range(int(strt_popul_num[n])):
        population.append(popul())

# 매년 같은 과정 반복 - 결과 출력, 번식, 사망 순
for i in range(generation_number + 1):
    # 연도에 따른 혼획률 증가 구현을 위한 코드
    if i<=18:
        random_death_number = (0.1 + 0.01*i) * len(population)
    else:
        random_death_number = 0.28 * len(population)
    '''elif i<=15:
        random_death_number = (0.3 - 0.01*i) * len(population)
    else:
        random_death_number = 0.15'''
    # 개체수가 0이 되면 '{i}세대에 멸종' 출력
    if len(population) == 0:
        print(f'{i}세대에 멸종')
        term_data = list(range(0, i, report_term))
        break
    # 입력한 report_term 간격마다 개체수와 전년도 대비 감소율 출력
    if i % report_term == 0:
        popul_data.append(len(population))
        male_data.append(len([x for x in population if x.sex == 1]))
        female_data.append(len([x for x in population if x.sex == 2]))
        print(f'{i}세대 개체수: {len(population)}({[len([x for x in population if x.age == n]) for n in range(0,19+1)]})')
        print(f'전년도 대비 감소율: {round((1-popul_data[i]/popul_data[i-1]) * 100, 2)}%')
    population = generate(population)
    population = death(population)

# 최대최소값 출력
print('\n')
print(f'최소 개체수:{min(popul_data)}마리, 최다 개체수:{max(popul_data)}마리')

# 엑셀로 데이터 저장
data = {'population': popul_data}
data = pd.DataFrame(data)
data.to_excel(excel_writer='2-5.xlsx')

# 그래프 작성 코드
plt.figure(figsize=(16, 12))
plt.subplots_adjust(hspace=0.5)

# 개체수 그래프
plt.subplot(311)
plt.plot(term_data, popul_data, marker='o', label='Total')
plt.plot(term_data, male_data, marker='o', label='Male')
plt.plot(term_data, female_data, marker='o', label='Female')
plt.xlabel('Generation Number')
plt.ylabel('Population Number')
plt.title('Population Graph')
plt.legend(loc='center right')

plt.show()