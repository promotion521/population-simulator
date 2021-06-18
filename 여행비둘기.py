import random
import string
import matplotlib.pyplot as plt
import numpy as np

# 개체(객체) 생성을 위한 클래스. 나이: 0살 수컷 1, 암컷 2
import pandas as pd

# 개체(객체) 생성을 위한 클래스
class popul(object):
    def __init__(self):
        global age_num
        self.age = age_num  # 시작 나이
        self.sex = random.randint(1, 2)  # 1은 수컷, 2는 암컷
        self.gene = ''.join(random.sample(string.ascii_lowercase, 6))  # 유전자 서열


class new_popul(object):
    def __init__(self, parent1, parent2):
        mutation_probability = 10000  # 돌연변이 확률 조절 1/n

        self.age = 0
        self.sex = random.randint(1, 2)  # 1은 수컷, 2는 암컷
        self.gene = ''.join(parent1.gene[random.randint(0, 1)] + parent2.gene[random.randint(0, 1)] + parent1.gene[
            random.randint(2, 3)] + parent2.gene[random.randint(2, 3)] + parent1.gene[random.randint(4, 5)] + parent1.gene[random.randint(4, 5)])  # 부모로부터 대립유전자쌍 중 하나씩 유전
        a = random.randint(1, mutation_probability)  # 전체 경우의 수
        if a == 1:  # 돌연변이가 일어나는 경우의 수
            lst_gene = list(self.gene)
            lst_gene[random.randint(0, 5)] = random.choice(string.ascii_letters)
            self.gene = ''.join(lst_gene)


# 유전적 요인으로 인한 사망 조건 함수
def gene_death(population):
    for i in population:
        if i.gene[0] == i.gene[1] and i.gene[2] == i.gene[3] and i.gene[4] == i.gene[5] or sum(1 for c in i.gene if c.isupper()) >= 2:
            gene_death_lst.append(i)
            population.remove(i)

# 환경저항(종내 경쟁)으로 인한 사망 함수
def competition_death(population):
    if population == []:
        new_population = []
    else:
        popul_density = len(population) / inhabiting_area  # 개체 밀도
        overpopulation = 0.05 * popul_density * len(population)  # 앞의 상수 임의로 설정하여 환경저항이 작용하는 세기 결정
        standard_deviation = 0.01 * len(population)  # 사망 개체수의 표준편차

        age0_4_popul, age5_9_popul = [x for x in population if x.age < 5], [x for x in population if 5 <= x.age <= 9]
        age10_14_popul = [x for x in population if x.age > 9]

        # 나이대별 가중치 설정을 위해 나이대를 나누어 적용
        age0_4_popul = list(
            set(age0_4_popul) - set(random.sample(age0_4_popul, abs(int(
                (standard_deviation * float(np.random.randn(1)) + overpopulation) * len(age0_4_popul) / len(
                    population))))))
        age5_9_popul = list(
            set(age5_9_popul) - set(random.sample(age5_9_popul, abs(int(
                0.8 * (standard_deviation * float(np.random.randn(1)) + overpopulation) * len(age5_9_popul) / len(
                    population))))))
        age10_14_popul = list(
            set(age10_14_popul) - set(random.sample(age10_14_popul, abs(int(
                1.2 * (standard_deviation * float(np.random.randn(1)) + overpopulation) * len(age10_14_popul) / len(
                    population))))))
        new_population = age0_4_popul + age5_9_popul + age10_14_popul
    return new_population


# 수렵, 피식 등에 의한 사망 함수
def random_death(population):
    age0_4_popul, age5_9_popul = [x for x in population if x.age < 5], [x for x in population if 5 <= x.age <= 9]
    age10_14_popul = [x for x in population if x.age > 9]

    age0_4_popul = list(
        set(age0_4_popul) - set(random.sample(age0_4_popul, abs(int(1.2 * (0.05 * len(age0_4_popul) * float(np.random.randn(1)) + random_death_number * len(age0_4_popul)))))))
    age5_9_popul = list(
        set(age5_9_popul) - set(random.sample(age5_9_popul, abs(int(0.05 * len(age5_9_popul) * float(np.random.randn(1)) + random_death_number * len(age5_9_popul))))))
    age10_14_popul = list(
        set(age10_14_popul) - set(random.sample(age10_14_popul, abs(int(0.05 * len(age10_14_popul) * float(np.random.randn(1)) + random_death_number * len(age10_14_popul))))))
    new_population = age0_4_popul + age5_9_popul + age10_14_popul
    return new_population

# 전염병에 의한 사망 함수(특정 유전자에 치명적) - 현실과 괴리, 큰 의미 없음
'''def plague_death(population):
    if random.choice(range(26)) == 0:
        weak_gene = random.choice(list(gene_pool))
        print(weak_gene)
        infected_lst = []
        for n in population:
            if weak_gene in n.gene:
                infected_lst.append(n)
                population.remove(n)
        print(len(infected_lst))
    else:
        pass'''


# 전체 사망 함수
def death(population):
    gene_death(population)
    population = competition_death(population)
    population = random_death(population)  # 일단 주석처리, 큰 의미가 없음.
    plague_death(population)  # 일단 주석처리
    return population

# 번식
def generate(population):
    for n in population:
        n.age += 1  # 나이 추가
    old = {n for n in population if n.age >= 15}  # 죽을 개체 set
    population = list(set(population) - old)  # 늙은 개체 사망
    adult_popul = [n for n in population if n.age >= 2]  # 번식 가능 나이(성체)
    baby_popul = [n for n in population if n.age <= 1]  # 번식 불가능 나이(새끼)
    generating_popul = int(0.48 * len(adult_popul))  # 번식 참여 개체 수(암수 각각) 설정, 상수 0~0.5 사이의 값, 번식참여율은 n*200%
    if generating_popul > len([n for n in adult_popul if n.sex == 1]):  # 전체개체수의 n%가 전체 수컷 개체수보다 많을 경우 전체 수컷 번식 참여
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
    offspring_num = float(np.random.randn(1)) + average_birth  # 새끼수
    for n in range(int(offspring_num)):
        lst.append(new_popul(p1, p2))  # 전체 개체수 리스트에 새로 태어난 새끼 추가
    return lst


# 메인 프로그램
strt_popul_num = list(input('시작 개체수: n세의 개체수').split(','))
generation_number = int(input('세대수'))
inhabiting_area = int(input('서식 면적'))
report_term = int(input('결과 출력 간격'))

# 일부 변수, 상수는 입력받지 않고 프로그램 내에서 수정
average_birth = 1  # 새끼 평균 1마리
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
for n in range(0, 14 + 1):
    age_num = n
    for i in range(int(strt_popul_num[n])):
        population.append(popul())

# 매년 같은 과정 반복 - 결과 출력, 번식, 사망 순
for i in range(generation_number + 1):
    # 연도에 따른 수렵비율 변화를 위한 코드
    if i < 50:
        random_death_number = 0.05
    elif 50 <= i < 100:
        random_death_number = 0.5 * (1-(i-50)/50)
    else:
        random_death_number = 0
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
        print(
            f'{i}세대 개체수: {len(population)}({[len([x for x in population if x.age == n]) for n in range(0,14+1)]})')
        print(f'전년도 대비 감소율: {round((1 - popul_data[i] / popul_data[i - 1]) * 100, 2)}%')
        # 밑의 성비 출력 부분은 현재 프로그램상으로는 의미 없으므로 생략
        '''if len([x for x in population if x.sex==2]) == 0:
            pass
        else:
            print(f'{i}세대 수/암: {round(len([x for x in population if x.sex==1])/len([x for x in population if x.sex==2]), 3)}')'''
        gene_pool = {x.gene[n] for n in range(0, 3) for x in population}
        gene_pool_data.append(len(gene_pool))
        gene_death_data.append(len(sum_gene_death_lst))
        sum_gene_death_lst = []
    sum_gene_death_lst.extend(gene_death_lst)
    gene_death_lst = []
    population = generate(population)
    population = death(population)

# 최대최소값 출력
print('\n')
print(f'최소 개체수:{min(popul_data)}마리, 최다 개체수:{max(popul_data)}마리')
print(f'최소 유전자 수:{min(gene_pool_data)}개, 최다 유전자 수:{max(gene_pool_data)}개')

# 엑셀로 데이터 저장
data = {'population': popul_data}
data = pd.DataFrame(data)
data.to_excel(excel_writer='test.xlsx')

# 그래프 작성 코드
plt.figure(figsize=(16, 12))
plt.subplots_adjust(hspace=0.5)

# 개체수 그래프
plt.subplot(311)
plt.plot(term_data, popul_data, marker='o', label='Total')
#암수 개체수 그래프 생략
'''plt.plot(term_data, male_data, marker='o', label='Male')
plt.plot(term_data, female_data, marker='o', label='Female')'''
plt.xlabel('Generation Number')
plt.ylabel('Population Number')
plt.title('Population Graph')
plt.legend(loc='center right')

# gene_death 그래프
plt.subplot(312)
plt.bar(term_data, gene_death_data)
plt.xlabel('Generation Number')
plt.ylabel('Gene Death Number')
plt.title('gene_death')

# 유전자풀 내 유전자 개수 그래프
plt.subplot(313)
plt.plot(term_data, gene_pool_data, marker='o', label='Gene Pool')
plt.xlabel('Generation Number')
plt.ylabel('Number of Genes')
plt.title('Gene Pool')
plt.legend(loc='center right')

plt.show()