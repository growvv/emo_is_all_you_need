import re
import ipdb

def myreplace_1(matched):
    # ipdb.set_trace()
    matched = matched.group()
    return matched[:1] + '和' + matched[1:3]

def myreplace_2(matched):
    # ipdb.set_trace()
    matched = matched.group()
    return matched[:2] + '和' + matched[2:3]

def myreplace_two(matched):
    # ipdb.set_trace()
    matched = matched.group()
    return matched[:2] + '和' + matched[2:4]

def myreplace_three(matched):
    # ipdb.set_trace()
    matched = matched.group()
    return matched[:2] + '和' + matched[2:4] + '和' + matched[4:]

def myreplace(matched):
    matched = matched.group()
    return ' ' +  matched + ' '


def clear_data(str):
    #if str ==  "":
    #    return str
    #print("hhh: ", str)
    #str = re.sub('[a-z][0-9][a-z][0-9][a-z][0-9]', myreplace_three, str)
    #str = re.sub('[a-z][0-9][a-z][0-9]', myreplace_two, str)
    #str = re.sub('[a-z][a-z][0-9]', myreplace_1, str, flags=re.I)
    #str = re.sub('[a-z][0-9][a-z]', myreplace_2, str, flags=re.I)
    str = re.sub('[a-z][0-9]', myreplace, str, flags=re.I)
    return str


if __name__ == '__main__':
    # str = 'b1u2接过袋子打开看了看那个a1b2'
    # str2 = '外五人路过婚纱店，m2r2不约而同回头走了进去，q2o2p2跟进'
    str = '防暴u2OS：住手，停止抵抗'
    # print(clear_data(str))
    print(clear_data(str))
