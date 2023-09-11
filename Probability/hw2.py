from cmath import log
from re import X
import sys
import math
from collections import Counter

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict() #Stores values in (letter : # of letter occurrences) pairs
    for x in range(65, 91):
      X.update({chr(x) : 0})

    with open (filename,encoding='utf-8') as f:
        fileString = f.read() #Convert file into a string
        #go through each value in the file string, identifying letters with ASCII, ignoring all nonletters
        for element in fileString: 
            counter = 0
            if ord(element) >= 65 and ord(element) <= 90:  #if letter is uppercase, check it and its lowercase counterpart
                counter = fileString.count(element) + fileString.count(chr(ord(element) + 32))
          
                #Update dictionary counts with letter:letterCount key, value pairs 
                #so if there are 30 'a' characters in the file, it'll input {A:30} into the dictionary
                X.update({element.upper() : counter})

            elif ord(element) >= 97 and ord(element) <= 122: #if letter is lowercase, check it and its lowercase counterpart
                counter = fileString.count(element) + fileString.count(chr(ord(element) - 32))
                X.update({element.upper() : counter})
        
       
        return X

# Question 1
print("Q1")
letters = shred("letter.txt")
for letter in letters:
    print(letter, letters.get(letter))

#Question 2
print("Q2")
paramVect = get_parameter_vectors()
x1 = letters.get('A')

e1 = paramVect[0][0] #get e1
log1 = x1 * math.log(e1)
print('{:.4f}'.format(log1))

s1 = paramVect[1][0] #get s1
log2 = x1 * math.log(s1)
print('{:.4f}'.format(log2))

#Question 3
#Compute F(English) and F(Spanish) print values up to 4 decimal places in two separate lines
print("Q3")
fEng = 0
fSpan = 0
sumE = 0
sumS = 0
currLetter = 65 #start at A

for x in range(26):
    sumE = sumE + letters.get(chr(currLetter))*math.log(paramVect[0][x])
    sumS = sumS + letters.get(chr(currLetter))*math.log(paramVect[1][x])
    currLetter = currLetter + 1
  
fEng = math.log(0.6) + sumE
fSpan = math.log(0.4) + sumS

print('{:.4f}'.format(fEng))
print('{:.4f}'.format(fSpan))

#Question 4
#Computer P(Y = English | X) printe value up to 4 decimal places
print("Q4")
pEnglish = 0

if fSpan - fEng >= 100:
    pEnglish = 0
elif fSpan - fEng <= -100:
    pEnglish = 1
else:
    pEnglish = 1/(1+math.exp(fSpan - fEng))

print('{:.4f}'.format(pEnglish))

# You are free to implement it as you wish!
# Happy Coding!
