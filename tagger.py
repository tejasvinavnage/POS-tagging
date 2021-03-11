# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:36:29 2019

@author: tejasvi
"""

import nltk
import re
from nltk.util import ngrams

# Reading the file_train
myfile = 'pos-train.txt'  
wtxt = open(myfile).read()
#token the whole text to word
wtokens = nltk.word_tokenize(wtxt)
wtfreq = nltk.FreqDist(wtokens)#get the dictionary of the frequecncy per token in the whole text


#############################p(w/t)

tDict = {} ##tDict: the total frequency per tagging 
# now loop over each token, and its frequency in wtfreq
#count the tagging 
for word, frequency in wtfreq.most_common(len(wtfreq)): #len(wfreq):Number of unique words in text
    if '/' in word:
    #if (re.match('*//*',word)):
        list=re.split(r'[/]', word) #split "word/tagging" to word and tagging
        if list[0] and list[1]: #valid word tagging paris
            tDict[list[1]]= tDict.get(list[1],0)+frequency #total tagging frequency (tagging is Case Sensitive)
            #print(u'{};{}'.format(word, frequency)) #can comment later
            

###Probablity: wtProbdict           
#wtProbdict: the table of the probability p(w|t)               
wtProbdict ={}               
for word, frequency in wtfreq.most_common(len(wtfreq)):
    if '/' in word:
        list=re.split(r'[/]', word)
        if list[0] and list[1]:
            wtProbdict[word]=frequency/tDict.get(list[1])#the probability p(w|t) =P(w,t)/p(t)
            
#print(wtProbdict)#wtProbdict: the table of the probability p(w|t) , THE TABLE U CAN USE   

#case sensitive ,i.e. "The" is different from "the";  "And", "and","aNd",... are different, "OF", "of", "Of","oF" are different,etc
#test the table--wtProbdict:
#e.g.:
#the
print('the/DT')
print(wtProbdict.get('the/DT',0)) #if not exit, return default 0
print('the/JJ')
print(wtProbdict.get('the/JJ',0))
print()
#The
print('The/DT')
print(wtProbdict.get('The/DT',0))
print()
#sound
print('sound/NN')
print(wtProbdict.get('sound/NN',0))
print('sound/VBP')
print(wtProbdict.get('sound/VBP',0))
print()
#sounds
print('sounds/VBZ')
print(wtProbdict.get('sounds/VBZ',0))
print('sounds/NNS')
print(wtProbdict.get('sounds/NNS',0))
#more vbz
print('sounds/VBZ')
print(wtProbdict.get('is/VBZ',0))
print('has/VBZ')
print(wtProbdict.get('has/VBZ',0))
print('says/VBZ')
print(wtProbdict.get('says/VBZ',0)) 



#############################p(t_i/t_i-1)
#ttDict: the table counting (t_i-1,t_i)               
ttDict ={} 
#ti_1Dict: the table counting  tagging  (t_i-1)            
ti_1Dict ={} 

sent_text = nltk.sent_tokenize(wtxt) #  a list of sentences
# now loop over each sentence and tokenize it separately
#count the tagging per each sentence
for sentence in sent_text:
    wordlist = []
    tokenized_text = nltk.word_tokenize(sentence)
    for word in tokenized_text:
        #ingnore [,], and other punctuations, just leave the word/tagging ones
        if word != '[' and word != ']' and ('/' in word) and word!='/' and word!='': 
            list=re.split(r'[/]', word) #split "word/tagging" to word and tagging
            wordlist.append(list[1])#only count the taggings
#        print(wordlist)##!can comment later, for each sentence
#        print()##!can comment later, for each sentence
        bigrams = ngrams(wordlist, 2)
        for b in bigrams:
            if b[0]!='' and b[1]!='':#filter the invalid ones
                #check if the bigram is in ttDict or not
                if b not in ttDict:#not in, add it
                    ttDict[b] = 1
                else:#already in, add 1 to the count
                    ttDict[b] += 1
                #check if the bigram[0], i.e., t_i-1 is in ti_1Dict or not
                if b[0] not in ti_1Dict:#not in, add it
                    ti_1Dict[b[0]] = 1
                else:#already in, add 1 to the count
                    ti_1Dict[b[0]] += 1
                    
##can comment later, for ttDict: the table counting (t_i-1,t_i)               
#print(ttDict)##can comment later,  



###Probablity: wtProbdict                
#ttProbDict: the table of the probability p(t_i|t_i-1) p(currenttag|previoustag)               
ttProbDict ={}               
for bikey, value in ttDict.items(): #traverse the ttDict
    ttKey = bikey[0]+'/'+bikey[1] #creat key like: tag/tag
    ttProbDict[ttKey]=value/ti_1Dict.get(bikey[0])#get the probability p(t_i|t_i-1) =P(t_i-1,t_i)/p(t_i-1)
    
#print(ttProbDict)#print p(t_i|t_i-1)  result

#test the table--ttProbDict:
#e.g.:
print('NNP/NNP')
print(ttProbDict.get('NNP/NNP',0)) #if not exit, return default 0
print('NNP/CD')
print(ttProbDict.get('NNP/CD',0))
print('CD/NNS')
print(ttProbDict.get('CD/NNS',0))

#========================Reading the test file and assigning tags=============================
#variable initialization
testwordslist = []
testsenttokenize = []

#Reading the file_test
#file_test = "pos-test.txt"
file_test = "opt.txt"
mytest = open(file_test).read()

#file for writing the tags to the pos-test file
file_output = open("pos-test-with-tag.txt","w")

#findexactword(No, ['No/DT', 'No/RB'], [0.0016273393002441008, 0.0005624296962879641]) 
def findexacttag(word, wordtaglist, tagproblist): #for finding exact tags
    tagtag =[] #list consisting of current and previous tags
    ttprob = [] #list of probabilities of current and previous tags
    
    td =[]
    for w in wordtaglist:
        g = w.split('/')[1]
        td.append(g)
    
    f =''
    ft =''
#    td = ['RB', 'DT']
#    print('element tag:',td)
    for element in td:
        u = element+'/'
        j = [k for k,v in ttProbDict.items() if u in k]
        for n in j:
            f = n.split('/')[0]
            ft = n.split('/')[1]
            for element in td:
                if (element == f):
                    h = f+'/'+ft
                    tagtag.append(h)
                    ttprob.append(ttProbDict.get(h,0))
#    print('ttpair------------------------')
#    print(tagtag)
#    print(len(tagtag))
#    #['RB/VBN', 'RB/JJ', 'RB/IN', 'RB/JJR', 'RB/MD', 'RB/VB', 'RB/DT', 'RB/VBD', 'RB/TO', 'RB/NN', 'RB/VBG', 'RB/CC', 'RB/NNP', 'RB/VBP', 'RB/RB', 'RB/PRP', 'RB/CD', 'RB/VBZ', 'RB/RBR', 'RB/NNS', 'RB/WRB', 'RB/WDT', 'RB/WP', 'RB/EX', 'RB/JJS', 'DT/NN', 'DT/JJ', 'DT/NNP', 'DT/NNS', 'DT/JJS', 'DT/VBZ', 'DT/VBN', 'DT/CD', 'DT/IN', 'DT/WP', 'DT/JJR', 'DT/VBG', 'DT/TO', 'DT/NNPS', 'DT/RB', 'DT/VBD', 'DT/RBR', 'DT/CC', 'DT/PRP', 'DT/DT', 'DT/MD', 'DT/FW', 'DT/RBS', 'DT/McGraw', 'DT/VBP', 'DT/Contra', 'DT/WDT']
#    print('ttpair probability------------')
#    print(ttprob)
#    print(len(ttprob))
    #[0.09472435736605615, 0.09627534612138768, 0.14583856031749654, 0.008119882307323861, 0.014574732568482997, 0.08067422393540588, 0.09299089934539151, 0.0838218187624022, 0.021668225258308055, 0.027644093697967748, 0.03683598293912369, 0.01626257327281436, 0.01966106333964373, 0.03722373012795657, 0.08106197112423875, 0.015236183655315557, 0.045959446205779714, 0.05029309125744132, 0.007230344638824898, 0.014483497935816436, 0.003147594826996328, 0.0017790753369979243, 0.001254476199165203, 0.002896699587163287, 0.0003421298724996008, 0.47568659304954886, 0.19953458760510198, 0.1328254776697182, 0.08197405106364294, 0.007991043006739699, 0.006205498598640352, 0.009088713749423723, 0.02536351196095219, 0.010940118402084111, 0.0007390983000739098, 0.0076910130037393986, 0.013955054041989564, 0.0007171448852202293, 0.00223924831507541, 0.010398600835693326, 0.0011562131822938392, 0.001975807336831244, 0.0008122763495861782, 0.0020416675813922855, 0.001719684163538305, 0.001895311482367749, 0.0011123063525864782, 0.0031320205191250833, 0.00021221634358557808, 0.00033661902775643417, 0.00019026292873189758, 6.586024456104147e-05]        
    
    
    wtg = [k for k,v in wtProbdict.items() if 'No/' in k]
    print("wtg pairs -> ",wtg)
    #wtg pairs ->  ['No/DT', 'No/RB']
    
    wtgp = []
    for p in wtg:
        wtgp.append(wtProbdict.get(p,0))    
#        print('probability ->',wtProbdict.get(p,0))
#    print(wtgp)
    #[0.0016273393002441008, 0.0005624296962879641]
    
    
    maxm = 0
    prevmax = 0
#    index = 0
    pairindex = 0
    tagged = ''
    words = ''
    t = ''
    for i in range(len(wtg)):
#        print("------------------")
#        print("outerloop", i)
        k = wtg[i]
        l = k.split('/')[1]
#        print("l ->",l)
#        print("pairindex", pairindex)
        for n in range(len(tagtag)):
#            print("innerloop",n)
            u = tagtag[n]
            v = u.split('/')[0]
            if(l == v):
#                print(wtg[i],'X',tagtag[n])
                maxm = ttprob[n]*wtgp[i]
    
                if(maxm >= prevmax):
                    prevmax = maxm
#                    print("new max: index",n,"pairvalue",i,prevmax)
                    tagged = tagtag[n]
#                    print(tagged)
                    words = wtg[i]
#                    print(words)
                    t = words +'|'+ tagged
                    
                else:
                    prevmax = prevmax
    
    
    
#    print("final max:",prevmax)
#    print(tagged)
#    print(words)
                    

    return t

#    findexactword(No, sentence, ['No/DT', 'No/RB'], [0.0016273393002441008, 0.0005624296962879641])
def findexactword(w, s, wt, wtp):
    exactword = ''
    e = w
    twp = []
    for ew in wt:
        new = ew.split('/')[0]
        newt = ew.split('/')[1]
        if (e == new):#to match word with the key of wtprob retrieved tag
            twp.append(e)
            exactword = new + '/' + newt
#           exactword = [k for k,v in wtProbdict.items() if q in k]
           
#            write code for getting highest probability tag i.e another function
        elif e == ',' or e == '--' or e == '.' :
            exactword = e
        
#    findexacttag(e, new, newt)
    return exactword
                
            
    

#
#wt = ['No/DT', 'No/RB']
#e = 'No'
#twp = []
#for ew in wt:
#        new = ew.split('/')[0]
#        newt = ew.split('/')[1]
#        if (e == new): #to match word with the key of wtprob retrieved tag
#           twp.append(e)
#           exactword = new + '/' + newt
#           exactword = [k for k,v in wtProbdict.items() if q in k]
           
#            write code for getting highest probability tag i.e another function
#        elif e == ',' or e == '--' or e == '.' :
#            exactword = e
#print(exactword)

#Code for reading and assigning the tags
#formula for tag t = p(w|t)*p(t|t(n-1))
#No , 
#[ it ]
#[ was n't Black Monday ]
#.
    
#But while 
#[ the New York Stock Exchange ]
#did n't 
#[ fall ]
#apart 
#[ Friday ]
#as 
#[ the Dow Jones Industrial Average ]
#plunged 
#[ 190.58 points ]
#-- most of 
#[ it ]
#in 
#[ the final hour ]
#-- 
#[ it ]
#barely managed to stay 
#[ this side ]
#of 
#[ chaos ]
#.    
def mytokenize(w, s): #w -> word, s-> sentence
    wor = w #contains the word
    w = w+'/' #word modified to get from dict
    s = s
    starttag = ''
    endtag = ''
    wt = [k for k,v in wtProbdict.items() if w in k] #find the key witn the word
#    ['No/DT', 'No/RB']
    wtp = [v for k,v in wtProbdict.items() if w in k] #find the value of tag
#    [0.0016273393002441008, 0.0005624296962879641]
    
#    findexactword(No, ['No/DT', 'No/RB'], [0.0016273393002441008, 0.0005624296962879641]) 
    nwt = findexactword(w, s, wt, wtp)      
#    nwt = findexacttag(wor, wt, wtp)  
    return nwt

#nwt = findexactword('No', "No , [ it ] [ was n't Black Monday ] ." , ['No/DT', 'No/RB'], [0.0016273393002441008, 0.0005624296962879641])
#print(nwt)
        
        
#print([v for k,v in wtProbdict.items() if 'No' in k])
#    b = [k for k,v in wtProbdict.items() if 'No' in k]
#r = []
#t = []
#dl = ['Nov./NNP', 'No/DT', 'Northeast/NNP', 'November/NNP', 'Not/RB', 'Now/RB', 'Northern/NNP', 'Norman/NNP', 'Novello/NNP', 'None/NN', 'No/RB', 'Northampton/NNP', 'Nonetheless/RB', 'Norwegian/NNP', 'No-Smoking/NNP', 'Nobel/NNP', 'North/NNP', 'No./NN', 'Notes/NNPS', 'None/NNP', 'Nov./NN', 'Norwest/NNP', 'Northy/NNP', 'Norwick/NNP', 'Noriega/NNP', 'Nor/CC']
#for name in dl:
#    da = name.split('/')[0]
#    dt = name.split('/')[1]
#
#    if 'No' == da:
#        r.append(da)
#        t.append(dt)       
#        print(da, '|', dt)
#print(r)
#print(t)
#wt = [k for k,v in wtProbdict.items() if 'Nov./' in k] #find the key witn the word
#print(wt)
#wtp = [v for k,v in wtProbdict.items() if 'Nov./' in k] #find the value of tag
#print(wtp)    
#tt = [k for k,v in ttProbDict.items() if 'RB' in k] #find the key witn the word
#print(tt)
#wt = [k for k,v in wtProbdict.items() if 'No/' in k] #find the key witn the word
#print(wt)
#print(len(wt))


testsenttokenize = nltk.sent_tokenize(mytest) #tokenizing each sentence in the file
# now loop over each sentence and tokenize words separately
for sent in testsenttokenize:    
    testwordtokenize = nltk.word_tokenize(sent) #tokenizing the sentence in testsenttokenize list
    for testword in testwordtokenize:
        if testword != '[' and testword != ']': #leaving out [,]
            testwordslist.append(testword)
            y = mytokenize(testword, sent)
            print(testword,'|',y)
#['Nov./NNP', 'No/DT', 'Northeast/NNP', 'November/NNP', 'Not/RB', 'Now/RB', 'Northern/NNP', 'Norman/NNP', 'Novello/NNP', 'None/NN', 'No/RB', 'Northampton/NNP', 'Nonetheless/RB', 'Norwegian/NNP', 'No-Smoking/NNP', 'Nobel/NNP', 'North/NNP', 'No./NN', 'Notes/NNPS', 'None/NNP', 'Nov./NN', 'Norwest/NNP', 'Northy/NNP', 'Norwick/NNP', 'Noriega/NNP', 'Nor/CC']
#['15,000/CD', '10,000/CD', '5,000/CD', '1,000/CD', '100,000/CD', '20,000/CD', '30,000/CD', '2,500/CD', '25,000/CD', '11,000/CD', '6,000/CD', '350,000/CD', '16,000/CD', '4,000/CD', '18,000/CD', '50,000/CD', '7,500/CD', '60,000/CD', '45,000/CD', '2,000/CD', '2,700/CD', '100,980/CD', '120,000/CD', '325,000/CD', '340,000/CD', '1,620/CD', '4,393,237/CD', '3,288,453/CD', '2,303,328/CD', '500,004/CD', '361,376/CD', '30,841/CD', '13,056/CD', '600,000/CD', '240,000/CD', '1,298/CD', '321,000/CD', '26,000/CD', '271,124/CD', '400,000/CD', '95,142/CD', '331,000/CD', '550,000/CD', '23,000/CD', '858,000/CD', '89,500/CD', '77,000/CD', '126,000/CD', '93,000/CD', '992,000/CD', '500,000/CD', '40,000/CD', '200,000/CD', '16,072/CD', '18,444/CD', '12,252/CD', '11,762/CD', '14,821/CD', '82,389/CD', '250,000/CD', '3,500/CD', '9,118/CD', '4,645/CD', '30,537/CD', '23,403/CD', '1,500/CD', '3,600/CD', '877,663/CD', '244,000/CD']
#['it/PRP', 'with/IN', 'its/PRP', 'unit/NN', 'securities/NNS', 'without/IN', 'University/NNP', 'capital/NN', 'limit/NN', 'political/JJ', 'additional/JJ', 'United/NNP', 'Fujitsu/NNP', 'credit/NN', 'city/NN', 'profit/NN', 'Institute/NNP', 'position/NN', 'City/NNP', 'addition/NN', 'Securities/NNP', 'Mitsubishi/NNP', 'institutions/NNS', 'acquisition/NN', 'committee/NN', 'profits/NNS', 'competition/NN', 'White/NNP', 'Committee/NNP', 'ability/NN', 'Mitsui/NNP', 'acquisitions/NNS', 'editor/NN', 'Despite/IN', 'equity/NN', 'composite/JJ', 'capacity/NN', 'neither/DT', 'conditions/NNS', 'little/JJ', 'Britain/NNP', 'With/IN', 'within/IN', 'university/NN', 'Without/IN', 'little/RB', 'situation/NN', 'Constitution/NNP', 'admitting/VBG', 'crocidolite/NN', 'deficit/NN', 'units/NNS', 'itself/PRP', 'despite/IN', 'visit/NN', 'activity/NN', 'positive/JJ', 'commitments/NNS', 'military/JJ', 'lawsuit/NN', 'suit/NN', 'activities/NNS', 'benefits/NNS', 'circuit/NN', 'British/JJ', 'Securities/NNPS', 'priority/NN', 'Whiting/NNP', 'institutional/JJ', 'traditional/JJ', 'Smith/NNP', 'written/VBN', 'spirit/NN', 'editorial/NN', 'writer/NN', 'politics/NNS', 'politicians/NNS', 'security/NN', 'admits/VBZ', 'critics/NNS', 'items/NNS', 'majority/NN', 'Heritage/NNP', 'quite/RB', 'editors/NNS', 'opportunity/NN', 'initially/RB', 'magnitude/NN', 'minority/NN', 'deposit/NN', 'deposits/NNS', 'subcommittee/NN', 'arbitrage/NN', 'Neither/DT', 'maturity/NN', 'maturities/NNS', 'authority/NN', 'electricity/NN', 'withdraw/VB', 'initial/JJ', 'competitor/NN', 'positions/NNS', 'recognition/NN', 'permitted/VBN', 'responsibility/NN', 'wait/VB', 'architecture/NN', 'low-ability/JJ', 'cities/NNS', 'prohibits/VBZ', 'Profit/NN', 'possibility/NN', 'quality/NN', 'Koito/NNP', 'institution/NN', 'Triton/NNP', 'citations/NNS', 'unconstitutional/JJ', 'Vitulli/NNP', 'exhibition/NN', 'Utilities/NNP', 'Circuit/NNP', 'utility/NN', 'creativity/NN', 'ambitious/JJ', 'creditors/NNS', 'intensity/NN', 'writers/NNS', 'reality/NN', 'commitment/NN', 'Morita/NNP', 'visiting/VBG', 'entitles/VBZ', 'Little/NNP', 'either/CC', 'admitted/VBD', 'tradition/NN', 'credibility/NN', 'white/JJ', 'ambitions/NNS', 'favorite/JJ', 'furniture/NN', 'authorities/NNS', 'Institutes/NNPS', 'withdrawn/VBN', 'permit/VB', 'citing/VBG', 'Detroit/NNP', 'litigation/NN', 'Writers/NNPS', 'Title/NN', 'community/NN', 'Capital/NNP', 'facility/NN', 'traditionally/RB', 'Commodity/NNP', 'criticized/VBD', 'fatalities/NNS', 'British/NNP', 'opposition/NN', 'limited/VBN', 'Hermitage/NNP', 'pit/NN', 'competitive/JJ', 'profitable/JJ', 'Institution/NNP', 'volatility/NN', 'Capitol/NNP', 'bit/NN', 'double-digit/JJ', 'Perritt/NNP', 'Jerritts/NNP', 'citizen/NN', 'either/DT', 'limited-partnership/NN', 'limit/VB', 'index-arbitrage/NN', 'Zenith/NNP', 'initiatives/NNS', 'cited/VBD', 'Micronite/NN', 'morbidity/NN', 'permit/VBP', 'capital-gains/JJ', 'titans/NNS', 'visitors/NNS', 'Credit/NNP', 'credits/NNS', 'Audit/NNP', 'withdrawal/NN', 'expedited/VBN', 'audit/VB', 'benefited/VBD', 'longevity/NN', 'either/RB', 'Hitachi/NNP', 'electric-utility/NN', 'Utilities/NNPS', 'citizens/NNS', 'instituted/VBN', 'initialing/VBG', 'literary/JJ', 'lower-priority/JJ', 'merit/VB', 'creditor/NN', 'built-from-kit/JJ', 'Strait/NNP', 'quantities/NNS', 'petition/NN', 'securities-based/JJ', 'lawsuits/NNS', 'Acquisition/NNP', 'editions/NNS', 'Political/JJ', 'depositary/JJ', 'cite/VBP', 'Pitney/NNP', 'implicit/JJ', 'erudite/JJ', 'solicitous/JJ', 'politely/RB', 'hitter/NN', 'dormitory/NN', 'solidarity/NN', 'committed/VBD', 'hospitals/NNS', 'Siti/NNP', 'hospital/NN', 'underwriters/NNS', 'inheritor/NN', 'refitting/VBG', 'black-and-white/JJ', 'exciting/JJ', 'intertitles/NNS', 'white/NN', 'Cosmopolitan/NNP', 'traitor/NN', 'hit/VB', 'hostility/NN', 'Napolitan/NNP', 'constituent/NN', 'sensitivity/NN', 'fits/NNS', 'knitted/VBN', 'sites/NNS', 'attitude/NN', 'bitterness/NN', 'Aptitude/NNP', 'Critics/NNS', 'split/JJ', 'elite/NN', 'Literacy/NN', 'competitions/NNS', 'unpopularity/NN', 'Editorials/NNS', 'witnesses/NNS', 'red-and-white/JJ', 'kit/NN', 'Metropolitan/NNP', 'similarity/NN', 'kits/NNS', 'indefinitely/RB', 'recruit/VB', 'unavailability/NN', 'permitting/VBG', 'depositary/NN', 'definitive/JJ', 'Security/NNP', 'identities/NNS', 'witness/NN', 'prohibited/VBN', 'committing/VBG', 'submit/VB', 'initiated/VBN', 'quitting/VBG', 'screenwriters/NNS', 'limits/NNS', 'exhibited/VBN', 'prostitute/NN', 'discredit/VB', 'white-collar/JJ', 'competitors/NNS', 'antitrust-law/JJ', 'situations/NNS', 'extramarital/JJ', 'municipality/NN', 'limited/JJ', 'Saitama/NNP', 'municipalities/NNS', 'productivity/NN', 'Deposits-a/NNP', 'metropolitan/JJ', 'Monitor/NNP', 'switch/VB', 'switch/NN', 'additions/NNS', 'positioned/VBN', 'jitters/NNS', 'waiting/VBG', 'vitriolic/JJ', 'cites/VBZ', 'Either/CC', 'hard-hitting/JJ', 'recyclability/NN', 'pitches/VBZ', 'identity-management/NN', 'Pittsburgh/NNP', 'flexibility/NN', 'conditional/JJ', 'withstand/VB', 'sport-utility/JJ', 'hit/VBN', 'Lafite-Rothschild/NNP', 'Taittinger/NNP', 'three-digit/JJ', 'hit/NN', 'excited/VBN', 'capital-gains/NNS', 'commodity/NN', 'cited/VBN', 'solicitation/NN', 'punitive/JJ', 'Criticism/NNP', 'skittishness/NN', 'Initiative/NNP', 'vitally/RB', 'litany/NN', 'Little/RB', 'fit/NN', 'architectural/JJ', 'Mitsubishi/NNS', 'commodities/NNS', 'omitted/VBD', 'safe-deposit/JJ', 'switched/VBN', 'rate-sensitive/JJ', 'split/VB', 'resubmit/VB', 'critical/JJ', 'institute/VB', 'hitting/VBG', 'circuit-breaker/NN', 'respite/NN', 'sensitive/JJ', 'writing/VBG', 'committees/NNS', 'peculiarities/NNS', 'faithful/NN', 'water-authority/NN', 'fit/JJ', 'dexterity/NN', 'highest-pitched/JJ', 'ritual/NN', 'sit/VB', 'exit/NN', 'entitled/VBD', 'inner-city/NN', 'equal-opportunity/NN', 'quantitative/JJ', 'hospitable/JJ', 'exits/NNS', 'single-digit/JJ', 'wherewithal/JJ', 'expenditures/NNS', 'commit/VB', 'capitalist/JJ', 'reciting/VBG', 'platitudes/NNS', 'visited/VBD', 'Spitler/NNP', 'Gaithersburg/NNP', 'security-type/JJ', 'Rita/NNP', 'Pitcher/NNP', 'exploit/VB', 'Index-arbitrage/NN', 'conduit/NN', 'low-altitude/NN', 'unsolicited/JJ', 'universities/NNS', 'institute/NN', 'transition/NN', 'inherited/VBD', 'high-quality/JJ', 'committed/VBN', 'Majority/NNP', 'initiate/NN', 'Citizens/NNS', 'Spirit/NN', 'Citicorp/NNP', 'architects/NNS', 'visits/NNS', 'architect/NN', 'citation/NN', 'spite/NN', 'editing/NN', 'edition/NN', 'Editor/NN', 'multitude/NN', 'malnutrition/NN', 'necessities/NNS', 'nutrition/NN', 'exhibits/VBZ', 'insanity/NN', 'cite/VB', 'Charities/NNPS', 'nonprofit/JJ', 'substitute/NN', 'benefit/NN', 'criteria/NNS', 'hither/RB', 'facilities/NNS', 'notwithstanding/IN', 'surreptitiously/RB', 'responsibilities/NNS', 'biscuit/NN', 'profitability/NN', 'sitting/VBG', 'depository/NN', 'Switzerland/NNP', 'recruited/VBN', 'withhold/VB', 'Constitutional/NNP', 'accountability/NN', 'unitary/JJ', 'limitation/NN', 'rewrite/VB', 'cost-benefit/JJ', 'prohibiting/VBG']
#['was/VBD', 'waste/NN', 'Kawasaki/NNP', 'unwashed/JJ', 'wasted/VBN']
#["n't/RB"]
#['Black/NNP', 'Blackstone/NNP']
#['Monday/NNP']
#['Mr./NNP', 'U.S./NNP', 'Corp./NNP', 'Mrs./NNP', 'Inc./NNP', 'Co./NNP', 'Ltd./NNP', 'R./NNP', 'Dr./NNP', 'Rep./NNP', 'U.S/NNP', 'Oct./NNP', 'D./NNP', 'T./NNP', 'Ms./NNP', 'Nov./NNP', 'J./NNP', 'Jr./NNP', 'St./NNP', 'A./NNP', 'M./NNP', 'Dec./NNP', 'Calif./NNP', 'G./NNP', 'Fla./NNP', 'L./NNP', 'E./NNP', 'C./NNP', 'Mass./NNP', 'B./NNP', 'Conn./NNP', 'Ill./NNP', 'P./NNP', 'Colo./NNP', 'N.J./NNP', 'N.Y./NNP', 'Pa./NNP', 'H./NNP', 'Mich./NNP', 'Sept./NNP', 'N.J/NNP', 'Sen./NNP', 'U.S.A./NNP', 'N.V./NNP', '1.5/CD', 'W.R./NNP', '3.1/CD', '2.2/CD', 'K./NNP', 'U.K./NNP', 'M.D./NNP', '8.50/CD', '8.45/CD', '8.04/CD', '7.90/CD', '7.5/CD', '3.2/CD', '7.3/CD', '2.6/CD', '4.8/CD', '2.5/CD', 'A./NN', 'F./NNP', 'R.P./NNP', '2.4/CD', '47.6/CD', '0.1/CD', '0.3/CD', '0.9/CD', '4.3/CD', '17.95/CD', 'Gov./NNP', 'Aug./NNP', 'N.M./NNP', 'Messrs./NNPS', '42.5/CD', 'D.C./NNP', '35.7/CD', '1.8500/CD', '1.8415/CD', '143.80/CD', '142.85/CD', 'N.C./NNP', '1.1/CD', '62.5/CD', '3.18/CD', '7.88/CD', '1.85/CD', '1.2/CD', 'N.C/NNP', 'p.m/RB', '3.3/CD', 'W./NNP', 'J.L./NNP', 'Miss./NNP', 'Ore./NNP', 'Nev./NNP', 'S./NNP', '4.25/CD', '3.35/CD', '8.55/CD', '8.25/CD', '8.07/CD', '7.95/CD', '9.8/CD', '8.47/CD', '352.7/CD', '9.37/CD', '9.45/CD', '8.12/CD', '8.14/CD', '8.19/CD', '8.22/CD', '8.53/CD', '8.56/CD', 'J.P./NNP', '83.4/CD', 'S.p/NNP', '2.80/CD', '2.87/CD', '5.29/CD', '0.7/CD', '5.39/CD', '50.45/CD', '50.38/CD', '2.29/CD', '2.25/CD', 'N.H./NNP', '5.5/CD', '3.75/CD', 'Feb./NNP', '1.55/CD', '737.5/CD', '3.01/CD', '38.375/CD', '12.5/CD', '72.7/CD', '0.4/CD', '2.875/CD', '98.3/CD', '4.75/CD', '19.3/CD', '5.9/CD', '20.5/CD', 'Messrs./NNP', '38.3/CD', 'F.H./NNP', 'Pty./NNP', '10.2/CD', '37.3/CD', '16.125/CD', '13.73/CD', '37.5/CD', '9.625/CD', 'O./NNP', 'S.p.A./NNP', '1.82/CD', '84.29/CD', '150.00/CD', '86.12/CD', 'U.S.S.R./NNP', '236.74/CD', '236.79/CD', '59.6/CD', '2.1/CD', '415.6/CD', '415.8/CD', '109.73/CD', '0.2/CD', '127.03/CD', '3.9/CD', '1.6/CD', '234.4/CD', '5.4/CD', '0.5/CD', '497.34/CD', '191.9/CD', '99.1/CD', 'F.W./NNP', '18.95/CD', '2.44/CD', '85.1/CD', '10.5/CD', 'Lt./NNP', '5.57/CD', '705.6/CD', 'U.S.-Japanese/JJ', '66.5/CD', '64.5/CD', 'H.N./NNP', '1.01/CD', '456.64/CD', '118.6/CD', '133.8/CD', '1.39/CD', '446.62/CD', '1.28/CD', '449.04/CD', '3.23/CD', '436.01/CD', 'L.P./NNP', '273.5/CD', '25.50/CD', '70.7/CD', '89.9/CD', '57.50/CD', '4.4/CD', 'Ala./NNP', '225.6/CD', 'Dec./NN', 'Ga./NNP', '1.8/CD', 'No./NN', 'U.S.-Japan/JJ', '14.6/CD', '32.8/CD', '28.6/CD', '29.3/CD', '28.4/CD', '6.21/CD', '133.7/CD', '77.70/CD', '77.56/CD', '1.5755/CD', '1.5805/CD', 'U.S./JJ', '143.93/CD', '143.08/CD', '45.3/CD', '374.20/CD', '3.5/CD', '374.19/CD', 'A.C./NNP', 'R.I./NNP', 'Ky./NNP', '2.95/CD', 'E.C./NNP', '62.625/CD', '121.6/CD', '11.5/CD', 'Jan./NNP', 'N.J.-based/JJ', '1.9/CD', '0.25/CD', '8.75/CD', '13.5/CD', '23.5/CD', 'Nov./NN', '13.8/CD', '47.5/CD', '25.6/CD', '21.9/CD', '2645.90/CD', '154.2/CD', '2.50/CD', '5.276/CD', '36.9/CD', 'Sept./JJ', '3.253/CD', '4.898/CD', '1.457/CD', '9.9/CD', '1.75/CD', '51.25/CD', '22.75/CD', '46.1/CD', '251.2/CD', '278.7/CD', '26.2/CD', 'Va./NNP', 'Tenn./NNP', 'Ariz./NNP', '2.3/CD', '4.1/CD', 'Rev./NNP', 'W.D./NNP', 'C.J.B./NNP', '3.6/CD', 'vs./IN', 'Sino-U.S./NNP', 'I./NNP', 'Wash./NNP', 'W.N./NNP', 'Del./NNP', 'N./NNP', 'Minn./NNP', 'Sr./NNP', 'Mo./NNP', 'Md./NNP', 'La./NNP', '11.6/CD', '23.72/CD', '2.65/CD', '3.80/CD', '3.61/CD', '130.7/CD', '29.9/CD', '29.4/CD', '3.19/CD', 'A.D./NNP', '9.5/CD', '9.3/CD', '27.1/CD', '17.3/CD', 'Ind./NNP', '6.1/CD', 'Prof./NNP', 'R.D./NNP', 'N.Y/NNP', '148.9/CD', '153.3/CD', 'Z./NNP', '3.375/CD', '47.125/CD', '5.7/CD', '2.8/CD', 'U.S.A/NNP', '7.80/CD', '7.55/CD', '8.65/CD', '8.575/CD', '8.06/CD', 'C.D.s/NNS', '8.60/CD', '8.35/CD', '8.48/CD', '8.30/CD', '8.15/CD', '13.50/CD', '4.875/CD', '7.78/CD', '7.62/CD', '9.82/CD', '9.75/CD', '8.70/CD', '8.64/CD', 'S.I./NNP']

    

print(wtProbdict.get('No/DT',0))
print(wtProbdict.get('No/RB',0))

print(wtProbdict.get('But/CC',0))
print(wtProbdict.get('But/IN',0))

print(wtProbdict.get('sound/NN',0))
print(ttProbDict.get('NNP/NN',0))


        

#['NN/RB', 'RB/VBN', 'PRP/RBR', 'RBR/IN', 'VBZ/RB', 'RB/JJ', 'IN/RB', 'VBN/RBR', 'NNS/RB', 'VBD/RB', 'RB/IN', 'RB/JJR', 'RB/MD', 'NN/RBR', 'RBR/RB', 'RB/VB', 'RBR/JJ', 'VBP/RBR', 'RB/DT', 'CD/RB', 'JJ/WRB', 'WRB/DT', 'CC/RB', 'RB/VBD', 'VB/RB', 'JJ/RB', 'RB/TO', 'RP/RB', 'RB/NN', 'VBP/RB', 'RB/VBG', 'VBG/RB', 'RB/CC', 'NNP/RB', 'RB/NNP', 'WRB/PRP', 'RB/VBP', 'RB/RB', 'MD/RB', 'NN/WRB', 'WRB/NN', 'PRP/RB', 'RB/PRP', 'RB/CD', 'JJS/RB', 'WDT/RB', 'RB/VBZ', 'VBZ/RBR', 'RBR/VBN', 'DT/RB', 'IN/RBR', 'VBN/WRB', 'WRB/JJ', 'VBP/WRB', 'WRB/NNP', 'VBN/RB', 'DT/RBR', 'VBD/RBR', 'TO/RBR', 'NNP/WRB', 'WRB/RP', 'VBZ/WRB', 'WRB/RB', 'NNS/RBS', 'RBS/IN', 'TO/RB', 'VB/RBR', 'RB/RBR', 'RBR/VB', 'NNS/WRB', 'NNS/RBR', 'WP/RB', 'WRB/IN', 'IN/WRB', 'JJ/RBR', 'DT/RBS', 'RBS/JJ', 'WRB/NNS', 'RBR/DT', 'RBR/NNP', 'RB/NNS', 'RB/WRB', 'RBR/WRB', 'CC/RBR', 'RB/WDT', 'WRB/VBZ', 'VBD/WRB', 'CC/WRB', 'CC/RBS', 'RB/WP', 'RBS/RB', 'WRB/VBN', 'WDT/RBR', 'RBR/VBD', 'CD/WRB', 'EX/RB', 'VB/WRB', 'RB/EX', 'RBR/PRP', 'VBG/RBR', 'RBR/VBG', 'WRB/RBS', 'LS/RB', 'RBR/TO', 'RBR/CC', 'POS/RBS', 'VB/RBS', 'PRP/RBS', 'WRB/VBG', 'NN/RBS', 'RBR/VBP', 'VBN/RBS', 'NNPS/RB', 'WRB/VBP', 'RBR/NN', 'WRB/TO', 'WRB/VBD', 'JJR/RB', 'RB/JJS', 'WRB/MD', 'CD/RBR', 'RBR/VBZ']
tagtag =[]
ttprob = []

f =''
ft =''
td = ['RB', 'DT']
print('element tag:',td)
for element in td:
    u = element+'/'
    j = [k for k,v in ttProbDict.items() if u in k]
    for n in j:
        f = n.split('/')[0]
        ft = n.split('/')[1]
        for element in td:
            if (element == f):
                h = f+'/'+ft
                tagtag.append(h)
                ttprob.append(ttProbDict.get(h,0))
print('ttpair------------------------')
print(tagtag)
print(len(tagtag))
#['RB/VBN', 'RB/JJ', 'RB/IN', 'RB/JJR', 'RB/MD', 'RB/VB', 'RB/DT', 'RB/VBD', 'RB/TO', 'RB/NN', 'RB/VBG', 'RB/CC', 'RB/NNP', 'RB/VBP', 'RB/RB', 'RB/PRP', 'RB/CD', 'RB/VBZ', 'RB/RBR', 'RB/NNS', 'RB/WRB', 'RB/WDT', 'RB/WP', 'RB/EX', 'RB/JJS', 'DT/NN', 'DT/JJ', 'DT/NNP', 'DT/NNS', 'DT/JJS', 'DT/VBZ', 'DT/VBN', 'DT/CD', 'DT/IN', 'DT/WP', 'DT/JJR', 'DT/VBG', 'DT/TO', 'DT/NNPS', 'DT/RB', 'DT/VBD', 'DT/RBR', 'DT/CC', 'DT/PRP', 'DT/DT', 'DT/MD', 'DT/FW', 'DT/RBS', 'DT/McGraw', 'DT/VBP', 'DT/Contra', 'DT/WDT']
print('ttpair probability------------')
print(ttprob)
print(len(ttprob))
#[0.09472435736605615, 0.09627534612138768, 0.14583856031749654, 0.008119882307323861, 0.014574732568482997, 0.08067422393540588, 0.09299089934539151, 0.0838218187624022, 0.021668225258308055, 0.027644093697967748, 0.03683598293912369, 0.01626257327281436, 0.01966106333964373, 0.03722373012795657, 0.08106197112423875, 0.015236183655315557, 0.045959446205779714, 0.05029309125744132, 0.007230344638824898, 0.014483497935816436, 0.003147594826996328, 0.0017790753369979243, 0.001254476199165203, 0.002896699587163287, 0.0003421298724996008, 0.47568659304954886, 0.19953458760510198, 0.1328254776697182, 0.08197405106364294, 0.007991043006739699, 0.006205498598640352, 0.009088713749423723, 0.02536351196095219, 0.010940118402084111, 0.0007390983000739098, 0.0076910130037393986, 0.013955054041989564, 0.0007171448852202293, 0.00223924831507541, 0.010398600835693326, 0.0011562131822938392, 0.001975807336831244, 0.0008122763495861782, 0.0020416675813922855, 0.001719684163538305, 0.001895311482367749, 0.0011123063525864782, 0.0031320205191250833, 0.00021221634358557808, 0.00033661902775643417, 0.00019026292873189758, 6.586024456104147e-05]        


wtg = [k for k,v in wtProbdict.items() if 'No/' in k]
print("wtg pairs -> ",wtg)
#wtg pairs ->  ['No/DT', 'No/RB']

wtgp = []
for p in wtg:
    wtgp.append(wtProbdict.get(p,0))    
#    print('probability ->',wtProbdict.get(p,0))
print(wtgp)
#[0.0016273393002441008, 0.0005624296962879641]


maxm = 0
prevmax = 0
index = 0
pairindex = 0
for i in range(len(wtg)):
#    print("------------------")
#    print("outerloop", i)
    k = wtg[i]
    l = k.split('/')[1]
#    print("l ->",l)
#    print("pairindex", pairindex)
    for n in range(len(tagtag)):
#        print("innerloop",n)
        u = tagtag[n]
        v = u.split('/')[0]
        if(l == v):
#            print(wtg[i],'X',tagtag[n])
            maxm = ttprob[n]*wtgp[i]

            if(maxm >= prevmax):
                prevmax = maxm
#                print("new max: index",n,"pairvalue",i,prevmax)
                tagged = tagtag[n]
#                print(tagged)
                words = wtg[i]
#                print(words)
                
            else:
                prevmax = prevmax

print("final max:",prevmax)
print(tagged)
print(words)

