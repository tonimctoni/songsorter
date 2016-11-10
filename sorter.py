#300->200->100 network
import re
from os.path import join
from random import random
filenames=['blink-182.txt', 'queen.txt', 'the_beatles.txt', 'eminem.txt', 'nightwish.txt',
'frank_sinatra.txt','britney_spears.txt', 'shakira.txt', 'epica.txt', 'metallica.txt']
allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789 \n\"`'$()/?[].:,;!-"

# &quot;
prog1=re.compile("""\&.{2,10}?;""")
prog2=re.compile("""\s*\|\s*""")
prog3=re.compile("""################--> .*? <--################""")

for filename in filenames:
    with open(filename) as f:
        content=f.read()
        content=content.lower()
        content=content.replace("&quot;", "\"")
        content=prog1.sub("", content)
        content=prog2.sub("\n", content)
    contents=prog3.split(content)
    for i in range(len(contents)):
        old_content=contents[i].strip()
        contents[i]=""
        for c in old_content:
            if not c in allowed_chars: continue
            contents[i]+=c
    del contents[0]
    ntrain=0
    ntest=0
    with open(join("training", filename), "w") as ftrain:
        with open(join("testing", filename), "w") as ftest:
            for content in contents:
                if random()<.2:
                    ftest.write(content)
                    ftest.write("#"*32)
                    ntest+=1
                else:
                    ftrain.write(content)
                    ftrain.write("#"*32)
                    ntrain+=1
    print "Test:", ntest
    print "Train:", ntrain
    print "Train prop:", ntrain/float(ntest+ntrain)
    print ""


# Test: 30
# Train: 116
# Train prop: 0.794520547945

# Test: 41
# Train: 140
# Train prop: 0.773480662983

# Test: 62
# Train: 232
# Train prop: 0.789115646259

# Test: 69
# Train: 228
# Train prop: 0.767676767677

# Test: 17
# Train: 91
# Train prop: 0.842592592593

# Test: 162
# Train: 511
# Train prop: 0.759286775632

# Test: 42
# Train: 176
# Train prop: 0.807339449541

# Test: 27
# Train: 109
# Train prop: 0.801470588235

# Test: 15
# Train: 84
# Train prop: 0.848484848485

# Test: 26
# Train: 118
# Train prop: 0.819444444444