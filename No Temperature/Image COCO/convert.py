import cPickle
data_Name = "cotra"
vocab_file = "vocab_" + data_Name + ".pkl"

word, vocab = cPickle.load(open('save/'+vocab_file))
print len(word)
input_file = 'save/generator_sample.txt'
# input_file = 'save/coco_451.txt'
output_file = 'speech/' + data_Name + '_' + input_file.split('_')[-1]
with open(output_file, 'w')as fout:
    with open(input_file)as fin:
        for line in fin:
            #line.decode('utf-8')
            line = line.split()
            #line.pop()
            #line.pop()
            line = [int(x) for x in line]
            line = [word[x] for x in line]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)#.encode('utf-8'))
