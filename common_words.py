good = 'correct_answers_1'
bad = 'mistake_answers_1'
from collections import defaultdict

with open(good, 'r') as f:
	good_lines = [line.strip().split(' ') for line in f]
with open(bad, 'r') as f:
	bad_lines = [line.strip().split(' ') for line in f]

words = set()
for line in good_lines + bad_lines:
	for word in line:
		words.add(word)

word2freq_good = defaultdict(lambda : 0.0)
word2freq_bad = defaultdict(lambda : 0.0)


for line in good_lines:
	ws = set(line)
	for word in ws:
		word2freq_good[word] += 1.0

for line in bad_lines:
	ws = set(line)
	for word in ws:
		word2freq_bad[word] += 1.0

for key in word2freq_good:
	word2freq_good[key] /= len(good_lines)
for key in word2freq_bad:
	word2freq_bad[key] /= len(bad_lines)

#print(sorted(word2freq_good.items(), key = lambda x: x[1])[-10:])
print()
#print(sorted(word2freq_bad.items(), key = lambda x: x[1])[-10:])
print()

word2weight = defaultdict(lambda : 0.0)
for i in good_lines:
	l = len(i)
	for word in i:
		word2weight[word] += 1.0/l/len(good_lines)
for i in bad_lines:
	l = len(i)
	for word in i:
		word2weight[word] -= 1.0/l/len(bad_lines)


ranked = sorted(word2weight.items(), key = lambda x: x[1])
for i in ranked[:20]:
	print("'%s': %.3f," %(i[0], i[1]*1000))
print()
for i in ranked[-20:]:
	print("'%s': %.3f," %(i[0], i[1]*1000))

#print(len(good_lines))
#print(good_lines[0:5])
