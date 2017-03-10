import csv;

counts = [0, 0, 0, 0, 0, 0, 0, 0]
total = 0

with open('train.csv', 'rb') as train_csv:
	reader = csv.reader(train_csv, delimiter=',')
	for row in reader:
		if (row[0] != 'Id'):
			total += 1
			ind = int(row[1])-1
 			counts[ind] += 1

for i in range (0, len(counts)):
	counts[i] = float(counts[i] * 100) / total 
        bar = '=' * int(counts[i]) 
	print "Class %d: %s (%f%%)" % (i+1, bar, counts[i]) 
