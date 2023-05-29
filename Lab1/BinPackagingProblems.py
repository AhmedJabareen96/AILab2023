my_file = open("text files/binpack1.txt","r")

my_data = my_file.read()
my_data = my_data.split("\n")

number_of_test_problems = my_data[0]
problem_identifier = my_data[1]
bin_capacity = my_data[2].split(" ")[1]
number_of_items = my_data[2].split(" ")[2]
number_of_bins_best_sol = my_data[2].split(" ")[2]

items = list()
prob_num = 0
for i in my_data[3+prob_num:int(number_of_items)+3]:
    items.append(int(i))

print(items)

my_file.close()
