def print_headers(file):
    with open(file, 'r') as f:
        headers = f.readline().split('[')[0].strip().split('#')[1] # extract headers, removing the '#' and bracketed comment
        print(f'Column headers: {headers}')

def count_rows(file):
    with open(file, 'r') as f:
        next(f) 
        row_count = sum(1 for line in f)
        print(f'Number of data rows (excluding header/comments): {row_count}')

def midpoint_redshift(file):
    with open(file, 'r') as f:
        next(f) 
        for line in f:
            data = line.strip().split(' ') 
            z = float(data[0]) 
            x_HI = float(data[1]) 
            if x_HI <= 0.5:
                if prev_x_HI-0.5 < 0.5-x_HI: # print the redshift that is closer to the midpoint of reionization
                    print(f'Midpoint of reionization occurs at redshift: {prev_z}')
                else: 
                    print(f'Midpoint of reionization occurs at redshift: {z}')
                break
            prev_z = z
            prev_x_HI = x_HI

reion_file = 'data/reion_history_Thesan1.dat'

print('File: ' + reion_file)
print_headers(reion_file)
count_rows(reion_file)
midpoint_redshift(reion_file)

sfrd_file = 'data/sfrd_Thesan1.dat'

print('File: ' + sfrd_file)
print_headers(sfrd_file)
count_rows(sfrd_file)
