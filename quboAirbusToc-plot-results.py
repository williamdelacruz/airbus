import os
import matplotlib.pyplot as plt
import pickle





# Read file using pickle
#
def read_energies(filename):

    list_energies = {}

    print('{}'.format(filename))

    if  os.path.exists(filename)==False:
        print('File {} does not exists'.format(filename))
    else:
        # Load existing energies and solutions
        pfile = open(filename, 'rb')
        list_energies = pickle.load(pfile)
        pfile.close();
        
    return list_energies




# - - - - - - - - - - - - - - - - - - #
#     Fin de las definiciones         #
# - - - - - - - - - - - - - - - - - - #


def main():

    # Path to instances
    #
    path = "./alo//resultados-qubo/5-5-5/"


    for iter1 in range(10):
        # name of the instances
        fname = "i" + str(iter1+1) + "-5-5-5-energies-gravity-shear"
    
        # read file
        list_energies = read_energies(path+fname)
    
        
        # Plot energies
        #
        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Solution')
        plt.ylabel('Energy')
        plt.title(fname + " (obj1 + payload + gravity + shear)")

        last_n = len(list_energies)
        plt.scatter([i for i in range(last_n)], list_energies, marker='o', color='blue')
        plt.show()
        fig1.savefig(path+fname+'.pdf')



if __name__ == "__main__":
    main()


