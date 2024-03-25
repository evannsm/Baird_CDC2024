import blimp
import numpy as np
my_blimp = blimp.Blimp()
my_blimp.setup_discrete_lti_model(0.25)
A_discrete = my_blimp.A_discrete
B_discrete = my_blimp.B_discrete
A = my_blimp.jacobian_np(np.zeros((12,1)))
B = my_blimp.B


# Print out A for a smallmatrix environment in Latex.
print(r'\begin{bsmallmatrix}')
for row in A:
    # Print out each row in a latex format.
    # If the element is zero, print out '0' instead of '0.0'.
    # If the element is less than 1, do not print out the leading zero.
    for element in row:
        if element == 0:
            print('0', end=' & ')
        elif element < 1 and element > 0:
            # Print out the element with 4 decimal places, but do not print out the leading zero.
            print(f'{element:.4f}'[1:], end=' & ')
        elif element < 0 and element > -1:
            # Print out the element with 4 decimal places, but do not print out the leading zero. So skip the second element of the string.
            print(f'{element:.4f}'[0] + f'{element:.4f}'[2:], end=' & ')
        elif element == 1:
            print('1', end=' & ')
        else:
            print(f'{element:.4f}', end=' & ')
    print(r'\\') # End the row.
print(r'\end{bsmallmatrix}')


# Do this for B as well.
print(r'\begin{bsmallmatrix}')
for row in B:
    # Print out each row in a latex format.
    # If the element is zero, print out '0' instead of '0.0'.
    for element in row:
        if element == 0:
            print('0', end=' & ')
        elif element < 1 and element > 0:
            # Print out the element with 4 decimal places, but do not print out the leading zero.
            print(f'{element:.4f}'[1:], end=' & ')
        elif element < 0 and element > -1:
            # Print out the element with 4 decimal places, but do not print out the leading zero. So skip the second element of the string.
            print(f'{element:.4f}'[0] + f'{element:.4f}'[2:], end=' & ')
        elif element == 1:
            print('1', end=' & ')
        else:
            print(f'{element:.4f}', end=' & ')

    print(r'\\') # End the row.
print(r'\end{bsmallmatrix}')
print(B < 0)

# Finally, do this for a matrix G.
G = np.block([[np.eye(6)],
              [np.zeros((6,6))]])
print(r'\begin{bsmallmatrix}')
for row in G:
    # Print out each row in a latex format.
    # If the element is zero, print out '0' instead of '0.0'.
    for element in row:
        if element == 0:
            print('0', end=' & ')
        else:
            print(f'{element:.4f}', end=' & ')
    print(r'\\') # End the row.
print(r'\end{bsmallmatrix}')
