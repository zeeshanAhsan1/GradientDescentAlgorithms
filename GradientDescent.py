import numpy as np
import matplotlib.pyplot as plt

Y = []
for i in range(10):
        k = np.random.normal(size=[10,1])
        Y.append(k)
A = np.concatenate((Y[0],Y[1],Y[2],Y[3],Y[4],Y[5],Y[6],Y[7],Y[8],Y[9]),axis = 1)
C = np.random.normal(size=[10,1])
x_start = np.random.normal(size = [10,1])


def vanilla_gradient_descent(A,C,x_start):


    A_transpose = A.transpose()
    Q = np.matmul(A_transpose,A)

    gradient = np.subtract(np.matmul(Q,x_start),C)
    Q_inverse = np.linalg.inv(Q)
    x_minima = np.matmul(Q_inverse,C)
    alfa = 0.04


    error = np.linalg.norm(np.subtract(x_start,x_minima))
    print("Error : ", error)
    x_old = x_start

    count = 0
    error_arr = []
    x_val_arr = []
    angle_arr = []
    while error > 0.01:

        count +=1
        gradient = np.subtract(np.matmul(Q,x_old),C)
        decrement = np.dot(alfa, gradient)
        x_new = np.subtract(x_old,decrement)
        
        error = np.linalg.norm(np.subtract(x_new,x_minima))


        #Angle Question
        nr_fact1 = np.subtract(x_new,x_old)
        nr_fact1_transpose = nr_fact1.transpose()
        nr_fact2 = np.subtract(x_minima,x_old)
        numrtr = np.matmul(nr_fact1_transpose,nr_fact2)
        dn_fact1 = np.linalg.norm(np.subtract(x_new,x_old))
        dn_fact2 = np.linalg.norm(np.subtract(x_minima,x_old))
        dn = dn_fact1 * dn_fact2
        angle = numrtr/dn



        x_old = x_new
        error_arr.append(error)
        angle_arr.append(angle[0][0])
        x_val_arr.append(count)
        print("Angle : ", angle)
        print("Error : ", error)

    print("Steps : ", count)

    plt.plot(x_val_arr,error_arr)
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.title("Vanilla Gradient Descent")
    plt.show()

    plt.plot(x_val_arr,angle_arr)
    plt.xlabel("Steps")
    plt.ylabel("Angle")
    plt.title("Angle in Vanilla Gradient Descent")
    plt.show()

    return count


def grad_descent_opt_alfa(A,C,x_start):
    

    A_transpose = A.transpose()
    Q = np.matmul(A_transpose,A)

    gradient = np.subtract(np.matmul(Q,x_start),C)
    Q_inverse = np.linalg.inv(Q)
    x_minima = np.matmul(Q_inverse,C)

    #Optimal alfa calculation
    p_k = gradient
    p_k_transpose = p_k.transpose()

    alfa_nr = np.matmul(p_k_transpose,p_k)
    alfa_dr1 = np.matmul(p_k_transpose,Q)
    alfa_dr = np.matmul(alfa_dr1,p_k)
    # print("Alfa nr ", alfa_nr)
    # print("Alfa dr ", alfa_dr)

    
    alfa = alfa_nr / alfa_dr
    print("AlFA : ", alfa[0][0])
    #return

    error = np.linalg.norm(np.subtract(x_start,x_minima))
    print("Error : ", error)
    x_old = x_start

    count = 0
    error_arr = []
    x_val_arr = []
    angle_arr = []
    while error > 0.01:

        count +=1
        gradient = np.subtract(np.matmul(Q,x_old),C)


        #Optimal alfa calculation
        p_k = gradient
        p_k_transpose = p_k.transpose()

        alfa_nr = np.matmul(p_k_transpose,p_k)
        alfa_dr1 = np.matmul(p_k_transpose,Q)
        alfa_dr = np.matmul(alfa_dr1,p_k)
        alfa = alfa_nr / alfa_dr
        print("ALFA : ", alfa[0][0])
        decrement = np.dot(alfa[0][0], gradient)
        x_new = np.subtract(x_old,decrement)
        
        error = np.linalg.norm(np.subtract(x_new,x_minima))

        #Angle Question
        nr_fact1 = np.subtract(x_new,x_old)
        nr_fact1_transpose = nr_fact1.transpose()
        nr_fact2 = np.subtract(x_minima,x_old)
        numrtr = np.matmul(nr_fact1_transpose,nr_fact2)
        dn_fact1 = np.linalg.norm(np.subtract(x_new,x_old))
        dn_fact2 = np.linalg.norm(np.subtract(x_minima,x_old))
        dn = dn_fact1 * dn_fact2
        angle = numrtr/dn

        x_old = x_new
        error_arr.append(error)
        angle_arr.append(angle[0][0])
        x_val_arr.append(count)
        print("Angle : ", angle)
        print("Error : ", error)
    

    plt.plot(x_val_arr,error_arr)
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.title("Optimal Alfa Gradient Descent")
    plt.show()

    plt.plot(x_val_arr,angle_arr)
    plt.xlabel("Steps")
    plt.ylabel("Angle")
    plt.title("Angle in Optimal Alfa Gradient Descent")
    plt.show()

    return count

def momentum_grad_descent(A,C,x_start):

    A_transpose = A.transpose()
    Q = np.matmul(A_transpose,A)

    gradient = np.subtract(np.matmul(Q,x_start),C)
    Q_inverse = np.linalg.inv(Q)
    x_minima = np.matmul(Q_inverse,C)
    alfa = 0.04
    beta = 0.05

    error = np.linalg.norm(np.subtract(x_start,x_minima))
    print("Error : ", error)
    x0 = 0
    x1 = x_start
    q_k = np.subtract(x1,x0)

    count = 0
    error_arr = []
    x_val_arr = []
    angle_arr = []
    count = 0
    while ( error > 0.01):
        count += 1

        gradient = np.subtract(np.matmul(Q,x1),C)
        x_new = np.add((np.subtract(x1,np.dot(alfa,gradient))), np.dot(beta,np.subtract(x1,x0)))
        error =  np.linalg.norm(np.subtract(x_new,x_minima))

        #Angle Question
        nr_fact1 = np.subtract(x_new,x1)
        nr_fact1_transpose = nr_fact1.transpose()
        nr_fact2 = np.subtract(x_minima,x1)
        numrtr = np.matmul(nr_fact1_transpose,nr_fact2)
        dn_fact1 = np.linalg.norm(np.subtract(x_new,x1))
        dn_fact2 = np.linalg.norm(np.subtract(x_minima,x1))
        dn = dn_fact1 * dn_fact2
        angle = numrtr/dn
        print("Angle :", angle)

        x0 = x1
        x1 = x_new
        
        error_arr.append(error)
        x_val_arr.append(count)
        angle_arr.append(angle[0][0])
        print("Error : ", error)


    plt.plot(x_val_arr,error_arr)
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.title("Momentum Gradient Descent")
    plt.show()


    plt.plot(x_val_arr,angle_arr)
    plt.xlabel("Steps")
    plt.ylabel("Angle")
    plt.title("Angle in Momentum Gradient Descent")
    plt.show()


    return count


def momentum_grad_descent_opt_alfa_beta(A, C, x_start):

    A_transpose = A.transpose()
    Q = np.matmul(A_transpose,A)

    gradient = np.subtract(np.matmul(Q,x_start),C)
    Q_inverse = np.linalg.inv(Q)
    x_minima = np.matmul(Q_inverse,C)
    #p_k = gradient

    x0 = 0
    x1 = x_start

    p_k = gradient
    q_k = np.subtract(x1,x0)

    n1 = np.matmul(q_k.transpose(),p_k)
    n2 = np.matmul(p_k.transpose(),Q)
    n3 = np.matmul(n2,q_k)
    n_f1 = np.matmul(n1,n3)
    n4 = np.matmul(p_k.transpose(), p_k)
    n5 = np.matmul(q_k.transpose(),Q)
    n6 = np.matmul(n5,q_k)
    n_f2 = np.matmul(n4,n6)

    alfa_nr = np.subtract(n_f1,n_f2)

    d1 = np.matmul(p_k.transpose(),Q)
    d2 = np.matmul(d1,q_k)
    d_f1 = np.matmul(d2,d2)
    d3 = np.matmul(p_k.transpose(),Q)
    d4 = np.matmul(d3,p_k)
    d5 = np.matmul(q_k.transpose(),Q)
    d6 = np.matmul(d5,q_k)
    d_f2 = np.matmul(d4,d6)

    alfa_dr = np.subtract(d_f1,d_f2)
        
    alfa = alfa_nr/alfa_dr

    n1 = np.matmul(p_k.transpose(),p_k)
    n2 = np.matmul(p_k.transpose(),Q)
    n3 = np.matmul(n2,q_k)
    n_f1 = np.matmul(n1,n3)
    n4 = np.matmul(q_k.transpose(),p_k)
    n5 = np.matmul(p_k.transpose(),Q)
    n6 = np.matmul(n5,p_k)
    n_f2 = np.matmul(n4,n6)

    beta_nr = np.subtract(n_f1,n_f2)

    d1 = np.matmul(q_k.transpose(),Q)
    d2 = np.matmul(d1,q_k)
    d3 = np.matmul(p_k.transpose(),Q)
    d4 = np.matmul(d3,p_k)
    d_f1 = np.matmul(d2,d4)
    d5 = np.matmul(p_k.transpose(),Q)
    d6 = np.matmul(d5,q_k)
    d_f2 = np.matmul(d6,d6)
        
    beta_dr = np.subtract(d_f1,d_f2)
    beta = beta_nr/beta_dr        

    error = np.linalg.norm(np.subtract(x_start,x_minima))
    print("Error : ", error)


    count = 0
    error_arr = []
    x_val_arr = []
    angle_arr = []
    count = 0

    while error > 0.01:

        count += 1

        gradient = np.subtract(np.matmul(Q,x1),C)
        p_k = gradient
        q_k = np.subtract(x1,x0)

        n1 = np.matmul(q_k.transpose(),p_k)
        n2 = np.matmul(p_k.transpose(),Q)
        n3 = np.matmul(n2,q_k)
        n_f1 = np.matmul(n1,n3)
        n4 = np.matmul(p_k.transpose(), p_k)
        n5 = np.matmul(q_k.transpose(),Q)
        n6 = np.matmul(n5,q_k)
        n_f2 = np.matmul(n4,n6)

        alfa_nr = np.subtract(n_f1,n_f2)

        d1 = np.matmul(p_k.transpose(),Q)
        d2 = np.matmul(d1,q_k)
        d_f1 = np.matmul(d2,d2)
        d3 = np.matmul(p_k.transpose(),Q)
        d4 = np.matmul(d3,p_k)
        d5 = np.matmul(q_k.transpose(),Q)
        d6 = np.matmul(d5,q_k)
        d_f2 = np.matmul(d4,d6)

        alfa_dr = np.subtract(d_f1,d_f2)
        
        alfa = alfa_nr/alfa_dr

        n1 = np.matmul(p_k.transpose(),p_k)
        n2 = np.matmul(p_k.transpose(),Q)
        n3 = np.matmul(n2,q_k)
        n_f1 = np.matmul(n1,n3)
        n4 = np.matmul(q_k.transpose(),p_k)
        n5 = np.matmul(p_k.transpose(),Q)
        n6 = np.matmul(n5,p_k)
        n_f2 = np.matmul(n4,n6)

        beta_nr = np.subtract(n_f1,n_f2)

        d1 = np.matmul(q_k.transpose(),Q)
        d2 = np.matmul(d1,q_k)
        d3 = np.matmul(p_k.transpose(),Q)
        d4 = np.matmul(d3,p_k)
        d_f1 = np.matmul(d2,d4)
        d5 = np.matmul(p_k.transpose(),Q)
        d6 = np.matmul(d5,q_k)
        d_f2 = np.matmul(d6,d6)
        
        beta_dr = np.subtract(d_f1,d_f2)
        beta = beta_nr/beta_dr
        
        print("Alfa : ", alfa)
        print("Beta : ", beta)
        x_new = np.add((np.subtract(x1,np.dot(alfa[0][0],gradient))), np.dot(beta[0][0],np.subtract(x1,x0)))
        error =  np.linalg.norm(np.subtract(x_new,x_minima))
        error_arr.append(error)

        #Angle Question
        nr_fact1 = np.subtract(x_new,x1)
        nr_fact1_transpose = nr_fact1.transpose()
        nr_fact2 = np.subtract(x_minima,x1)
        numrtr = np.matmul(nr_fact1_transpose,nr_fact2)
        dn_fact1 = np.linalg.norm(np.subtract(x_new,x1))
        dn_fact2 = np.linalg.norm(np.subtract(x_minima,x1))
        dn = dn_fact1 * dn_fact2
        angle = numrtr/dn
        print("Angle :", angle)
        angle_arr.append(angle[0][0])

        x0 = x1
        x1 = x_new

        x_val_arr.append(count)
        print("Error : ", error)
    


    plt.plot(x_val_arr,error_arr)
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.title("Momentum Opt Gradient Descent With Optimal Alfa-Beta")
    plt.show()


    plt.plot(x_val_arr,angle_arr)
    plt.xlabel("Steps")
    plt.ylabel("Angle")
    plt.title("Angle in Momentum Gradient Descent With Optimal Alfa-Beta")
    plt.show()

    return count



def get_orthogonal_qk(p_k):

    q_k = p_k.copy()

    for el in q_k:
        # print("El = ", el)
        el[0] = 0
    
    var1 = p_k[0][0]
    q_k[0][0] = var1
    q_k[1][0] = -(p_k[0][0] * p_k[0][0])/p_k[1][0]

    return q_k



def orthogonal_grad_descent(A, C, x_start):
    A_transpose = A.transpose()
    Q = np.matmul(A_transpose,A)

    gradient = np.subtract(np.matmul(Q,x_start),C)
    Q_inverse = np.linalg.inv(Q)
    x_minima = np.matmul(Q_inverse,C)
    #p_k = gradient

    x0 = 0
    x1 = x_start

    p_k = gradient
    q_k = get_orthogonal_qk(p_k)
    print("Q_k : ", q_k)

    print("Dot Product of p_k and q_k :", np.dot(p_k.transpose(),q_k))

    n1 = np.matmul(q_k.transpose(),p_k)
    n2 = np.matmul(p_k.transpose(),Q)
    n3 = np.matmul(n2,q_k)
    n_f1 = np.matmul(n1,n3)
    n4 = np.matmul(p_k.transpose(), p_k)
    n5 = np.matmul(q_k.transpose(),Q)
    n6 = np.matmul(n5,q_k)
    n_f2 = np.matmul(n4,n6)

    alfa_nr = np.subtract(n_f1,n_f2)

    d1 = np.matmul(p_k.transpose(),Q)
    d2 = np.matmul(d1,q_k)
    d_f1 = np.matmul(d2,d2)
    d3 = np.matmul(p_k.transpose(),Q)
    d4 = np.matmul(d3,p_k)
    d5 = np.matmul(q_k.transpose(),Q)
    d6 = np.matmul(d5,q_k)
    d_f2 = np.matmul(d4,d6)

    alfa_dr = np.subtract(d_f1,d_f2)
        
    alfa = alfa_nr/alfa_dr

    n1 = np.matmul(p_k.transpose(),p_k)
    n2 = np.matmul(p_k.transpose(),Q)
    n3 = np.matmul(n2,q_k)
    n_f1 = np.matmul(n1,n3)
    n4 = np.matmul(q_k.transpose(),p_k)
    n5 = np.matmul(p_k.transpose(),Q)
    n6 = np.matmul(n5,p_k)
    n_f2 = np.matmul(n4,n6)

    beta_nr = np.subtract(n_f1,n_f2)

    d1 = np.matmul(q_k.transpose(),Q)
    d2 = np.matmul(d1,q_k)
    d3 = np.matmul(p_k.transpose(),Q)
    d4 = np.matmul(d3,p_k)
    d_f1 = np.matmul(d2,d4)
    d5 = np.matmul(p_k.transpose(),Q)
    d6 = np.matmul(d5,q_k)
    d_f2 = np.matmul(d6,d6)
        
    beta_dr = np.subtract(d_f1,d_f2)
    beta = beta_nr/beta_dr        

    error = np.linalg.norm(np.subtract(x_start,x_minima))
    print("Error : ", error)


    count = 0
    error_arr = []
    x_val_arr = []
    angle_arr = []
    count = 0

    while error > 0.01:

        count += 1

        gradient = np.subtract(np.matmul(Q,x1),C)
        p_k = gradient
        q_k = get_orthogonal_qk(p_k)

        n1 = np.matmul(q_k.transpose(),p_k)
        n2 = np.matmul(p_k.transpose(),Q)
        n3 = np.matmul(n2,q_k)
        n_f1 = np.matmul(n1,n3)
        n4 = np.matmul(p_k.transpose(), p_k)
        n5 = np.matmul(q_k.transpose(),Q)
        n6 = np.matmul(n5,q_k)
        n_f2 = np.matmul(n4,n6)

        alfa_nr = np.subtract(n_f1,n_f2)

        d1 = np.matmul(p_k.transpose(),Q)
        d2 = np.matmul(d1,q_k)
        d_f1 = np.matmul(d2,d2)
        d3 = np.matmul(p_k.transpose(),Q)
        d4 = np.matmul(d3,p_k)
        d5 = np.matmul(q_k.transpose(),Q)
        d6 = np.matmul(d5,q_k)
        d_f2 = np.matmul(d4,d6)

        alfa_dr = np.subtract(d_f1,d_f2)
        
        alfa = alfa_nr/alfa_dr

        n1 = np.matmul(p_k.transpose(),p_k)
        n2 = np.matmul(p_k.transpose(),Q)
        n3 = np.matmul(n2,q_k)
        n_f1 = np.matmul(n1,n3)
        n4 = np.matmul(q_k.transpose(),p_k)
        n5 = np.matmul(p_k.transpose(),Q)
        n6 = np.matmul(n5,p_k)
        n_f2 = np.matmul(n4,n6)

        beta_nr = np.subtract(n_f1,n_f2)

        d1 = np.matmul(q_k.transpose(),Q)
        d2 = np.matmul(d1,q_k)
        d3 = np.matmul(p_k.transpose(),Q)
        d4 = np.matmul(d3,p_k)
        d_f1 = np.matmul(d2,d4)
        d5 = np.matmul(p_k.transpose(),Q)
        d6 = np.matmul(d5,q_k)
        d_f2 = np.matmul(d6,d6)
        
        beta_dr = np.subtract(d_f1,d_f2)
        beta = beta_nr/beta_dr
        
        print("Alfa : ", alfa)
        print("Beta : ", beta)
        x_new = np.add((np.subtract(x1,np.dot(alfa[0][0],gradient))), np.dot(beta[0][0],np.subtract(x1,x0)))
        error =  np.linalg.norm(np.subtract(x_new,x_minima))
        error_arr.append(error)

        #Angle Question
        nr_fact1 = np.subtract(x_new,x1)
        nr_fact1_transpose = nr_fact1.transpose()
        nr_fact2 = np.subtract(x_minima,x1)
        numrtr = np.matmul(nr_fact1_transpose,nr_fact2)
        dn_fact1 = np.linalg.norm(np.subtract(x_new,x1))
        dn_fact2 = np.linalg.norm(np.subtract(x_minima,x1))
        dn = dn_fact1 * dn_fact2
        angle = numrtr/dn
        print("Angle :", angle)
        angle_arr.append(angle[0][0])

        x0 = x1
        x1 = x_new
        x_val_arr.append(count)
        print("Error : ", error)

    plt.plot(x_val_arr,error_arr)
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.title("Orthogonal Momentum Gradient Descent With Optimal Alfa-Beta")
    plt.show()
    
    plt.plot(x_val_arr,angle_arr)
    plt.xlabel("Steps")
    plt.ylabel("Angle")
    plt.title("Angle in Orthogonal Momentum Gradient Descent with Optimal Alfa-Beta")
    plt.show()

    return count    



# steps = orthogonal_grad_descent(A, C, x_start)
# print("Orthogonal Grad Descent Steps : ", steps)


vanilla_ds = vanilla_gradient_descent(A, C, x_start)
opt_ds = grad_descent_opt_alfa(A, C, x_start)
momentum_ds = momentum_grad_descent(A, C, x_start)
momentum_opt = momentum_grad_descent_opt_alfa_beta(A, C, x_start)
orthogonal_descent = orthogonal_grad_descent(A, C, x_start)
print("vanilla_ds : ",vanilla_ds)
print("opt_ds : ",opt_ds)
print("momentum_ds : ",momentum_ds)
print("momentum_opt : ",momentum_opt)
print("Orthogonal Descent Steps : ", orthogonal_descent)



# steps_rand_alfa = random_alfa(A, C, x_start)
# ans = momentum_grad_descent(A, C, x_start)

# print("steps_rand_alfa : ", steps_rand_alfa)
# print("Steps momentum : ", ans)





