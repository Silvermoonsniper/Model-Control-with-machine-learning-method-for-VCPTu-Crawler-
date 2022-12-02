
#newton method to obtain numerical solution of model control output


def newton_control_solver(starting_point,desired_vel,parameter_vector):
    
    #starting point,
    control = starting_point
    iteration = 0
    func_val = parameter_vector[0]*control**3 + parameter_vector[1]*control**2 + parameter_vector[2]*control + parameter_vector[3]
 
    numerical_error = abs(func_val-desired_vel)
    for j in range(1000):
    #function value
        func_val = parameter_vector[0]*control**3 + parameter_vector[1]*control**2 + parameter_vector[2]*control - desired_vel + parameter_vector[3]
     #gradient val
        gradient_val = 3*parameter_vector[0]*control**2 + 2*parameter_vector[1]*control + 1*parameter_vector[2]
        control = control - (func_val /gradient_val)
        #solution error
        numerical_error = abs(func_val)
        
        iteration += 1
     #   print(iteration,numerical_error)
        if (numerical_error <= 0.00001):
           
            newcontrol = control
          
            return newcontrol

# use stochastic gradient descent to train model
 #function to calculate predicted distance at certain time step
def predicted_controlleroutput(dist_MTSdata,total_shift, local_timestep,predicted_velarray):
   
    
    predicted_distance = dist_MTSdata[local_timestep-total_shift] + np.sum(predicted_velarray)
    
    return predicted_distance

#desired distance

def desired_dist(counter,smoothed_dist,total_shift, step_startcontrol, op_amplitude,op_freq, sampling_freq, op_back_vel):
   
    
    desired_distance = smoothed_dist[step_startcontrol] + op_amplitude*np.sin(2*np.pi*(op_freq*counter/sampling_freq) ) + op_back_vel* counter
    return desired_distance

# SGD optimizer for linear model

def LinearSGD_optimizer(learning_rate,batchsize,local_vel,parameter_vector,current_control):

    w_1 =  parameter_vector[0]
    w_2 =  parameter_vector[1]
    w_3 =  parameter_vector[2]
    w_4 =  parameter_vector[3]
    
#calculate gradient of error function

    gradient_subterm_w3 = local_vel*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)
    gradient_subterm_w4 = w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control
  
#update parameter via SGD
    parameter_vector[2] = parameter_vector[2] - 2*learning_rate*gradient_subterm_w3
    parameter_vector[3] = parameter_vector[3] - 2*learning_rate*gradient_subterm_w4 

    
#calculate control output from model
#design matrix
#    design_matrix = np.array([local_vel**3,local_vel**2,local_vel,1]).flatten()

    model_regout = parameter_vector[0]*local_vel**3 + parameter_vector[1]*local_vel**2 + parameter_vector[2]*local_vel + parameter_vector[3]
    
   # print(model_regout,new)
 #calculate error 
    error = abs((model_regout - current_control))
    
    return parameter_vector,error,model_regout


# SGD optimer for polynomial model

def SGD_optimizer(learning_rate,batchsize,local_vel,parameter_vector,current_control):

    w_1 =  parameter_vector[0]
    w_2 =  parameter_vector[1]
    w_3 =  parameter_vector[2]
    w_4 =  parameter_vector[3]
    
#calculate gradient of error function

    gradient_subterm_w1 = local_vel**3*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)

    gradient_subterm_w2 = local_vel**2*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)
    gradient_subterm_w3 = local_vel*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)
    gradient_subterm_w4 = w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control
  
#update parameter via SGD
    parameter_vector[0] = parameter_vector[0] - 2*learning_rate*gradient_subterm_w1
    parameter_vector[1] = parameter_vector[1] - 2*learning_rate*gradient_subterm_w2
    parameter_vector[2] = parameter_vector[2] - 2*learning_rate*gradient_subterm_w3
    parameter_vector[3] = parameter_vector[3] - 2*learning_rate*gradient_subterm_w4 

    
#calculate control output from model
#design matrix
#    design_matrix = np.array([local_vel**3,local_vel**2,local_vel,1]).flatten()

    model_regout = parameter_vector[0]*local_vel**3 + parameter_vector[1]*local_vel**2 + parameter_vector[2]*local_vel + parameter_vector[3]
    
   # print(model_regout,new)
 #calculate error 
    error = abs((model_regout - current_control))
    
    return parameter_vector,error,model_regout

def data_fitting_model_main(counter,step_startcontrol,op_amplitude,op_freq, sampling_freq, op_back_vel, predicted_distance,batchsize,initial_w,iteration,learning_rate,control_data,velocity_data,smoothed_dist):
    
    
    w1_array = []
    w2_array = []
    w3_array = []
    w4_array = []
    
    predicted_distancearray = []
    desired_distance_array = []
    error_array = []
    model_regout_array = []
    new_model_controlarray = []
    current_control_array = []
    vel_array = []
    polynomial_model_regout_array = []
    for j in range(iteration-batchsize):
        if (j == 0):
            parameter_vector = initial_w
            linearparameter_vector = initial_w 
        current_control = control_data[j]
        local_vel = velocity_data[j]
        
        
       
            
        
        
        # calculate desired dist
        if (j >= step_startcontrol):
            desired_distance = desired_dist(counter,smoothed_dist,total_shift, step_startcontrol, op_amplitude,op_freq, sampling_freq, op_back_vel)
            counter = counter + 1 
        else:
            desired_distance = smoothed_dist[j]
            
        #calculate predicted distance   
        total_shift = 37
        if (j > total_shift):
            predicted_velarray = np.array(model_regout_array[j-total_shift:j]).flatten()
            predicted_distance = predicted_controlleroutput(smoothed_dist,total_shift, j,predicted_velarray)
            #calculate velocity deviation
      
            vel_devi = (predicted_distance- smoothed_dist[j- (total_shift)]) / 82
           
        if (j >= step_startcontrol):
            #calculate desired velocity
            desired_vel = (desired_distance - smoothed_dist[j]) / 40
          #  current_control = desired_vel 
            #calculate model contro output
           
            new_model_control = (desired_vel + vel_devi - linearparameter_vector[3]) /linearparameter_vector[2]
            
        else:
            if (j > total_shift):
                new_model_control = (current_control + vel_devi - linearparameter_vector[3]) /linearparameter_vector[2]
            else:
                 new_model_control = (current_control  - linearparameter_vector[3]) /linearparameter_vector[2]
                    
        # parameters update vias SGD         
        
        parameter_vector,error,model_regout = SGD_optimizer(learning_rate,batchsize,local_vel,parameter_vector,current_control)
        # parameters update vias SGD     
        
        linearparameter_vector,error,linearmodel_regout = LinearSGD_optimizer(learning_rate,batchsize,local_vel,linearparameter_vector,current_control)
        
        #calculate numeric modelled controller output from polynomial model
      #  print(parameter_vector)
        if (j > total_shift):
            if (j < step_startcontrol):
                polynomial_model_control = newton_control_solver(2,current_control,parameter_vector)
            else:
                polynomial_model_control = newton_control_solver(2,desired_vel+ vel_devi,parameter_vector)
        else:
            polynomial_model_control = local_vel
        polynomial_model_regout_array.append(polynomial_model_control)
        
        
        
        #array append
        w1_array.append(parameter_vector[0])
        w2_array.append(parameter_vector[1])
        w3_array.append(parameter_vector[2])
        w4_array.append(parameter_vector[3])
       
        error_array.append(error)
        new_model_controlarray.append(new_model_control)
        predicted_distancearray.append(predicted_distance)
        current_control_array.append(current_control)
        
    
        vel_array.append(local_vel)
        desired_distance_array.append(desired_distance)
        model_regout_array.append(linearmodel_regout)
         
    #plot predicted and smoothed distance, desired distance
    total_shift = 26
    
    plt.plot(np.arange(iteration)[100::],np.array(predicted_distancearray)[100::].flatten())
    plt.plot(np.arange(iteration)[100:int(0.8*iteration)],np.array(predicted_distancearray)[100+total_shift:int(0.8*iteration)+total_shift].flatten())
    plt.plot(np.arange(iteration),np.array(smoothed_dist).flatten())
    plt.plot(np.arange(iteration),np.array(desired_distance_array).flatten())
    plt.legend(['predicted distance','shifted prediction','smoothed'])
    plt.title('Distance graph with polynomial model')
    plt.show()
    #plot modelled control output and current control output
    
    plt.plot(np.arange(iteration-batchsize)[20::],np.array(current_control_array).flatten()[20::])
    plt.plot(np.arange(iteration-batchsize)[20::],np.array(model_regout_array).flatten()[20::])
    plt.title('The comparison between real and fitted velocity with polynomial model')
    plt.xlabel('iteration')
    plt.legend(['real','model'])
    plt.show()
    
    plt.plot(np.arange(iteration-batchsize)[total_shift::],1e5*np.array(vel_array).flatten()[total_shift::])
   
    plt.plot(np.arange(iteration-batchsize)[total_shift::],1e5*np.array(new_model_controlarray).flatten()[total_shift::])
    plt.plot(np.arange(iteration-batchsize)[total_shift::],1e5*np.array(polynomial_model_regout_array).flatten()[total_shift::])
    plt.xlabel('iteration')
    plt.legend(['real','linear model','polynomial model'])
    plt.title('The comparison between real and estimated control output after starting model control')
    plt.show()
    
    #plot shifted control output vs velocity data
    plt.scatter(np.array(current_control_array)[20::].flatten(),np.array(vel_array)[20::].flatten())
    plt.scatter(np.array(model_regout_array)[20::].flatten(),np.array(vel_array)[20::].flatten())
    plt.xlabel('iteration')
    plt.legend(['real','model'])
    plt.show()
    
    #plot parameter update
    plt.plot(np.arange(iteration),np.array(w1_array).flatten())
    plt.plot(np.arange(iteration-batchsize),np.array(w2_array).flatten())
    plt.plot(np.arange(iteration),np.array(w3_array).flatten())
    plt.plot(np.arange(iteration),np.array(w4_array).flatten())
    plt.xlabel('iteration')
    plt.legend(['$w_1$','$w_2$','$w_3$','$w_4$'])
    plt.show()
    #plot error
    
    plt.plot(np.arange(iteration-batchsize)[0:],np.array(error_array).flatten()[0:])
    plt.xlabel('iteration')
    plt.legend(['normalized error'])
    plt.show()
if  __name__ == '__main__':
 #   [ 1.14099490e-13 -1.13033019e-08  3.74847769e-04 -4.16062874e+00]


    #initialize parameters
    initial_w = [ 0,0,2.76566664,0.33001263]          #initial parameters for polynomial model
    learning_rate = 0.34                              # learning rate for SGD update
    iteration = 950                                   # number of iteration for running algorithm
    batchsize = 0                 
    total_shift = 82                                  # time delay 
    predicted_distance = 0                            #initial predicted distance
    counter = 0                 
    step_startcontrol = 800                           #step to start the control
    op_amplitude  = 1                                 # vibration amplitude 
    op_freq = 10                                      # vibration frequency
    sampling_freq = 5000                              #sampling frequency
    op_back_vel = 0                                   #background velocity
    
    #get control and velocity data 
    velocity_data  = 1e-5*np.array(raw_measurementdata['Stellwert_Ventil'][4070+82:4070+iteration+82])
    control_data = 1*np.array(raw_measurementdata['vel_local'][4070:4070+iteration])
   # smoothed distance 
    smoothed_dist = np.array(raw_measurementdata['smoothed_dist'][4070:4070+iteration])
    
     
    para = np.polyfit(control_data,velocity_data,1)
   #para[0]*velocity_data**3 + para[1]*velocity_data**2 + 
    fitted =  para[0]*control_data +  para[1]
    
    
      #velocity_data =  np.array((desired_dist[1000:1000+iteration]-smoothed_dist[1000:1000+iteration])/40)
    #start training and fitting with data
    data_fitting_model_main(counter,step_startcontrol,op_amplitude,op_freq, sampling_freq, op_back_vel, predicted_distance,batchsize,initial_w,iteration,learning_rate,control_data,velocity_data,smoothed_dist)
    
    


    
    
    
   
